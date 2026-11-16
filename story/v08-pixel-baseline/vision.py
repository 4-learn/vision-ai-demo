"""影像 AI 課進度存檔 v08：更強的偵測器，裝到現場。

執行（在這個資料夾裡）：
    LINE=factory python vision.py     # 工廠線：螺帽瑕疵、彎腰搬重物
    LINE=care python vision.py        # 健康線：寢室畫面異常、跌倒
"""
import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")   # ultralytics 不要印進度條

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2 as T
from sklearn.metrics import roc_auc_score

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
DATA = STORY / "data" / LINE
ANOMALY = DATA / "anomaly"
NEW_EVENT = {"factory": "defect_detected", "care": "room_anomaly"}[LINE]
torch.manual_seed(0)
torch.set_num_threads(4)


def image_paths(split, kind="*"):
    return sorted((ANOMALY / split).glob(f"{kind}/*.png"))


def test_set():
    """測試集：路徑、類型（good 或瑕疵類型）、答案（0 正常、1 異常）。"""
    paths = image_paths("test")
    kinds = [p.parent.name for p in paths]
    return paths, kinds, np.array([0 if k == "good" else 1 for k in kinds])


def mask_of(path):
    """瑕疵遮罩（白＝異常的位置）；正常圖回傳全黑。"""
    m = ANOMALY / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    return cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) if m.exists() else np.zeros((256, 256), np.uint8)


def clahe(gray, clip=2.0, grid=8):
    """CLAHE：每一小區自己做直方圖等化，對比不准拉超過 clip。"""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def contrast(gray, mask):
    """「東西」和「背景」的亮度差：瑕疵或異常要看得出來，這個數字要大。"""
    inside, outside = gray[mask > 0], gray[mask == 0]
    return float(abs(inside.mean() - outside.mean())) if len(inside) else 0.0


def denoise(gray, h=10):
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def sharpen(gray, amount=1.0, sigma=2.0):
    """unsharp mask：原圖 +（原圖 − 模糊圖）× amount。"""
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def main_axis_angle(gray):
    """前景（比背景亮的部分）的主軸角度：用 cv2.moments 算二階矩。"""
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.moments(fg, binaryImage=True)
    return 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def best_rotation(gray, reference, step=3):
    """每隔 step 度轉一次，找出和參考圖最像（平均像素差最小）的角度。"""
    small, ref = cv2.resize(gray, (64, 64)).astype(np.float32), cv2.resize(reference, (64, 64)).astype(np.float32)
    diffs = [(np.abs(rotate(small, a) - ref).mean(), a) for a in range(0, 360, step)]
    return min(diffs)[1]


def edge_density(gray, low=50, high=150):
    """Canny 邊緣占整張圖的比例。"""
    return float((cv2.Canny(gray, low, high) > 0).mean())


def shape_stats(gray):
    """最大前景輪廓的面積、周長、圓度（4π×面積÷周長²，正圓是 1）。"""
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=10, maxRadius=60)
    hole = float(circles[0, 0, 2]) if circles is not None else 0.0
    return {"area": area, "perimeter": perim, "circularity": 4 * np.pi * area / perim ** 2, "hole_radius": hole}


def memory_calib_split(paths, frac=0.8, seed=0):
    """良品切兩份：80% 當「記憶」、20% 當「校正」（定門檻用，不碰測試集）。"""
    order = np.random.default_rng(seed).permutation(len(paths))
    cut = int(len(paths) * frac)
    return [paths[i] for i in order[:cut]], [paths[i] for i in order[cut:]]


RULE_FEATURES = ["edge_density", "area", "perimeter", "circularity", "hole_radius"]


def rule_table(paths):
    rows = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        rows.append({"edge_density": edge_density(g), **shape_stats(g)})
    return pd.DataFrame(rows)


def rule_baseline(train, test):
    """規則：任何一個量測超出良品的範圍（最小～最大）就判異常。"""
    lo, hi = train.min(), train.max()
    return ((test < lo) | (test > hi)).any(axis=1).astype(int)


AUGMENT = T.Compose([T.RandomRotation(degrees=180), T.ColorJitter(brightness=0.3, contrast=0.3)])
EVAL = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def small_gray(path, size=64):
    return cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (size, size)).astype(np.float32) / 255


def pixel_scores(memory, paths):
    mean = np.mean([small_gray(p) for p in memory], axis=0)
    return np.array([np.abs(small_gray(p) - mean).mean() for p in paths])


def main():
    memory, calib = memory_calib_split(image_paths("train", "good"))
    paths, kinds, y = test_set()
    s, c = pixel_scores(memory, paths), pixel_scores(memory, calib)
    thr = c.max()
    pred = s > thr
    print(f"== {LINE}：像素差 baseline（和「平均良品」比）==")
    print(f"記憶 {len(memory)} 張、校正 {len(calib)} 張、測試 {len(paths)} 張")
    print(f"AUROC {roc_auc_score(y, s):.3f}；門檻＝校正良品最大值 {thr:.4f} → 抓到 {int(pred[y == 1].sum())}/{int(y.sum())}、"
          f"誤判 {int(pred[y == 0].sum())}")


if __name__ == "__main__":
    main()

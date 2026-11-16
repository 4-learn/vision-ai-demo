"""影像 AI 課進度存檔 v04：更強的偵測器，裝到現場。

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

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
DATA = STORY / "data" / LINE
ANOMALY = DATA / "anomaly"
NEW_EVENT = {"factory": "defect_detected", "care": "room_anomaly"}[LINE]


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


def main():
    paths, kinds, y = test_set()
    df = pd.DataFrame({"type": kinds, "edge_density": [edge_density(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]})
    print(f"== {LINE}：Canny 邊緣占整張圖的比例（依類型）==")
    print(df.groupby("type")["edge_density"].agg(["count", "mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()

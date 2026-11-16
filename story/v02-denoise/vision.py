"""影像 AI 課進度存檔 v02：更強的偵測器，裝到現場。

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


def main():
    paths, kinds, y = test_set()
    g = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    rng = np.random.default_rng(0)
    noisy = np.clip(g.astype(np.float32) + rng.normal(0, 15, g.shape), 0, 255).astype(np.uint8)
    print(f"== {LINE}：加上已知雜訊（σ=15），再看哪種去噪救得回來（PSNR，越大越像原圖）==")
    for name, out in [("雜訊圖", noisy), ("GaussianBlur 5×5", cv2.GaussianBlur(noisy, (5, 5), 0)),
                      ("medianBlur 5", cv2.medianBlur(noisy, 5)), ("fastNlMeans h=10", denoise(noisy, 10)),
                      ("fastNlMeans h=30", denoise(noisy, 30))]:
        print(f"{name:<18}PSNR {cv2.PSNR(g, out):.2f} dB")
    s = sharpen(g)
    print(f"\n銳化前後的邊緣強度（Laplacian 標準差）：{cv2.Laplacian(g, cv2.CV_64F).std():.1f} → {cv2.Laplacian(s, cv2.CV_64F).std():.1f}")


if __name__ == "__main__":
    main()

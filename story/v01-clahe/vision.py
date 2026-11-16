"""影像 AI 課進度存檔 v01：更強的偵測器，裝到現場。

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


def main():
    paths, kinds, y = test_set()
    print(f"== {LINE}：測試集 {len(paths)} 張（異常 {int(y.sum())} 張）==")
    rows = []
    for p, k in zip(paths, kinds):
        if k == "good":
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        m = mask_of(p)
        rows.append({"type": k, "原圖對比": contrast(g, m), "equalizeHist": contrast(cv2.equalizeHist(g), m),
                     "CLAHE": contrast(clahe(g), m)})
    print("異常位置和背景的亮度差（依類型平均，越大越容易看出來）：")
    print(pd.DataFrame(rows).groupby("type").mean().round(1).to_string())


if __name__ == "__main__":
    main()

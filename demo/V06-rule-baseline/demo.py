"""V06｜OpenCV 量測＋規則：只看良品，量出「正常的範圍」，超出就判異常。這是之後每個模型都要贏過的 baseline。

執行（在 repo 根目錄）：python demo/V06-rule-baseline/demo.py
工廠線的螺帽照片要先由教師執行 tools/build_data.py 產生（MVTec AD 衍生，不隨 repo 散佈）。約 20 秒。
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "story" / "data"


def measure(path):
    """V04、V05 的量測：邊緣比例、最大輪廓的面積／周長／圓度、孔的半徑。"""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    _, fg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    circles = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=10, maxRadius=60)
    return {"edge_density": float((cv2.Canny(g, 50, 150) > 0).mean()), "area": area, "perimeter": perim,
            "circularity": 4 * np.pi * area / perim ** 2,
            "hole_radius": float(circles[0, 0, 2]) if circles is not None else 0.0}


def table(paths):
    return pd.DataFrame([measure(p) for p in paths])


def outside(df, lo, hi):
    """每一列、每一個量測：有沒有超出 [lo, hi]。"""
    return (df < lo) | (df > hi)


goods = sorted((DATA / "factory/anomaly/train/good").glob("*.png"))
order = np.random.default_rng(0).permutation(len(goods))
cut = int(len(goods) * 0.8)
memory = table([goods[i] for i in order[:cut]])
calib = table([goods[i] for i in order[cut:]])

print("== 1. 良品切兩份：記憶 80%、校正 20%（seed 0）==")
print(f"良品 {len(goods)} 張 → 記憶 {len(memory)}、校正 {len(calib)}")

print("\n== 2. 記憶裡的良品：每個量測的範圍 ==")
summary = memory.describe().loc[["min", "50%", "max"]]
print(summary.round(3).to_string())

print("\n== 3. 規則：任何一個量測超出 [最小, 最大] 就判異常 ==")
lo, hi = memory.min(), memory.max()
flag = (calib < lo) | (calib > hi)
flagged = flag.any(axis=1).sum()
print(f"校正的 {len(calib)} 張良品，被判異常 {int(flagged)} 張")
print("是哪一個量測超出：", flag.sum().to_dict())

test_paths = sorted((DATA / "factory/anomaly/test").glob("*/*.png"))
test = table(test_paths)
kinds = pd.Series([p.parent.name for p in test_paths], name="type")
flag = outside(test, lo, hi)
print("\n== 4. 拆開看：每個量測單獨用，各抓到哪些類型 ==")
print(flag.groupby(kinds).sum().to_string())
print("任何一個超出就判：", flag.any(axis=1).groupby(kinds).sum().to_dict())

print("\n== 5. 範圍定多寬：min～max、1%～99%、平均 ± 3 個標準差 ==")
ranges = {"min～max": (memory.min(), memory.max()),
          "1%～99%": (memory.quantile(0.01), memory.quantile(0.99)),
          "平均±3σ": (memory.mean() - 3 * memory.std(), memory.mean() + 3 * memory.std())}
for name, (l, h) in ranges.items():
    hit = outside(test, l, h).any(axis=1)
    bad = kinds != "good"
    print(f"{name:<9}校正良品誤判 {int(outside(calib, l, h).any(axis=1).sum()):>2}/{len(calib)}；"
          f"test 抓到異常 {int(hit[bad].sum()):>2}/{int(bad.sum())}、良品誤判 {int(hit[~bad].sum())}/{int((~bad).sum())}")

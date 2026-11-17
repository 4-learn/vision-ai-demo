"""V05｜OpenCV：霍夫與量測。contourArea、arcLength 量輪廓，算圓度；HoughCircles 找圓（螺帽中間的孔）。

執行（在 repo 根目錄）：python demo/V05-hough/demo.py
工廠線的螺帽照片要先由教師執行 tools/build_data.py 產生（MVTec AD 衍生，不隨 repo 散佈）。
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "story" / "data"


def read_gray(rel):
    img = cv2.imread(str(DATA / rel), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(rel)
    return img


def measure(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / perim ** 2
    return area, perim, circularity


print("== 1. contourArea、arcLength：畫三個形狀來量 ==")
shapes = {}
shapes["圓 r=60"] = cv2.circle(np.zeros((200, 200), np.uint8), (100, 100), 60, 255, -1)
shapes["正方形 邊 106"] = cv2.rectangle(np.zeros((200, 200), np.uint8), (47, 47), (152, 152), 255, -1)
star = np.array([[100 + r * np.cos(t), 100 + r * np.sin(t)]
                 for t, r in zip(np.linspace(0, 2 * np.pi, 16, endpoint=False), [80, 40] * 8)], np.int32)
shapes["八角星"] = cv2.fillPoly(np.zeros((200, 200), np.uint8), [star], 255)
print(f"{'形狀':<12}{'面積':>9}{'周長':>9}{'圓度':>7}")
for name, mask in shapes.items():
    area, perim, circ = measure(mask)
    print(f"{name:<12}{area:9.0f}{perim:9.1f}{circ:7.3f}")
print(f"（理論值：圓面積 π×60² = {np.pi * 60 ** 2:.0f}、周長 2π×60 = {2 * np.pi * 60:.1f}）")

print("\n== 2. arcLength 的 closed：要不要把頭尾接起來 ==")
contours, _ = cv2.findContours(shapes["正方形 邊 106"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)
print(f"closed=True {cv2.arcLength(c, True):.0f}；closed=False {cv2.arcLength(c, False):.0f}（少了最後一邊）")

print("\n== 3. 螺帽：良品和翻面（flip）==")
for rel in ["factory/anomaly/train/good/000.png", "factory/anomaly/test/flip/000.png"]:
    g = read_gray(rel)
    _, fg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    area, perim, circ = measure(fg)
    print(f"{rel.split('anomaly/')[1]:<22}面積 {area:7.0f}、周長 {perim:6.1f}、圓度 {circ:.3f}")

print("\n== 4. cv2.HoughCircles：找螺帽中間的孔 ==")
g = read_gray("factory/anomaly/train/good/000.png")
circles = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                           param1=100, param2=30, minRadius=10, maxRadius=60)
x, y, r = circles[0, 0]
print(f"回傳 shape {circles.shape}（1 組 × 找到幾個圓 × (x, y, r)）")
print(f"第一個圓：中心 ({x:.1f}, {y:.1f})、半徑 {r:.1f}")

print("\n== 5. param2（累加門檻）：越小越容易「看到」圓 ==")
for p2 in [20, 30, 100, 150]:
    c = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                         param1=100, param2=p2, minRadius=10, maxRadius=60)
    found = [] if c is None else [round(float(v[2]), 1) for v in c[0]]
    print(f"param2={p2:<4}找到 {len(found)} 個，半徑 {found}")

print("\n== 6. minDist、半徑範圍：限制在哪裡找 ==")
for label, kw in [("minDist=100、半徑 10–60 ", dict(minDist=100, minRadius=10, maxRadius=60)),
                  ("minDist=10 、半徑 10–60 ", dict(minDist=10, minRadius=10, maxRadius=60)),
                  ("minDist=100、半徑 60–120", dict(minDist=100, minRadius=60, maxRadius=120))]:
    c = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, dp=1.2, param1=100, param2=20, **kw)
    found = [] if c is None else [round(float(v[2]), 1) for v in c[0]]
    print(f"{label}（param2=20）找到 {len(found):>3} 個，前 5 個半徑 {found[:5]}")


def hole_radius(rel):
    c = cv2.HoughCircles(cv2.medianBlur(read_gray(rel), 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                         param1=100, param2=30, minRadius=10, maxRadius=60)
    return float(c[0, 0, 2]) if c is not None else 0.0


print("\n== 7. 同一個設定量 220 張良品：孔的半徑 ==")
radii = np.array([hole_radius(p.relative_to(DATA)) for p in sorted((DATA / "factory/anomaly/train/good").glob("*.png"))])
print(f"最小 {radii.min():.1f}、中位數 {np.median(radii):.1f}、最大 {radii.max():.1f}")
print(f"半徑 < 35 的 {int((radii < 35).sum())} 張、35–50 的 {int(((radii >= 35) & (radii < 50)).sum())} 張、≥ 50 的 {int((radii >= 50).sum())} 張")
flips = np.array([hole_radius(p.relative_to(DATA)) for p in sorted((DATA / "factory/anomaly/test/flip").glob("*.png"))])
print(f"翻面 {len(flips)} 張：最小 {flips.min():.1f}、中位數 {np.median(flips):.1f}、最大 {flips.max():.1f}")

print("\n== 8. 畫出來看：同一個設定，有時抓到孔、有時抓到外圈 ==")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
panels = []
for name in ["000.png", "008.png"]:
    color = cv2.imread(str(DATA / "factory/anomaly/train/good" / name))
    c = cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, dp=1.2,
                         minDist=100, param1=100, param2=30, minRadius=10, maxRadius=60)
    x, y, r = c[0, 0]
    cv2.circle(color, (int(round(x)), int(round(y))), int(round(r)), (0, 0, 255), 2)
    panels.append(color)
    print(f"{name}：中心 ({x:.1f}, {y:.1f})、半徑 {r:.1f}")
print("imwrite out/hough.png →", cv2.imwrite(str(OUT / "hough.png"), np.hstack(panels)))

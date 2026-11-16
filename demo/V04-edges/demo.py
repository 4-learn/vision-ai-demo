"""V04｜OpenCV：找邊緣。Sobel 算梯度（亮度變化多快、往哪個方向），Canny 用兩個門檻把梯度變成細細的邊緣線。

執行（在 repo 根目錄）：python demo/V04-edges/demo.py
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


def density(edges):
    return float((edges > 0).mean())


print("== 1. cv2.Sobel：一條垂直的亮暗交界 ==")
step = np.zeros((5, 8), np.uint8)
step[:, 4:] = 200                                       # 左黑右亮
gx = cv2.Sobel(step, cv2.CV_64F, 1, 0, ksize=3)         # 對 x 微分：左右變化
gy = cv2.Sobel(step, cv2.CV_64F, 0, 1, ksize=3)         # 對 y 微分：上下變化
print("原圖第 2 列：", step[2].tolist())
print("Sobel x 第 2 列：", gx[2].astype(int).tolist())
print("Sobel y 第 2 列：", gy[2].astype(int).tolist())
back = cv2.Sobel(255 - step, cv2.CV_64F, 1, 0, ksize=3)
print("左亮右黑時 Sobel x：", back[2].astype(int).tolist())
print("同一張用 CV_8U：", cv2.Sobel(255 - step, cv2.CV_8U, 1, 0, ksize=3)[2].tolist(), "← 負數全變 0")

print("\n== 2. 螺帽的梯度：x、y 合起來看強度 ==")
nut = read_gray("factory/anomaly/train/good/000.png")
gx = cv2.Sobel(nut, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(nut, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(gx ** 2 + gy ** 2)
print(f"Sobel x 範圍 {gx.min():.0f} ～ {gx.max():.0f}（有正有負）")
print(f"梯度強度：平均 {mag.mean():.1f}、中位數 {np.median(mag):.1f}、前 1% {np.percentile(mag, 99):.1f}")
print(f"黑色背景的梯度強度平均 {mag[nut < 40].mean():.1f}；螺帽上 {mag[nut >= 40].mean():.1f}")

print("\n== 3. cv2.Canny：兩個門檻（低、高）==")
for low, high in [(20, 60), (50, 150), (100, 200), (150, 300)]:
    print(f"Canny({low:>3}, {high:>3})：邊緣占 {density(cv2.Canny(nut, low, high)):.4f}")
print(f"只用一個門檻 150（梯度 > 150 的都算）：占 {float((mag > 150).mean()):.4f}")

print("\n== 4. 低門檻的作用：把斷掉的邊接起來 ==")
for low in [50, 100, 150]:
    e = cv2.Canny(nut, low, 150)
    n, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print(f"Canny({low:>3}, 150)：邊緣占 {density(e):.4f}，分成 {len(n):>3} 段")

print("\n== 5. 刮痕：邊緣在瑕疵裡比較多，但整張圖看不出來 ==")
img = read_gray("factory/anomaly/test/scratch/000.png")
mask = read_gray("factory/anomaly/ground_truth/scratch/000_mask.png")
e = cv2.Canny(img, 50, 150) > 0
print(f"刮痕位置（占整張 {(mask > 0).mean():.3f}）裡的邊緣比例 {e[mask > 0].mean():.3f}；其他地方 {e[mask == 0].mean():.3f}")
print(f"整張圖：刮痕 {e.mean():.4f}，良品 {density(cv2.Canny(nut, 50, 150)):.4f}")

print("\n== 6. 健康線：夜間畫面，門檻太低就把雜訊當邊緣 ==")
for name in ["027.png", "002.png"]:
    g = read_gray(f"care/anomaly/train/good/{name}")
    row = "；".join(f"({lo}, {hi}) {density(cv2.Canny(g, lo, hi)):.4f}" for lo, hi in [(10, 30), (20, 60), (50, 150)])
    print(f"{name}（平均亮度 {g.mean():5.1f}）{row}")

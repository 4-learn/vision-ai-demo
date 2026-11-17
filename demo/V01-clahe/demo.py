"""V01｜OpenCV：打光不均。calcHist 看亮度分布、equalizeHist 全域拉開、createCLAHE 一小區一小區拉開。

執行（在 repo 根目錄）：python demo/V01-clahe/demo.py
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


def hist8(gray):
    """亮度分成 8 段（每段 32），每段占幾 %。"""
    h = cv2.calcHist([gray], [0], None, [8], [0, 256]).ravel()
    return " ".join(f"{v:4.1f}" for v in h / h.sum() * 100)


def contrast(gray, mask):
    inside, outside = gray[mask > 0], gray[mask == 0]
    return abs(float(inside.mean()) - float(outside.mean()))


nut = read_gray("factory/anomaly/test/scratch/000.png")
scratch = read_gray("factory/anomaly/ground_truth/scratch/000_mask.png")
background = nut < 40                     # 螺帽外面的黑色背景

print("== 1. cv2.calcHist：亮度分布（8 段，每段 32）==")
h = cv2.calcHist([nut], [0], None, [8], [0, 256])
pct = h.ravel() / h.sum() * 100
print(f"回傳 shape {h.shape}、dtype {h.dtype}；總數 {int(h.sum())} = 256×256")
print("各段占比 %：", " ".join(f"{v:4.1f}" for v in pct))
print(f"平均亮度 {nut.mean():.1f}；最暗 {nut.min()}、最亮 {nut.max()}")

print("\n== 2. cv2.equalizeHist：整張圖一起拉開 ==")
eq = cv2.equalizeHist(nut)
print("等化後各段占比 %：", hist8(eq))
print(f"平均亮度 {nut.mean():.1f} → {eq.mean():.1f}")
print(f"刮痕和周圍的亮度差 {contrast(nut, scratch):.1f} → {contrast(eq, scratch):.1f}")
print(f"黑色背景的起伏（標準差）{nut[background].std():.1f} → {eq[background].std():.1f}")

print("\n== 3. cv2.createCLAHE：一小區一小區拉開，clipLimit 限制拉多少 ==")
print("方法            刮痕亮度差  背景起伏")
print(f"{'原圖':<14}{contrast(nut, scratch):10.1f}{nut[background].std():10.1f}")
print(f"{'equalizeHist':<14}{contrast(eq, scratch):10.1f}{eq[background].std():10.1f}")
for clip in [1.0, 2.0, 4.0, 8.0]:
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(nut)
    print(f"{'CLAHE clip=' + str(clip):<14}{contrast(out, scratch):10.1f}{out[background].std():10.1f}")

print("\n== 4. 打光不均：左邊暗、右邊亮，兩邊各有一塊只亮 6 的污點 ==")
rng = np.random.default_rng(0)
demo = np.tile(np.linspace(30, 220, 256), (256, 1))
demo[100:140, 30:60] += 6
demo[100:140, 196:226] += 6
demo = np.clip(demo + rng.normal(0, 2, demo.shape), 0, 255).astype(np.uint8)


def spot_contrast(img, x1, x2):
    """污點和它周圍一圈的亮度差。"""
    around = img[80:160, x1 - 10:x2 + 10].astype(float)
    inside = np.zeros(around.shape, bool)
    inside[20:60, 10:-10] = True
    return abs(around[inside].mean() - around[~inside].mean())


for name, out in [("原圖", demo), ("equalizeHist", cv2.equalizeHist(demo)),
                  ("CLAHE clip=2", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(demo)),
                  ("CLAHE clip=4", cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(demo))]:
    print(f"{name:<13}暗邊污點差 {spot_contrast(out, 30, 60):5.1f}；亮邊污點差 {spot_contrast(out, 196, 226):5.1f}")

print("\n== 5. 健康線：同一個房間，白天和夜間 ==")
for rel in ["care/anomaly/train/good/027.png", "care/anomaly/train/good/002.png"]:
    g = read_gray(rel)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    print(f"{Path(rel).name}：平均亮度 {g.mean():5.1f}，各段 % {hist8(g)}")
    print(f"{'':9}CLAHE 後 {c.mean():5.1f}，各段 % {hist8(c)}")

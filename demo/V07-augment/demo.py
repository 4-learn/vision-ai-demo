"""V07｜torchvision.transforms.v2：資料增強，以及哪些增強不能用。"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T

torch.manual_seed(0)
FACTORY = Path("story/data/factory/anomaly")
CARE = Path("story/data/care/anomaly")

print("== 1. ToImage＋ToDtype：PIL 圖 → 0～1 的 float tensor ==")
img = Image.open(FACTORY / "train/good/005.png").convert("RGB")
x = T.ToImage()(img)
x = T.ToDtype(torch.float32, scale=True)(x)
EVAL = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
u8 = T.ToImage()(img)
print(f"PIL 圖：size {img.size}、mode {img.mode}")
print(f"ToImage 之後：shape {tuple(u8.shape)}、dtype {u8.dtype}、範圍 {int(u8.min())}～{int(u8.max())}")
print(f"ToDtype(scale=True) 之後：dtype {x.dtype}、範圍 {x.min():.3f}～{x.max():.3f}")

print("\n== 2. Compose：把幾個步驟串成一個 ==")
print(EVAL)
print(f"EVAL(img) 和上面一步一步做的結果相同：{torch.equal(EVAL(img), x)}")

print("\n== 3. RandomRotation、ColorJitter：每呼叫一次，結果都不一樣 ==")
AUGMENT = T.Compose([T.RandomRotation(degrees=180), T.ColorJitter(brightness=0.3, contrast=0.3)])
for i in range(4):
    o = EVAL(AUGMENT(img))
    print(f"第 {i + 1} 次：平均亮度 {o.mean():.3f}，和原圖的平均差 {(o - x).abs().mean():.3f}")

print("\n== 4. torch.manual_seed：固定亂數，才重現得了 ==")
torch.manual_seed(42)
a = EVAL(AUGMENT(img))
torch.manual_seed(42)
b = EVAL(AUGMENT(img))
c = EVAL(AUGMENT(img))
print(f"同一個 seed 再做一次，結果相同：{torch.equal(a, b)}；不重設 seed 接著做，相同：{torch.equal(b, c)}")


def silhouette(pil):
    """螺帽的輪廓（前景＝1）：縮成 64×64，用 Otsu 自動切亮暗。"""
    g = cv2.resize(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2GRAY), (64, 64))
    return cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1].astype(np.float32)


def best_iou(a, ref):
    """把 a 每隔 3 度轉一次，和 ref 重疊得最好的那一次（交集÷聯集）。"""
    best = 0.0
    for angle in range(0, 360, 3):
        M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
        r = cv2.warpAffine(a, M, (64, 64))
        best = max(best, np.minimum(r, ref).sum() / np.maximum(r, ref).sum())
    return best


print("\n== 5. 工廠線：轉一轉找得回來，左右翻轉找不回來 ==")
goods = [Image.open(p).convert("RGB") for p in sorted((FACTORY / "train/good").glob("*.png"))[:10]]
flips = [Image.open(p).convert("RGB") for p in sorted((FACTORY / "test/flip").glob("*.png"))[:10]]
torch.manual_seed(0)
rows = {"RandomRotation(180)": [], "RandomHorizontalFlip(p=1)": []}
for g in goods:
    ref = silhouette(g)
    rows["RandomRotation(180)"].append(best_iou(silhouette(T.RandomRotation(180)(g)), ref))
    rows["RandomHorizontalFlip(p=1)"].append(best_iou(silhouette(T.RandomHorizontalFlip(p=1)(g)), ref))
for name, v in rows.items():
    print(f"{name}：轉回去和原圖最多重疊 {np.mean(v):.3f}")


def nearest(a, refs):
    return np.mean([max(best_iou(silhouette(x), r) for r in refs) for x in a])


good_refs = [silhouette(g) for g in goods[:5]]
flip_refs = [silhouette(f) for f in flips[:5]]
print("和誰比較像（轉到最好的角度後的重疊，越大越像）：")
for name, group in [("另外 5 張良品", goods[5:]), ("那 5 張良品左右翻轉", [T.RandomHorizontalFlip(p=1)(g) for g in goods[5:]]),
                    ("另外 5 張真的 flip 瑕疵", flips[5:])]:
    print(f"  {name}：像良品 {nearest(group, good_refs):.3f}、像 flip 瑕疵 {nearest(group, flip_refs):.3f}")

print("\n== 6. 健康線：鏡頭固定，翻轉、旋轉做出來的是現場不會有的畫面 ==")
care = [Image.open(p).convert("RGB") for p in sorted((CARE / "train/good").glob("*.png"))[:40]]
memory, query = care[:30], care[30:]
gray = lambda pil: cv2.resize(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float32)
mem = np.stack([gray(m) for m in memory])
far = lambda pils: np.mean([np.abs(mem - gray(p)).mean((1, 2)).min() for p in pils])
torch.manual_seed(0)
for name, pils in [("良品原圖", query), ("RandomHorizontalFlip(p=1)", [T.RandomHorizontalFlip(p=1)(q) for q in query]),
                   ("RandomRotation(180)", [T.RandomRotation(180)(q) for q in query]),
                   ("ColorJitter(0.3, 0.3)", [T.ColorJitter(0.3, 0.3)(q) for q in query])]:
    print(f"{name}：和最像的良品平均差 {far(pils):.1f}")
for kind in ["fallen", "obstacle", "blocked"]:
    pils = [Image.open(p).convert("RGB") for p in sorted((CARE / "test" / kind).glob("*.png"))[:10]]
    print(f"真的異常 {kind}：和最像的良品平均差 {far(pils):.1f}")

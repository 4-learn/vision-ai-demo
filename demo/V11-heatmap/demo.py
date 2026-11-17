"""V11｜OpenCV 熱圖：28×28 的距離圖放大、轉成 0–255、上色、疊回原圖，並檢查最熱的點在不在異常上。"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

torch.set_num_threads(4)
OUT = Path("demo/V11-heatmap/out")
OUT.mkdir(parents=True, exist_ok=True)

print("== 1. cv2.resize：4×4 放大成 8×8，兩種補法 ==")
small = np.zeros((4, 4), np.float32)
small[1, 2] = 8
print("原本：")
print(small.astype(int))
big = cv2.resize(small, (8, 8), interpolation=cv2.INTER_NEAREST)
print(f"INTER_NEAREST（第 3 列）：{big[2].round(1).tolist()}")
big = cv2.resize(small, (8, 8), interpolation=cv2.INTER_LINEAR)
print(f"INTER_LINEAR（第 3 列）：{big[2].round(1).tolist()}")

print("\n== 2. cv2.normalize：任何範圍 → 0～255 ==")
d = np.array([[2.1, 3.5], [4.8, 6.0]], np.float32)
n = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print(f"距離 {[round(float(v), 1) for v in d.ravel()]} → {n.ravel().tolist()}")
d2 = d + 10
n2 = cv2.normalize(d2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print(f"全部加 10：{[round(float(v), 1) for v in d2.ravel()]} → {n2.ravel().tolist()}（一樣！每張圖各自拉，熱圖看不出整張多異常）")

print("\n== 3. cv2.applyColorMap：0～255 的灰階 → 藍到紅 ==")
ramp = np.array([[0, 64, 128, 192, 255]], np.uint8)
color = cv2.applyColorMap(ramp, cv2.COLORMAP_JET)
for v, c in zip(ramp[0], color[0]):
    print(f"{v:>3} → BGR {c.tolist()}")

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
extractor = create_feature_extractor(resnet18(weights=WEIGHTS).eval(), {"layer2": "l2", "layer3": "l3"})
pre = WEIGHTS.transforms()


def patch_features(paths, batch=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            f = extractor(torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]))
            l3 = torch.nn.functional.interpolate(f["l3"], size=f["l2"].shape[-2:], mode="bilinear")
            out.append(torch.cat([f["l2"], l3], 1))
    return torch.cat(out).numpy()


def distance_maps(memory, paths):
    Fm, Ft = patch_features(memory), patch_features(paths)
    C = Fm.shape[1]
    bank = NearestNeighbors(n_neighbors=1).fit(Fm.transpose(0, 2, 3, 1).reshape(-1, C))
    return bank.kneighbors(Ft.transpose(0, 2, 3, 1).reshape(-1, C))[0][:, 0].reshape(len(Ft), 28, 28)


root = Path("story/data/care/anomaly")
memory = sorted((root / "train/good").glob("*.png"))[:40]
paths = sorted((root / "test/obstacle").glob("*.png"))[:10] + sorted((root / "test/good").glob("*.png"))[:10]
maps = distance_maps(memory, paths)
masks = [cv2.imread(str(root / "ground_truth/obstacle" / f"{p.stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
         if p.parent.name != "good" else np.zeros((256, 256), np.uint8) for p in paths]

print("\n== 4. 一張真的熱圖（健康線 obstacle/000.png）==")
img = cv2.imread(str(paths[0]))
big = cv2.resize(maps[0], (256, 256), interpolation=cv2.INTER_LINEAR)
norm = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
print(f"距離圖 {maps[0].shape} → resize {big.shape} → normalize {norm.dtype} → applyColorMap {heat.shape}")
print(f"addWeighted(原圖 0.6, 熱圖 0.4)：{overlay.shape}；左上角像素 原圖 {img[0, 0].tolist()}、熱圖 {heat[0, 0].tolist()}、"
      f"疊起來 {overlay[0, 0].tolist()}")
contours, _ = cv2.findContours((masks[0] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
side = cv2.drawContours(overlay.copy(), contours, -1, (255, 255, 255), 1)
cv2.imwrite(str(OUT / "care-obstacle-000.png"), np.hstack([img, heat, side]))
print("已存 out/care-obstacle-000.png（左：原圖、中：熱圖、右：疊圖＋白線是答案）")

print("\n== 5. 最熱的點在哪裡？argmax＋unravel_index ==")
y, x = np.unravel_index(big.argmax(), big.shape)
print(f"argmax 是攤平後的第 {big.argmax()} 個 → unravel_index → 第 {y} 列、第 {x} 行（x={x}, y={y}）")
print(f"遮罩在那個點的值：{masks[0][y, x]}（255＝異常）")
ys, xs = np.nonzero(masks[0])
print(f"答案（遮罩）的範圍：x {xs.min()}～{xs.max()}、y {ys.min()}～{ys.max()}")

print("\n== 6. 「落在附近」：cv2.dilate 把遮罩向外長 16 像素再判斷 ==")
hits_exact, hits_near = [], []
for m, mask, p in zip(maps, masks, paths):
    if p.parent.name == "good":
        continue
    big = cv2.resize(m, (256, 256), interpolation=cv2.INTER_LINEAR)
    y, x = np.unravel_index(big.argmax(), big.shape)
    near = cv2.dilate((mask > 0).astype(np.uint8), np.ones((33, 33), np.uint8))
    hit = near[y, x] > 0
    hits_exact.append(mask[y, x] > 0)
    hits_near.append(hit)
print(f"obstacle 10 張：最熱點正好在遮罩上 {sum(hits_exact)} 張、在 16 像素內 {sum(hits_near)} 張")

print("\n== 7. pixel-level AUROC：每一小格都當一題 ==")
cell_y = np.concatenate([(cv2.resize(mk, (28, 28), interpolation=cv2.INTER_NEAREST) > 0).ravel() for mk in masks])
print(f"{len(paths)} 張 × 28×28 = {cell_y.size} 格，其中異常 {int(cell_y.sum())} 格（{cell_y.mean():.1%}）")
print(f"pixel-level AUROC：{roc_auc_score(cell_y, maps.ravel()):.3f}")
img_score = np.sort(maps.reshape(len(maps), -1), axis=1)[:, -10:].mean(1)
print(f"對照：同 20 張的整張 AUROC（top-10 平均）{roc_auc_score([p.parent.name != 'good' for p in paths], img_score):.3f}")

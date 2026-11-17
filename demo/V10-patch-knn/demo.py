"""V10｜小格記憶庫（PatchCore 簡化版）：每一小格找最像的正常小格，最遠的 10 格平均成一張圖的分數。"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

torch.set_num_threads(4)

print("== 1. 小例子：4×4 的「圖」，每格 1 個特徵 ==")
good = np.array([[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
                 [[1, 2, 2, 1], [1, 2, 2, 1], [3, 4, 4, 3], [3, 4, 4, 3]]], dtype=float)
test = np.array([[1, 1, 2, 2], [1, 9, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=float)
print("測試圖（第 2 列第 2 格是 9，其他格的數字都在良品裡出現過）：")
print(test.astype(int))
print(f"整張平均：良品 {good.mean((1, 2)).tolist()}、測試 {test.mean():.3f} → 差 {abs(test.mean() - good.mean((1, 2))).min():.3f}，看不太出來")

print("\n== 2. reshape(-1, C)：所有良品的所有小格，攤平成一個記憶庫 ==")
bank = good.reshape(-1, 1)
nn = NearestNeighbors(n_neighbors=1).fit(bank)
dist_map = nn.kneighbors(test.reshape(-1, 1))[0][:, 0].reshape(4, 4)
print(f"良品 {good.shape} → 記憶庫 {bank.shape}（{good.shape[0]} 張 × {good.shape[1]}×{good.shape[2]} 格，每格 1 個特徵）")
print("每一格到最像的正常小格的距離：")
print(dist_map.astype(int))

print("\n== 3. 一張圖的分數：max、top-k 平均、整張平均 ==")
flat = np.sort(dist_map.ravel())
print(f"max {flat[-1]:.3f}、top-3 平均 {flat[-3:].mean():.3f}、整張平均 {flat.mean():.3f}")
noisy = np.zeros((4, 4))
noisy[0, 3] = 6
real = np.zeros((4, 4))
real[1:3, 1:3] = 4
for name, m in [("一格雜訊 6", noisy), ("一塊 2×2 的異常，每格 4", real)]:
    f = np.sort(m.ravel())
    print(f"{name}：max {f[-1]:.3f}、top-3 平均 {f[-3:].mean():.3f}")
print("只看 max，一格雜訊會比一整塊異常還可疑；取最遠的幾格平均，比較穩")

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
extractor = create_feature_extractor(resnet18(weights=WEIGHTS).eval(), {"layer2": "l2", "layer3": "l3"})
pre = WEIGHTS.transforms()


def patch_features(paths, batch=16):
    """每張圖 → (384, 28, 28)：layer3 用 interpolate 放大成 28×28，和 layer2 疊在一起。"""
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            f = extractor(torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]))
            l3 = torch.nn.functional.interpolate(f["l3"], size=f["l2"].shape[-2:], mode="bilinear")
            out.append(torch.cat([f["l2"], l3], 1))
    return torch.cat(out).numpy()


print("\n== 4. 真的資料：layer2＋放大的 layer3，每張圖 28×28 格 ==")
root = Path("story/data/factory/anomaly")
memory = sorted((root / "train/good").glob("*.png"))[:60]
kinds_wanted = ["good", "scratch", "flip"]
paths = [p for k in kinds_wanted for p in sorted((root / "test" / k).glob("*.png"))]
kinds = np.array([p.parent.name for p in paths])
Fm, Ft = patch_features(memory), patch_features(paths)
print(f"記憶良品 {Fm.shape}；測試 {Ft.shape}")
C = Fm.shape[1]
all_patches = Fm.transpose(0, 2, 3, 1).reshape(-1, C)
print(f"transpose(0, 2, 3, 1) 把特徵移到最後 → reshape(-1, {C})：記憶庫 {all_patches.shape}")

print("\n== 5. 記憶庫太大就隨機抽：抽多少格，分數差多少 ==")
for size in [2000, 10000, len(all_patches)]:
    idx = np.random.default_rng(0).choice(len(all_patches), size, replace=False)
    bank = NearestNeighbors(n_neighbors=1).fit(all_patches[idx])
    maps = bank.kneighbors(Ft.transpose(0, 2, 3, 1).reshape(-1, C))[0][:, 0].reshape(len(Ft), 28, 28)
    top10 = np.sort(maps.reshape(len(maps), -1), axis=1)[:, -10:].mean(1)
    per = {k: round(roc_auc_score(kinds[(kinds == "good") | (kinds == k)] == k,
                                  top10[(kinds == "good") | (kinds == k)]), 3) for k in ["scratch", "flip"]}
    print(f"記憶庫 {size:>6} 格：AUROC {roc_auc_score(kinds != 'good', top10):.3f}；依類型 {per}")

print("\n== 6. 一張圖的分數怎麼收：max、top-10 平均、整張平均（記憶庫全部） ==")
flat = np.sort(maps.reshape(len(maps), -1), axis=1)
for name, s in [("max", flat[:, -1]), ("top-10 平均", flat[:, -10:].mean(1)), ("整張平均", flat.mean(1))]:
    per = {k: round(roc_auc_score(kinds[(kinds == "good") | (kinds == k)] == k,
                                  s[(kinds == "good") | (kinds == k)]), 3) for k in ["scratch", "flip"]}
    print(f"{name}：AUROC {roc_auc_score(kinds != 'good', s):.3f}；依類型 {per}")
global_bank = NearestNeighbors(n_neighbors=1).fit(Fm.mean((2, 3)))
g = global_bank.kneighbors(Ft.mean((2, 3)))[0][:, 0]
per = {k: round(roc_auc_score(kinds[(kinds == "good") | (kinds == k)] == k,
                              g[(kinds == "good") | (kinds == k)]), 3) for k in ["scratch", "flip"]}
print(f"對照 V09 整張特徵 kNN：AUROC {roc_auc_score(kinds != 'good', g):.3f}；依類型 {per}")

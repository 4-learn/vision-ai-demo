"""V09｜create_feature_extractor＋NearestNeighbors：用預訓練網路的特徵，找最像的良品。"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

torch.set_num_threads(4)
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=WEIGHTS).eval()

print("== 1. get_graph_node_names：網路裡每一步叫什麼名字 ==")
_, nodes = get_graph_node_names(model)
print(f"共 {len(nodes)} 個節點；前 5 個 {nodes[:5]}")
print("layer2 底下的最後 3 個：", [n for n in nodes if n.startswith("layer2")][-3:])
print("最後 3 個：", nodes[-3:])
print("return_nodes 寫 \"layer2\"，就是拿 layer2 這一整塊最後的輸出")

print("\n== 2. create_feature_extractor：只拿中間幾層的輸出 ==")
extractor = create_feature_extractor(model, return_nodes={"layer2": "l2", "layer3": "l3", "avgpool": "pool"})
pre = WEIGHTS.transforms()
img = Image.open("story/data/factory/anomaly/train/good/000.png").convert("RGB")
with torch.no_grad():
    out = extractor(pre(img).unsqueeze(0))
for name, t in out.items():
    print(f"{name}: shape {tuple(t.shape)}")
print("l2 每一格看原圖約 224/28 = 8 像素見方；l3 約 16 像素；pool 把整張壓成 512 個數字")

print("\n== 3. 一張圖 → 一串數字：每個特徵在整張圖上取平均 ==")
l2_mean = out["l2"].mean(dim=(2, 3))
l3_mean = out["l3"].mean(dim=(2, 3))
vec = torch.cat([l2_mean, l3_mean], dim=1)
print(f"l2 平均 {tuple(l2_mean.shape)} ＋ l3 平均 {tuple(l3_mean.shape)} → 一張圖 {tuple(vec.shape)}")

print("\n== 4. NearestNeighbors：不用標籤，只問「離最近的良品多遠」 ==")
normal = np.array([[1.0, 1.0], [1.2, 0.9], [0.8, 1.1], [1.1, 1.3], [0.9, 0.8]])
query = np.array([[1.0, 1.05], [2.5, 2.0]])
nn = NearestNeighbors(n_neighbors=2).fit(normal)
dist, idx = nn.kneighbors(query)
for q, d, i in zip(query, dist, idx):
    print(f"點 {q.tolist()}：最近的 2 個良品是第 {i.tolist()} 個，距離 {d.round(3).tolist()} → 分數（最近的距離）{d[0]:.3f}")


def vectors(paths, batch=16):
    """一批圖 → (張數, 384)：layer2、layer3 各自在整張圖上平均，再接起來。"""
    rows = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            f = extractor(x)
            rows.append(torch.cat([f["l2"].mean((2, 3)), f["l3"].mean((2, 3))], 1))
    return torch.cat(rows).numpy()


def memory_calib_split(paths, frac=0.8, seed=0):
    order = np.random.default_rng(seed).permutation(len(paths))
    cut = int(len(paths) * frac)
    return [paths[i] for i in order[:cut]], [paths[i] for i in order[cut:]]


print("\n== 5. 真的資料：記憶良品的特徵 → 測試圖離最近良品多遠 ==")
for line in ["factory", "care"]:
    root = Path(f"story/data/{line}/anomaly")
    memory, _ = memory_calib_split(sorted((root / "train/good").glob("*.png")))
    paths = sorted((root / "test").glob("*/*.png"))
    kinds = np.array([p.parent.name for p in paths])
    M, X = vectors(memory), vectors(paths)
    for k in [1, 5]:
        d = NearestNeighbors(n_neighbors=k).fit(M).kneighbors(X)[0].mean(axis=1)
        per = {str(t): round(roc_auc_score(kinds[(kinds == "good") | (kinds == t)] == t,
                                           d[(kinds == "good") | (kinds == t)]), 3) for t in sorted(set(kinds) - {"good"})}
        print(f"{line} n_neighbors={k}：AUROC {roc_auc_score(kinds != 'good', d):.3f}；依類型 {per}")

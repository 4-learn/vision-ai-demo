"""V08｜只用正常學正常：像素差 baseline、記憶／校正切法、roc_auc_score。"""
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

print("== 1. AUROC 白話：隨機抽一張異常、一張正常，異常分數比較高的機率 ==")
toy = pd.DataFrame({"image": ["n1", "n2", "n3", "n4", "a1", "a2", "a3"],
                    "y":     [0,    0,    0,    0,    1,    1,    1],
                    "score": [0.10, 0.30, 0.35, 0.60, 0.40, 0.70, 0.90]})
print(toy.to_string(index=False))
normal, anomaly = toy[toy["y"] == 0]["score"], toy[toy["y"] == 1]["score"]
pairs = list(product(anomaly, normal))
wins = sum(a > n for a, n in pairs)
print(f"異常×正常 共 {len(pairs)} 對，異常分數比較高的 {wins} 對 → {wins / len(pairs):.3f}")
print(f"roc_auc_score(y, score) = {roc_auc_score(toy['y'], toy['score']):.3f}")

print("\n== 2. 0.5 是亂猜，低於 0.5 是「方向反了」 ==")
rng = np.random.default_rng(0)
y_big = np.r_[np.zeros(10000), np.ones(10000)]
print(f"20000 張、分數用亂數給：{roc_auc_score(y_big, rng.random(20000)):.3f}")
print(f"分數全部相同：{roc_auc_score(toy['y'], np.zeros(len(toy))):.3f}")
print(f"分數加負號：{roc_auc_score(toy['y'], -toy['score']):.3f}（= 1 − {roc_auc_score(toy['y'], toy['score']):.3f}）")


def memory_calib_split(paths, frac=0.8, seed=0):
    order = np.random.default_rng(seed).permutation(len(paths))
    cut = int(len(paths) * frac)
    return [paths[i] for i in order[:cut]], [paths[i] for i in order[cut:]]


print("\n== 3. 良品切兩份：記憶 80%、校正 20% ==")
for line in ["factory", "care"]:
    goods = sorted(Path(f"story/data/{line}/anomaly/train/good").glob("*.png"))
    memory, calib = memory_calib_split(goods)
    print(f"{line}：良品 {len(goods)} 張 → 記憶 {len(memory)}、校正 {len(calib)}；校正前 3 張 {[p.stem for p in calib[:3]]}")
    again = memory_calib_split(goods)[1]
    print(f"  同一個 seed 再切一次，校正集相同：{again == calib}；兩份有重疊：{bool(set(memory) & set(calib))}")


def small_gray(path, size=64):
    return cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (size, size)).astype(np.float32) / 255


print("\n== 4. 「平均良品」長什麼樣子 ==")
for line in ["factory", "care"]:
    memory, _ = memory_calib_split(sorted(Path(f"story/data/{line}/anomaly/train/good").glob("*.png")))
    stack = np.stack([small_gray(p) for p in memory])
    mean = stack.mean(axis=0)
    print(f"{line}：疊起來 shape {stack.shape} → 平均 {mean.shape}；每個位置良品之間的標準差平均 {stack.std(axis=0).mean():.3f}")
    Path("demo/V08-pixel-baseline/out").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"demo/V08-pixel-baseline/out/{line}-mean.png", cv2.resize((mean * 255).astype(np.uint8), (256, 256)))
print("已存 out/factory-mean.png、out/care-mean.png")

print("\n== 5. 像素差分數：依瑕疵類型拆開看 AUROC ==")
for line in ["factory", "care"]:
    root = Path(f"story/data/{line}/anomaly")
    memory, calib = memory_calib_split(sorted((root / "train/good").glob("*.png")))
    paths = sorted((root / "test").glob("*/*.png"))
    kinds = np.array([p.parent.name for p in paths])
    stack = np.stack([small_gray(p) for p in memory])
    mean = stack.mean(axis=0)
    score = np.array([np.abs(small_gray(p) - mean).mean() for p in paths])
    calib_score = np.array([np.abs(small_gray(p) - mean).mean() for p in calib])
    per = {}
    for k in sorted(set(kinds) - {"good"}):
        keep = (kinds == "good") | (kinds == k)
        per[str(k)] = round(roc_auc_score(kinds[keep] == k, score[keep]), 3)
    print(f"{line}：全部 {roc_auc_score(kinds != 'good', score):.3f}；依類型 {per}")
    print(f"  平均分數：校正良品 {calib_score.mean():.4f}、測試良品 {score[kinds == 'good'].mean():.4f}、"
          f"測試異常 {score[kinds != 'good'].mean():.4f}")

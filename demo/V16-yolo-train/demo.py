"""V16｜ultralytics YOLO.train：看懂教師的訓練紀錄（epochs、imgsz、batch、freeze、三種 loss），再用 model.val() 重算 val 的 mAP50。

執行（在 repo 根目錄）：python demo/V16-yolo-train/demo.py
不重新訓練（CPU 要 8–11 分鐘）；讀的是教師用 tools/train_models.py 訓練好的 models/*-anomaly-yolo11n.pt。
工廠線的螺帽照片要先由教師執行 tools/build_data.py 產生（MVTec AD 衍生，不隨 repo 散佈）。
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")   # 不上網檢查更新
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ["LINE"] = "factory"

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

print("== 1. 教師的訓練紀錄（models/*-anomaly-yolo11n.json）==")
records = {line: json.loads((MODELS / f"{line}-anomaly-yolo11n.json").read_text()) for line in ["factory", "care"]}
print(pd.DataFrame(records).loc[["base", "epochs", "imgsz", "batch", "freeze", "train_seconds", "sha256_12"]].to_string())

print("\n== 2. 權重檔裡也存了一份：ckpt['train_args'] ==")
model = YOLO(str(MODELS / "factory-anomaly-yolo11n.pt"))
args = model.ckpt["train_args"]
print({k: args[k] for k in ["epochs", "imgsz", "batch", "freeze", "seed", "optimizer", "lr0", "patience"]})
print(f"類別：{model.names}；授權：{model.ckpt['license']}")

print("\n== 3. freeze=10：前 10 層不動 ==")
layers = model.model.model
rows = [{"層": i, "種類": type(m).__name__, "參數": sum(p.numel() for p in m.parameters())} for i, m in enumerate(layers)]
table = pd.DataFrame(rows)
print(f"YOLO11n 偵測模型共 {len(layers)} 層：0–10 是骨幹（backbone，看圖抽特徵），最後一層是 {rows[-1]['種類']}（出框）")
print(table.iloc[[0, 9, 10, 23]].to_string(index=False))
frozen, total = table["參數"][:10].sum(), table["參數"].sum()
print(f"freeze=10 凍結第 0–9 層，參數 {frozen:,} / 全部 {total:,}（{frozen / total:.1%}）")

print("\n== 4. 每一輪的三種 loss 與 val 分數（ckpt['train_results']）==")
hist = pd.DataFrame(model.ckpt["train_results"])
cols = ["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
show = hist[cols].rename(columns=lambda c: c.split("/")[-1].replace("(B)", ""))
print(show[show["epoch"].isin([1, 5, 10, 15, 20, 25, 30])].round(3).to_string(index=False))
best = hist.loc[hist["metrics/mAP50-95(B)"].idxmax()]
print(f"best.pt 是第 {int(best['epoch'])} 輪：mAP50-95 最高（{best['metrics/mAP50-95(B)']:.3f}），那一輪 mAP50 {best['metrics/mAP50(B)']:.3f}")

print("\n== 5. model.val()：在 val 上重算一次 ==")
spec = importlib.util.spec_from_file_location("v15", ROOT / "story" / "v15-mask-to-box" / "vision.py")
v15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v15)
data = ROOT / "demo" / "V16-yolo-train" / "out" / "det"
v15.yolo_det_dataset(data)                     # 15 節的資料集：遮罩變框、依類型分層切
r = model.val(data=str(data / "data.yaml"), split="val", imgsz=320, device="cpu", verbose=False, plots=False)
n_img = len(list((data / "images" / "val").glob("*.png")))
print(f"val 有 {n_img} 張圖、{int(r.nt_per_class.sum())} 個框")
print(f"r.box.map50   {r.box.map50:.3f}")
print(f"r.box.map     {r.box.map:.3f}（mAP50-95）")
print(f"r.box.mp      {r.box.mp:.3f}（precision）")
print(f"r.box.mr      {r.box.mr:.3f}（recall）")
print(f"和訓練最後存下的 train_metrics 一樣：{abs(r.box.map50 - model.ckpt['train_metrics']['metrics/mAP50(B)']) < 0.001}")

"""V14｜ultralytics YOLO 分類模式：資料夾就是標籤。載入教師訓練好的分類器，從 Results.probs 讀出答案。

執行（在 repo 根目錄）：python demo/V14-yolo-cls/demo.py
工廠線的螺帽照片要先由教師執行 tools/build_data.py 產生（MVTec AD 衍生，不隨 repo 散佈）。
"""
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")   # 不上網檢查更新
os.environ.setdefault("YOLO_VERBOSE", "False")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "story" / "data" / "factory" / "anomaly" / "test"
OUT = ROOT / "demo" / "V14-yolo-cls" / "out" / "cls"

print("== 1. 分類模式的資料夾：split／類別名稱／圖 ==")
picks = {"train": {"good": 4, "anomaly": 4}, "val": {"good": 2, "anomaly": 2}}
shutil.rmtree(OUT, ignore_errors=True)
pool = {"good": sorted((TEST / "good").glob("*.png")),
        "anomaly": sorted((TEST / "bent").glob("*.png")) + sorted((TEST / "scratch").glob("*.png"))}
used = {"good": 0, "anomaly": 0}
for split, counts in picks.items():
    for label, n in counts.items():
        (OUT / split / label).mkdir(parents=True, exist_ok=True)
        for p in pool[label][used[label]: used[label] + n]:
            shutil.copy(p, OUT / split / label / f"{p.parent.name}-{p.name}")
        used[label] += n
for d in sorted(OUT.glob("*/*")):
    print(f"{d.relative_to(OUT.parent)}/  {len(list(d.glob('*.png')))} 張，例如 {sorted(d.glob('*.png'))[0].name}")
print("沒有標註檔：圖放在哪個資料夾，標籤就是那個資料夾的名字")

print("\n== 2. 起點 yolo11n-cls.pt：ImageNet 的 1000 類 ==")
base = YOLO(str(ROOT / "models" / "yolo11n-cls.pt"))
print(f"task：{base.task}；類別數 {len(base.names)}，前 3 類 {[base.names[i] for i in range(3)]}")
r = base.predict(str(TEST / "scratch" / "000.png"), imgsz=224, device="cpu", verbose=False)[0]
print(f"一顆有刮痕的螺帽 → {r.names[r.probs.top1]}（{r.probs.top1conf.item():.3f}）")

print("\n== 3. 教師訓練好的分類器：models/factory-cls.pt ==")
model = YOLO(str(ROOT / "models" / "factory-cls.pt"))
record = json.loads((ROOT / "models" / "factory-cls.json").read_text())
print(f"task：{model.task}；類別 {model.names}")
print(f"訓練紀錄：起點 {record['base']}、epochs {record['epochs']}、imgsz {record['imgsz']}、batch {record['batch']}、CPU {record['train_seconds']} 秒")

print("\n== 4. predict：一張圖一個 Results，答案在 .probs ==")
images = [TEST / "good" / "000.png", TEST / "good" / "002.png", TEST / "bent" / "007.png",
          TEST / "color" / "004.png", TEST / "scratch" / "020.png"]   # 都是教師訓練時沒用到的 test 圖
results = model.predict([str(p) for p in images], imgsz=224, device="cpu", verbose=False)
r = results[0]
print(f"r.boxes：{r.boxes}（分類模式沒有框）")
print(f"r.probs.data：{[round(x, 3) for x in r.probs.data.tolist()]}（順序同 names，加起來是 1）")
print(f"r.probs.top1：{r.probs.top1} → {r.names[r.probs.top1]}；top1conf {r.probs.top1conf.item():.3f}")

print("\n== 5. 五張 test 圖的答案 ==")
for p, r in zip(images, results):
    anomaly = r.probs.data[0].item()
    print(f"{p.parent.name + '/' + p.name:<16}→ {r.names[r.probs.top1]:<8}anomaly 機率 {anomaly:.3f}")

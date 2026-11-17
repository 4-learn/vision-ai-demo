"""V18｜有標註 vs 只有正常：讀 compare.json、數兩種做法要準備的資料，再做一個「沒見過的異常」小實驗。

執行（在 repo 根目錄）：python demo/V18-compare/demo.py
compare.json 由教師執行 tools/build_tables.py 產生；工廠線的螺帽照片要先由教師執行 tools/build_data.py 產生
（MVTec AD 衍生，不隨 repo 散佈）。
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")   # 不上網檢查更新
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ["LINE"] = "factory"

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STORY = ROOT / "story"
OUT = ROOT / "demo" / "V18-compare" / "out"

print("== 1. compare.json：同一批 test，兩種做法 ==")
compare = json.loads((STORY / "v18-compare" / "compare.json").read_text())
for line, rows in compare.items():
    t = pd.DataFrame(rows).set_index("方法")[["抓到", "誤判"]]
    print(f"{line}：" + "；".join(f"{name} 抓到 {r['抓到']}、誤判 {r['誤判']}" for name, r in t.iterrows()))

print("\n== 2. 兩種做法各要準備什麼 ==")
rows = []
for line in ["factory", "care"]:
    anomaly = STORY / "data" / line / "anomaly"
    masks = sorted((anomaly / "ground_truth").glob("*/*_mask.png"))
    boxes = sum(cv2.connectedComponentsWithStats((cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8))[0] - 1
                for m in masks)
    rows.append({"line": line, "正常影像": len(list((anomaly / "train" / "good").glob("*.png"))),
                 "異常影像": len(masks), "要畫的框": boxes,
                 "異常類型": len(list((anomaly / "ground_truth").iterdir()))})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 3. 沒見過的異常：在 6 張良品上，畫兩種訓練時沒有的瑕疵 ==")
spec = importlib.util.spec_from_file_location("v17", STORY / "v17-yolo-eval" / "vision.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
OUT.mkdir(parents=True, exist_ok=True)
paths, kinds = [], []
for p in v.image_paths("test", "good")[:6]:
    img = cv2.imread(str(p))
    for kind in ["原圖", "黑點", "紅線"]:
        x = img.copy()
        if kind == "黑點":
            cv2.circle(x, (150, 110), 12, (0, 0, 0), -1)
        if kind == "紅線":
            cv2.line(x, (90, 90), (170, 170), (0, 0, 255), 3)
        q = OUT / f"{kind}-{p.name}"
        cv2.imwrite(str(q), x)
        paths.append(q)
        kinds.append(kind)
print(f"{len(paths)} 張：原圖、黑點、紅線各 6 張，存在 demo/V18-compare/out/")

model = v.YOLO(str(ROOT / "models" / "factory-anomaly-yolo11n.pt"))
memory, calib = v.memory_calib_split(v.image_paths("train", "good"))
ex = v.load_extractor()
bank = v.patch_bank(v.patch_features(ex, memory))
threshold = float(v.image_score(v.patch_maps(bank, v.patch_features(ex, calib))).max())
knn = v.image_score(v.patch_maps(bank, v.patch_features(ex, paths))) > threshold
yolo = [len(r.boxes) > 0 for r in model.predict([str(p) for p in paths], imgsz=320, conf=0.25, device="cpu", verbose=False)]
result = pd.DataFrame({"種類": kinds, "只有正常（小格 kNN）": knn.astype(int), "有標註（YOLO 偵測）": np.array(yolo, dtype=int)})
print(result.groupby("種類", sort=False).sum().rename(columns=lambda c: c + " 判異常").to_string())

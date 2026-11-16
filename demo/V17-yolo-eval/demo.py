"""V17｜ultralytics val()：看懂框層級的 precision／recall／mAP，再自己算影像層級（有框就判異常）。

執行（在 repo 根目錄）：python demo/V17-yolo-eval/demo.py
用工廠線的 val（test 留給 Workshop 只開一次）。IoU 與「一個標註框只配一次」見 ML 課 21 節。
工廠線的螺帽照片要先由教師執行 tools/build_data.py 產生（MVTec AD 衍生，不隨 repo 散佈）。
"""
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")   # 不上網檢查更新
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ["LINE"] = "factory"

import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("v15", ROOT / "story" / "v15-mask-to-box" / "vision.py")
v15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v15)
DATA = ROOT / "demo" / "V17-yolo-eval" / "out" / "det"
v15.yolo_det_dataset(DATA)                     # 15 節的資料集
model = YOLO(str(ROOT / "models" / "factory-anomaly-yolo11n.pt"))

print("== 1. val() 的 conf 預設很低：同一份 val，兩種 conf ==")
for conf in [None, 0.25]:
    r = model.val(data=str(DATA / "data.yaml"), split="val", imgsz=320, device="cpu", verbose=False, plots=False, conf=conf)
    print(f"conf={str(conf):<5}mAP50 {r.box.map50:.3f}、precision {r.box.mp:.3f}、recall {r.box.mr:.3f}")

print("\n== 2. 一張圖一列：標註幾個框、預測幾個框、配上幾個 ==")


def truth_boxes(label, size=256):
    rows = [list(map(float, l.split()))[1:] for l in label.read_text().splitlines() if l.strip()]
    return [[(cx - w / 2) * size, (cy - h / 2) * size, (cx + w / 2) * size, (cy + h / 2) * size] for cx, cy, w, h in rows]


def matched(pred, conf, truth, iou_min=0.5):
    """信心值由高到低，各自找 IoU 最大、還沒被配過的標註框（ML 課 21 節的規則）。"""
    if len(pred) == 0 or len(truth) == 0:
        return 0
    table = box_iou(pred, torch.tensor(truth))
    used = set()
    for i in conf.argsort(descending=True).tolist():
        cands = [(float(table[i, j]), j) for j in range(len(truth)) if j not in used]
        best, j = max(cands) if cands else (0.0, None)
        if best >= iou_min:
            used.add(j)
    return len(used)


images = sorted((DATA / "images" / "val").glob("*.png"))
rows = []
for p, res in zip(images, model.predict([str(p) for p in images], imgsz=320, conf=0.25, device="cpu", verbose=False)):
    truth = truth_boxes(DATA / "labels" / "val" / f"{p.stem}.txt")
    rows.append({"image": p.stem, "type": p.stem.split("-")[0], "標註框": len(truth), "預測框": len(res.boxes),
                 "配上": matched(res.boxes.xyxy, res.boxes.conf, truth)})
per_image = pd.DataFrame(rows)
print(per_image[per_image["type"].isin(["bent", "scratch", "good"])].head(9).to_string(index=False))

print("\n== 3. 框層級 vs 影像層級 ==")
tp, n_pred, n_truth = per_image["配上"].sum(), per_image["預測框"].sum(), per_image["標註框"].sum()
print(f"框層級：配上 {tp}／預測 {n_pred} 個框 → precision {tp / n_pred:.3f}；配上 {tp}／標註 {n_truth} 個框 → recall {tp / n_truth:.3f}")
per_image["y"] = (per_image["type"] != "good").astype(int)
per_image["判異常"] = (per_image["預測框"] > 0).astype(int)
bad, good = per_image[per_image["y"] == 1], per_image[per_image["y"] == 0]
print(f"影像層級（有框就判異常）：抓到 {bad['判異常'].sum()}/{len(bad)}、誤判 {good['判異常'].sum()}/{len(good)}")

print("\n== 4. 依類型：框漏掉了，圖卻抓到了 ==")
table = per_image.groupby("type").agg(張數=("y", "size"), 標註框=("標註框", "sum"), 配上=("配上", "sum"), 判異常=("判異常", "sum"))
print(table.to_string())

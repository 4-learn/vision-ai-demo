"""教師開課前跑一次：兩條線各訓練 YOLO 分類模式（V14）與異常偵測（V16）。

    python tools/train_models.py

產出 models/{LINE}-cls.pt、models/{LINE}-anomaly-yolo11n.pt 與對應的 .json（設定、秒數、sha256_12）。
"""
import hashlib, json, os, shutil, sys, time
from pathlib import Path
import importlib.util
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
for line in os.environ.get("LINES", "factory care").split():
    os.environ["LINE"] = line
    spec = importlib.util.spec_from_file_location(f"v_{line}", ROOT / "story/v16-yolo-train/vision.py")
    v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    work = Path(f"/tmp/vision-train-{line}")
    cls_data, det_data = work / "cls", work / "det"
    v.yolo_cls_dataset(cls_data); v.yolo_det_dataset(det_data)
    for name, base, kw in [
        ("cls", "yolo11n-cls.pt", dict(data=str(cls_data), epochs=15, imgsz=224, batch=16)),
        ("anomaly-yolo11n", "yolo11n.pt", dict(data=str(det_data / "data.yaml"), epochs=30, imgsz=320, batch=8, freeze=10)),
    ]:
        t = time.time()
        YOLO(str(ROOT / "models" / base)).train(device="cpu", workers=0, seed=0, deterministic=True, project=str(work / "runs"),
                                                 name=name, exist_ok=True, plots=False, verbose=False, **kw)
        dst = ROOT / "models" / f"{line}-{name}.pt"
        shutil.copy(work / "runs" / name / "weights" / "best.pt", dst)
        meta = dict(base=base, train_seconds=round(time.time() - t), sha256_12=hashlib.sha256(dst.read_bytes()).hexdigest()[:12],
                    **{"freeze": 0, **{k: (str(val).replace(str(work), "<work>")) if k == "data" else val for k, val in kw.items()}})
        (ROOT / "models" / f"{line}-{name}.json").write_text(json.dumps(meta, indent=1))
        print(line, name, meta, flush=True)

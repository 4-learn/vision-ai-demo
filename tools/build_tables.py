"""產生 V18 compare.json 與 V26 budget.json（教師先跑；講義讀這兩個檔）。

    LINES="factory care" python tools/build_tables.py
"""
import importlib.util, json, os, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
compare, budget = {}, {}
for line in os.environ.get("LINES", "factory care").split():
    os.environ["LINE"] = line
    spec = importlib.util.spec_from_file_location(f"v_{line}", ROOT / "story/v17-yolo-eval/vision.py")
    v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    # 同一批 test 影像：YOLO 偵測資料集的 test split
    data = Path(f"/tmp/tables-{line}/det"); v.yolo_det_dataset(data)
    imgs = sorted((data / "images" / "test").glob("*.png"))
    truth = np.array([0 if p.name.startswith("good") else 1 for p in imgs])
    # 只有正常：小格 kNN
    memory, calib = v.memory_calib_split(v.image_paths("train", "good"))
    ex = v.load_extractor(); bank = v.patch_bank(v.patch_features(ex, memory))
    thr = float(v.image_score(v.patch_maps(bank, v.patch_features(ex, calib))).max())
    t = time.perf_counter(); s = v.image_score(v.patch_maps(bank, v.patch_features(ex, imgs))); knn_ms = (time.perf_counter() - t) / len(imgs) * 1000
    pk = s > thr
    # 有標註：YOLO 偵測
    model = v.YOLO(str(ROOT / "models" / f"{line}-anomaly-yolo11n.pt"))
    t = time.perf_counter(); res = model.predict([str(p) for p in imgs], imgsz=320, conf=0.25, device="cpu", verbose=False); yolo_ms = (time.perf_counter() - t) / len(imgs) * 1000
    py = np.array([int(len(r.boxes) > 0) for r in res])
    compare[line] = [
        {"方法": "只有正常（小格 kNN）", "需要的標註": "只要正常影像", "抓到": f"{int(pk[truth == 1].sum())}/{int(truth.sum())}",
         "誤判": f"{int(pk[truth == 0].sum())}/{int((truth == 0).sum())}", "沒見過的異常": "可以（離正常遠就算）"},
        {"方法": "有標註（YOLO 偵測）", "需要的標註": "每張異常畫框", "抓到": f"{int(py[truth == 1].sum())}/{int(truth.sum())}",
         "誤判": f"{int(py[truth == 0].sum())}/{int((truth == 0).sum())}", "沒見過的異常": "不一定（只學過畫過框的）"},
    ]
    # 解析度預算：小格 kNN 在三種輸入大小
    rows = []
    paths, kinds, y = v.test_set()
    for size in [160, 224, 320]:
        pre_size = size
        from torchvision.transforms import v2 as T
        import torch, cv2
        tf = T.Compose([T.Resize(size, antialias=True), T.CenterCrop(size), T.ToDtype(torch.float32, scale=True),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        def feats(ps):
            out = []
            with torch.no_grad():
                for i in range(0, len(ps), 16):
                    x = torch.stack([tf(torch.from_numpy(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) for p in ps[i:i + 16]])
                    f = ex(x); l3 = torch.nn.functional.interpolate(f["l3"], size=f["l2"].shape[-2:], mode="bilinear")
                    out.append(torch.cat([f["l2"], l3], 1))
            return torch.cat(out).numpy()
        b = v.patch_bank(feats(memory))
        t = time.perf_counter(); st = v.image_score(v.patch_maps(b, feats(paths))); ms = (time.perf_counter() - t) / len(paths) * 1000
        rows.append({"輸入大小": size, "AUROC": round(roc_auc_score(y, st), 3), "每張毫秒（教師機）": round(ms)})
    budget[line] = rows
    print(line, compare[line], rows, flush=True)
for name, obj in [("v18-compare/compare.json", compare), ("v26-budget/budget.json", budget)]:
    path = ROOT / "story" / name
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(obj); path.write_text(json.dumps(old, ensure_ascii=False, indent=1))

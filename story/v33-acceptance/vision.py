"""影像 AI 課進度存檔 v33：更強的偵測器，裝到現場。

執行（在這個資料夾裡）：
    LINE=factory python vision.py     # 工廠線：螺帽瑕疵、彎腰搬重物
    LINE=care python vision.py        # 健康線：寢室畫面異常、跌倒
"""
import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")   # ultralytics 不要印進度條

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2 as T
from sklearn.metrics import roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve
import shutil
from ultralytics import YOLO
from collections import deque
from scipy.optimize import linear_sum_assignment
import onnxruntime as ort
import time
from onnxruntime.quantization import QuantType, quantize_dynamic
import requests
import hashlib
import logging

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
DATA = STORY / "data" / LINE
ANOMALY = DATA / "anomaly"
NEW_EVENT = {"factory": "defect_detected", "care": "room_anomaly"}[LINE]
torch.manual_seed(0)
torch.set_num_threads(4)


def image_paths(split, kind="*"):
    return sorted((ANOMALY / split).glob(f"{kind}/*.png"))


def test_set():
    """測試集：路徑、類型（good 或瑕疵類型）、答案（0 正常、1 異常）。"""
    paths = image_paths("test")
    kinds = [p.parent.name for p in paths]
    return paths, kinds, np.array([0 if k == "good" else 1 for k in kinds])


def mask_of(path):
    """瑕疵遮罩（白＝異常的位置）；正常圖回傳全黑。"""
    m = ANOMALY / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    return cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) if m.exists() else np.zeros((256, 256), np.uint8)


def clahe(gray, clip=2.0, grid=8):
    """CLAHE：每一小區自己做直方圖等化，對比不准拉超過 clip。"""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def contrast(gray, mask):
    """「東西」和「背景」的亮度差：瑕疵或異常要看得出來，這個數字要大。"""
    inside, outside = gray[mask > 0], gray[mask == 0]
    return float(abs(inside.mean() - outside.mean())) if len(inside) else 0.0


def denoise(gray, h=10):
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def sharpen(gray, amount=1.0, sigma=2.0):
    """unsharp mask：原圖 +（原圖 − 模糊圖）× amount。"""
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def main_axis_angle(gray):
    """前景（比背景亮的部分）的主軸角度：用 cv2.moments 算二階矩。"""
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.moments(fg, binaryImage=True)
    return 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def best_rotation(gray, reference, step=3):
    """每隔 step 度轉一次，找出和參考圖最像（平均像素差最小）的角度。"""
    small, ref = cv2.resize(gray, (64, 64)).astype(np.float32), cv2.resize(reference, (64, 64)).astype(np.float32)
    diffs = [(np.abs(rotate(small, a) - ref).mean(), a) for a in range(0, 360, step)]
    return min(diffs)[1]


def edge_density(gray, low=50, high=150):
    """Canny 邊緣占整張圖的比例。"""
    return float((cv2.Canny(gray, low, high) > 0).mean())


def shape_stats(gray):
    """最大前景輪廓的面積、周長、圓度（4π×面積÷周長²，正圓是 1）。"""
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=10, maxRadius=60)
    hole = float(circles[0, 0, 2]) if circles is not None else 0.0
    return {"area": area, "perimeter": perim, "circularity": 4 * np.pi * area / perim ** 2, "hole_radius": hole}


def memory_calib_split(paths, frac=0.8, seed=0):
    """良品切兩份：80% 當「記憶」、20% 當「校正」（定門檻用，不碰測試集）。"""
    order = np.random.default_rng(seed).permutation(len(paths))
    cut = int(len(paths) * frac)
    return [paths[i] for i in order[:cut]], [paths[i] for i in order[cut:]]


RULE_FEATURES = ["edge_density", "area", "perimeter", "circularity", "hole_radius"]


def rule_table(paths):
    rows = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        rows.append({"edge_density": edge_density(g), **shape_stats(g)})
    return pd.DataFrame(rows)


def rule_baseline(train, test):
    """規則：任何一個量測超出良品的範圍（最小～最大）就判異常。"""
    lo, hi = train.min(), train.max()
    return ((test < lo) | (test > hi)).any(axis=1).astype(int)


AUGMENT = T.Compose([T.RandomRotation(degrees=180), T.ColorJitter(brightness=0.3, contrast=0.3)])
EVAL = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def small_gray(path, size=64):
    return cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (size, size)).astype(np.float32) / 255


def pixel_scores(memory, paths):
    mean = np.mean([small_gray(p) for p in memory], axis=0)
    return np.array([np.abs(small_gray(p) - mean).mean() for p in paths])


WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def load_extractor():
    return create_feature_extractor(resnet18(weights=WEIGHTS).eval(), {"layer2": "l2", "layer3": "l3"})


def patch_features(extractor, paths, batch=16):
    """每張圖 → (512 個特徵, 28, 28)：layer2 與放大後的 layer3 疊在一起。"""
    pre = WEIGHTS.transforms()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([pre(torch.from_numpy(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
                             for p in paths[i:i + batch]])
            f = extractor(x)
            l3 = torch.nn.functional.interpolate(f["l3"], size=f["l2"].shape[-2:], mode="bilinear")
            out.append(torch.cat([f["l2"], l3], 1))
    return torch.cat(out).numpy()


def global_scores(memory_feats, feats):
    nn = NearestNeighbors(n_neighbors=1).fit(memory_feats.mean((2, 3)))
    return nn.kneighbors(feats.mean((2, 3)))[0][:, 0]


def patch_bank(memory_feats, size=20000, seed=0):
    """正常小格記憶庫：所有良品的所有小格，隨機抽 size 個。"""
    C = memory_feats.shape[1]
    bank = memory_feats.transpose(0, 2, 3, 1).reshape(-1, C)
    idx = np.random.default_rng(seed).choice(len(bank), min(size, len(bank)), replace=False)
    return NearestNeighbors(n_neighbors=1).fit(bank[idx])


def patch_maps(bank, feats):
    """每一格到最像的正常小格多遠 → (張數, 28, 28) 的距離圖。"""
    n, C, h, w = feats.shape
    return bank.kneighbors(feats.transpose(0, 2, 3, 1).reshape(-1, C))[0][:, 0].reshape(n, h, w)


def image_score(maps, k=10):
    """一張圖的分數：最遠的 k 格取平均（只取最大值會被一格雜訊拉走）。"""
    flat = maps.reshape(len(maps), -1)
    return np.sort(flat, axis=1)[:, -k:].mean(1)


def heatmap(dist_map, size=256):
    """28×28 距離圖放大回原圖大小、轉成 0–255，再上色。"""
    big = cv2.resize(dist_map, (size, size), interpolation=cv2.INTER_LINEAR)
    norm = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return big, cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def peak_in_mask(dist_map, mask, tolerance=16):
    """最熱的點落在遮罩附近嗎？一格小格對應原圖約 9 像素，所以允許 tolerance 像素的誤差。"""
    big = cv2.resize(dist_map, mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
    y, x = np.unravel_index(big.argmax(), big.shape)
    near = cv2.dilate((mask > 0).astype(np.uint8), np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8))
    return bool(near[y, x] > 0)


def choose_threshold(calib_scores, val_scores=None, val_y=None, min_recall=0.9):
    """沒有瑕疵樣本：用校正良品的最大值。有少量瑕疵：在 val 上找 recall 達標、誤剔最少的門檻。"""
    if val_scores is None:
        return float(calib_scores.max())
    fpr, tpr, thr = roc_curve(val_y, val_scores)
    ok = np.where(tpr >= min_recall)[0]
    return float(thr[ok[0]])


def yolo_cls_dataset(out):
    """YOLO 分類模式要的資料夾：train／val／test 底下各分 good、anomaly。良品用記憶、瑕疵切一半給訓練。"""
    memory, calib = memory_calib_split(image_paths("train", "good"))
    paths, kinds, y = test_set()
    bad = [p for p, t in zip(paths, y) if t == 1]
    good_test = [p for p, t in zip(paths, y) if t == 0]
    rng = np.random.default_rng(0)
    bad = [bad[i] for i in rng.permutation(len(bad))]
    plan = {"train": (memory[:120], bad[: len(bad) // 2]), "val": (calib, bad[len(bad) // 2: len(bad) * 3 // 4]),
            "test": (good_test, bad[len(bad) * 3 // 4:])}
    shutil.rmtree(out, ignore_errors=True)
    for split, (goods, bads) in plan.items():
        for label, items in [("good", goods), ("anomaly", bads)]:
            (out / split / label).mkdir(parents=True, exist_ok=True)
            for p in items:
                shutil.copy(p, out / split / label / f"{p.parent.name}-{p.name}")
    return {s: {"good": len(g), "anomaly": len(b)} for s, (g, b) in plan.items()}


def mask_to_boxes(mask, min_area=20):
    """遮罩 → 每一塊白色一個框（YOLO 格式：中心、寬高，都是 0–1）。"""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
    h, w = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_area:
            boxes.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return boxes


def yolo_det_dataset(out):
    """瑕疵影像依類型分層切 60／20／20；良品放空的 label 檔。"""
    paths, kinds, y = test_set()
    rng = np.random.default_rng(0)
    split_of = {}
    for k in sorted(set(kinds)):
        idx = [i for i, kk in enumerate(kinds) if kk == k]
        idx = [idx[j] for j in rng.permutation(len(idx))]
        a, b = int(round(len(idx) * 0.6)), int(round(len(idx) * 0.8))
        for j, i in enumerate(idx):
            split_of[i] = "train" if j < a else "val" if j < b else "test"
    shutil.rmtree(out, ignore_errors=True)
    stats = {}
    for i, p in enumerate(paths):
        s = split_of[i]
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        name = f"{kinds[i]}-{p.stem}"
        shutil.copy(p, out / "images" / s / f"{name}.png")
        boxes = mask_to_boxes(mask_of(p)) if y[i] else []
        (out / "labels" / s / f"{name}.txt").write_text("".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in boxes))
        st = stats.setdefault(s, {"images": 0, "boxes": 0})
        st["images"] += 1
        st["boxes"] += len(boxes)
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: anomaly\n")
    return stats


KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


def load_poses(name="poses.csv"):
    """標籤欄一律當字串讀：新地點的資料有空字串（沒有標籤），主資料全是 0／1。"""
    return pd.read_csv(DATA / "pose" / name, keep_default_na=False, dtype={"alert": str, "activity": str, "posture": str})


def frames(seq):
    """把一段骨架變成「一格一格流進來」：每次 yield 一格。"""
    for _, row in seq.iterrows():
        yield row


def torso_angle(row):
    sx = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    sy = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hx = (row["left_hip_x"] + row["right_hip_x"]) / 2
    hy = (row["left_hip_y"] + row["right_hip_y"]) / 2
    return float(np.degrees(np.arctan2(abs(sx - hx), abs(hy - sy))))


class Window:
    """只留最近 n 格：deque(maxlen=n) 滿了會自動丟掉最舊的。"""

    def __init__(self, n=10):
        self.angles = deque(maxlen=n)

    def push(self, row):
        self.angles.append(torso_angle(row))
        return max(self.angles) - min(self.angles) if len(self.angles) == self.angles.maxlen else 0.0


def centers(df_frame):
    """一格裡每副骨架的中心（髖的中點）。"""
    return np.c_[(df_frame["left_hip_x"] + df_frame["right_hip_x"]) / 2,
                 (df_frame["left_hip_y"] + df_frame["right_hip_y"]) / 2]


def match_tracks(prev, curr):
    """上一格的人 × 這一格的人，距離最小的配在一起（匈牙利法）。"""
    cost = np.linalg.norm(prev[:, None, :] - curr[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist())), cost[rows, cols]


ALERT_MOVE = {"factory": "bend_start", "care": "fall_start"}[LINE]
TRANSITIONS = {
    "normal": {"big_change": "watching"},
    "watching": {"still_down": "alert", "recovered": "normal"},
    "alert": {"acked": "normal"},
}


def step(state, event):
    """狀態機查表：不在表上的事件，狀態不變。"""
    return TRANSITIONS.get(state, {}).get(event, state)


RULE = dict(down_min=55, hold=10)   # 為跌倒設計：倒下之後「一直低姿勢」1 秒以上


def run_state_machine(seq, change_min=40, down_min=RULE["down_min"], hold=RULE["hold"]):
    """逐格：姿勢在 1 秒內變很大 → watching；之後一直「低姿勢」超過 hold 格 → alert。"""
    w = Window(10)
    state, low, history = "normal", 0, []
    for row in frames(seq):
        change = w.push(row)
        angle = torso_angle(row)
        low = low + 1 if angle >= down_min else 0
        if state == "normal" and change >= change_min:
            state = step(state, "big_change")
        elif state == "watching" and low >= hold:
            state = step(state, "still_down")
        elif state == "watching" and low == 0 and change < 10:
            state = step(state, "recovered")
        history.append(state)
    return history


def debounce(flags, cooldown=60):
    """逐格的告警旗標 → 通知：連續的 True 只算一次；通知後 cooldown 格內不再通知。"""
    notes, last = [], -10 ** 9
    s = pd.Series(flags).astype(bool)
    starts = s & ~s.shift(fill_value=False)
    for i in np.flatnonzero(starts.to_numpy()):
        if i - last >= cooldown:
            notes.append(int(i))
            last = i
    return notes


def export_feature_onnx(path):
    """把「ResNet18 算到 layer2＋layer3」包成一個模型，匯出 ONNX。"""
    class Patch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            r = resnet18(weights=WEIGHTS).eval()
            self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
            self.l2, self.l3 = r.layer2, r.layer3

        def forward(self, x):
            a = self.l2(self.stem(x))
            b = torch.nn.functional.interpolate(self.l3(a), size=a.shape[-2:], mode="bilinear")
            return torch.cat([a, b], 1)

    model = Patch().eval()
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    torch.onnx.export(model, x, str(path), input_names=["image"], output_names=["features"],
                      dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}}, opset_version=17, dynamo=False)
    return model, x


def time_it(fn, n=10, warmup=2):
    """暖機 warmup 次不算，再量 n 次，回傳中位數（毫秒）。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def platform_url():
    url = os.getenv("PLATFORM_URL")
    if url:
        return url
    import sys
    sys.path.insert(0, str(STORY))
    import platform_mock
    return platform_mock.start(PROFILE)


def send(url, event, api_key="dev-device"):
    r = requests.post(f"{url}/events", json=event, headers={"X-API-Key": api_key}, timeout=5)
    return r.status_code, r.json()


def anomaly_event(path, score, threshold, peak):
    return {"event_type": NEW_EVENT, "location": PROFILE["locations"][0], "source": "camera_line_01" if LINE == "factory" else "camera_room_01",
            "confidence": round(float(min(1.0, 0.5 + (score - threshold))), 3), "occurred_at": "2026-02-02T09:00:00",
            "detail": {"image": path.name, "score": round(float(score), 3), "threshold": round(threshold, 3),
                       "peak_xy": [int(peak[0]), int(peak[1])]}}


def sha256_12(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def startup_check(manifest):
    """啟動自檢：設定檔裡記的指紋和檔案對不上，就不准啟動。"""
    for name, expected in manifest["files"].items():
        if sha256_12(STORY.parent / name) != expected:
            raise RuntimeError(f"{name} 的指紋對不上，拒絕啟動")
    return True


def json_logger():
    log = logging.getLogger(f"vision-{LINE}")
    log.handlers.clear()
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('{"ts": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}', "%Y-%m-%dT%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


class AnomalyDetector:
    """V28：小格 kNN 異常偵測包成一個物件：先建記憶庫，之後每張圖回傳分數與最熱點。"""

    def __init__(self):
        memory, calib = memory_calib_split(image_paths("train", "good"))
        self.extractor = load_extractor()
        self.bank = patch_bank(patch_features(self.extractor, memory))
        self.threshold = float(image_score(patch_maps(self.bank, patch_features(self.extractor, calib))).max())

    def score(self, paths):
        maps = patch_maps(self.bank, patch_features(self.extractor, paths))
        out = []
        for path, m in zip(paths, maps):
            big = cv2.resize(m, (256, 256), interpolation=cv2.INTER_LINEAR)
            y, x = np.unravel_index(big.argmax(), big.shape)
            out.append((path, float(image_score(m[None])[0]), (int(x), int(y))))
        return out


class FallMonitor:
    """V29：一個人的骨架串流 → 狀態機 → 只在「剛進入 alert」時回傳事件。"""

    def __init__(self, person_id, location):
        self.window, self.state, self.low = Window(10), "normal", 0
        self.person_id, self.location = person_id, location

    def push(self, row, t, change_min=40, down_min=RULE["down_min"], hold=RULE["hold"]):
        change, angle = self.window.push(row), torso_angle(row)
        self.low = self.low + 1 if angle >= down_min else 0
        before = self.state
        if self.state == "normal" and change >= change_min:
            self.state = step(self.state, "big_change")
        elif self.state == "watching" and self.low >= hold:
            self.state = step(self.state, "still_down")
        elif self.state == "watching" and self.low == 0 and change < 10:
            self.state = step(self.state, "recovered")
        if before != "alert" and self.state == "alert":
            return {"event_type": {"factory": "unsafe_lifting", "care": "fall_detected"}[LINE], "location": self.location,
                    "source": "pose_01", "confidence": 0.9, "occurred_at": t,
                    "detail": {"person_id": self.person_id, "torso_angle": round(angle, 1)}}
        return None


def verdict_rate(alerts):
    """值班的人按下的確認：resolved＝真的、dismissed＝誤報。真告警率的分母只算已經確認的。"""
    done = alerts[alerts["status"].isin(["resolved", "dismissed"])]
    return round((done["status"] == "resolved").mean(), 3) if len(done) else None


class FakeCamera:
    """Mock：不連攝影機，依序讀資料夾裡的圖。"""

    def __init__(self, paths):
        self.paths, self.i = list(paths), 0

    def capture(self):
        p = self.paths[self.i % len(self.paths)]
        self.i += 1
        return p


def pipeline(camera, detector, url, n):
    """取像 → 異常分數 → 過門檻才送事件。回傳送出的事件數。"""
    sent = 0
    for _ in range(n):
        path = camera.capture()
        _, score, peak = detector.score([path])[0]
        if score > detector.threshold:
            code, _ = send(url, anomaly_event(path, score, detector.threshold, peak))
            sent += int(code == 200)
    return sent


ACCEPTANCE = {"factory": {"min_recall": 0.75, "max_false_rate": 0.1}, "care": {"min_recall": 0.9, "max_false_rate": 0.1}}[LINE]


def main():
    det = AnomalyDetector()
    paths, kinds, y = test_set()
    scores = np.array([s for _, s, _ in det.score(paths)])
    pred = scores > det.threshold
    recall, false_rate = pred[y == 1].mean(), pred[y == 0].mean()
    results = pd.DataFrame({"image": [f"{k}/{p.name}" for p, k in zip(paths, kinds)], "score": scores.round(4), "pred": pred.astype(int), "y": y})
    print(f"== {LINE}：驗收報告（標準先寫好：recall ≥ {ACCEPTANCE['min_recall']}、誤判率 ≤ {ACCEPTANCE['max_false_rate']}）==")
    print(f"recall {recall:.3f}（{'通過' if recall >= ACCEPTANCE['min_recall'] else '不通過'}）、"
          f"誤判率 {false_rate:.3f}（{'通過' if false_rate <= ACCEPTANCE['max_false_rate'] else '不通過'}）")
    print(results.assign(type=kinds).groupby("type")["pred"].agg(["size", "sum"]).rename(columns={"size": "張數", "sum": "判異常"}).to_string())


if __name__ == "__main__":
    main()

"""V28｜Workshop：異常偵測 MVP 接上平台。把「建記憶庫」和「打分數」拆成一個類別的兩個方法。

執行（在 repo 根目錄）：python demo/V28-mvp-anomaly/demo.py
只用健康線的合成畫面（程式畫的），不需要工廠線資料。偵測器用最簡單的像素差，重點在類別的設計。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
STORY = ROOT / "story"
ANOMALY = STORY / "data" / "care" / "anomaly"
PROFILE = json.loads((STORY / "profiles" / "care.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(STORY))
import platform_mock  # noqa: E402

LOADS = {"count": 0}          # 數一數讀了幾張「正常」的圖


def small(path):
    return cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (32, 32)).astype(np.float32) / 255


def score_once(path, memory, calib):
    """函式寫法：每判一張，都從頭建一次「正常的樣子」和門檻。"""
    LOADS["count"] += len(memory) + len(calib)
    mean = np.mean([small(p) for p in memory], axis=0)
    threshold = max(float(np.abs(small(p) - mean).mean()) for p in calib)
    return float(np.abs(small(path) - mean).mean()), threshold


class TinyDetector:
    """類別寫法：__init__ 建一次記憶與門檻，score() 每張圖只做減法。"""

    def __init__(self, memory, calib):
        LOADS["count"] += len(memory) + len(calib)
        self.mean = np.mean([small(p) for p in memory], axis=0)
        self.threshold = max(self._score(p)[0] for p in calib)

    def _score(self, path):
        diff = np.abs(small(path) - self.mean)
        y, x = np.unravel_index(diff.argmax(), diff.shape)
        return float(diff.mean()), (int(x * 8 + 4), int(y * 8 + 4))   # 32×32 的一格 ＝ 原圖 8 像素

    def score(self, paths):
        """回傳 [(路徑, 分數, 最熱點 (x, y)), ...]，一張圖一筆。"""
        return [(p, *self._score(p)) for p in paths]


goods = sorted((ANOMALY / "train" / "good").glob("*.png"))
memory, calib = goods[:160], goods[160:]
tests = [ANOMALY / "test" / k / "000.png" for k in ["good", "fallen", "obstacle", "blocked"]]

print("== 1. 函式寫法：判 4 張，每張都重建記憶 ==")
LOADS["count"] = 0
for p in tests:
    score_once(p, memory, calib)
print(f"讀了正常圖 {LOADS['count']} 次（{len(memory) + len(calib)} 張 × 4）")

print("\n== 2. 類別寫法：建一次，之後只打分數 ==")
LOADS["count"] = 0
det = TinyDetector(memory, calib)
rows = det.score(tests)
print(f"讀了正常圖 {LOADS['count']} 次；門檻（校正最大值）{det.threshold:.4f}")
for path, score, peak in rows:
    print(f"{path.parent.name:<9}{path.name}  分數 {score:.4f}  最熱點 {peak}  {'異常' if score > det.threshold else '正常'}")

print("\n== 3. 分數 → 事件的信心值：0.5 ＋（分數 − 門檻）==")
for path, score, peak in rows:
    conf = round(min(1.0, 0.5 + (score - det.threshold)), 3)
    print(f"{path.parent.name:<9}超過門檻 {score - det.threshold:+.4f} → confidence {conf}")

print("\n== 4. 只有超過門檻的才送進平台 ==")
PROFILE["event_types"].setdefault("room_anomaly", {"severity": "high", "escalate_after_minutes": 2})
url = platform_mock.start(PROFILE)
for path, score, peak in rows:
    if score <= det.threshold:
        continue
    event = {"event_type": "room_anomaly", "location": "room_301", "source": "camera_room_01",
             "confidence": round(min(1.0, 0.5 + (score - det.threshold)), 3), "occurred_at": "2026-02-02T09:00:00",
             "detail": {"image": f"{path.parent.name}/{path.name}", "score": round(score, 4),
                        "threshold": round(det.threshold, 4), "peak_xy": list(peak)}}
    r = requests.post(f"{url}/events", json=event, headers={"X-API-Key": "dev-device"}, timeout=5)
    print(f"{event['detail']['image']:<17}→ {r.status_code} {r.json()['alert_status']}")
print("平台上的告警：", [a["status"] for a in requests.get(f"{url}/alerts", timeout=5).json()])

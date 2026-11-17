"""V31｜Mock：取像、偵測、送出三段各自可以換。換成假的，整條流程不用攝影機、不用模型、不用網路也能跑。

執行（在 repo 根目錄）：python demo/V31-mock/demo.py
只用健康線的合成畫面（程式畫的）與 story/platform_mock.py。
"""
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
STORY = ROOT / "story"
TEST = STORY / "data" / "care" / "anomaly" / "test"
PROFILE = json.loads((STORY / "profiles" / "care.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(STORY))
import platform_mock  # noqa: E402


# ---- 取像只要有 capture()、送出只要有 send(event)：先各做一個假的 ----
# FakeCamera 不連攝影機：依序讀資料夾裡的圖，讀完從頭再來。
# RecordingSender 不連網路：把事件記在串列裡，事後檢查送了什麼。
class FakeCamera:
    def __init__(self, paths):
        self.paths, self.i = list(paths), 0

    def capture(self):
        p = self.paths[self.i % len(self.paths)]
        self.i += 1
        return p


class RecordingSender:
    def __init__(self):
        self.sent = []

    def send(self, event):
        self.sent.append(event)
        return 200


# 模擬攝影機斷線：第 n 次之後就丟錯。
class BrokenCamera:
    def __init__(self, inner, fail_after):
        self.inner, self.left = inner, fail_after

    def capture(self):
        if self.left == 0:
            raise ConnectionError("camera offline")
        self.left -= 1
        return self.inner.capture()


# ---- 偵測：只要有 threshold 和 score(paths) 就行 ----
class FakeDetector:
    """不載模型：資料夾名稱不是 good 就給高分。用來測「流程」，不是測「準不準」。"""
    threshold = 1.0

    def score(self, paths):
        return [(p, 2.0 if p.parent.name != "good" else 0.5, (128, 128)) for p in paths]


# ---- 真的送出 ----
class HttpSender:
    def __init__(self, url, api_key="dev-device"):
        self.url, self.key = url, api_key

    def send(self, event):
        r = requests.post(f"{self.url}/events", json=event, headers={"X-API-Key": self.key}, timeout=5)
        return r.status_code


def pipeline(camera, detector, sender, n):
    """三段串起來。它不知道、也不在乎拿到的是真的還是假的。"""
    sent = errors = 0
    for _ in range(n):
        try:
            path = camera.capture()
        except ConnectionError:
            errors += 1
            continue
        _, score, peak = detector.score([path])[0]
        if score > detector.threshold:
            event = {"event_type": "room_anomaly", "location": "room_301", "source": "camera_room_01",
                     "confidence": 0.9, "occurred_at": "2026-02-02T09:00:00",
                     "detail": {"image": f"{path.parent.name}/{path.name}", "score": score}}
            sent += int(sender.send(event) == 200)
    return {"sent": sent, "camera_errors": errors}


frames = [TEST / "fallen" / "000.png", TEST / "good" / "000.png", TEST / "obstacle" / "000.png",
          TEST / "good" / "001.png", TEST / "blocked" / "000.png"]

print("== 1. 三段全假：不用攝影機、模型、網路 ==")
rec = RecordingSender()
print(pipeline(FakeCamera(frames), FakeDetector(), rec, 10))
print("送了哪些：", [e["detail"]["image"] for e in rec.sent])

print("\n== 2. FakeCamera 讀完從頭再來 ==")
cam = FakeCamera(frames)
print([cam.capture().parent.name for _ in range(7)])

print("\n== 3. 換掉一段：攝影機第 3 次之後斷線 ==")
rec = RecordingSender()
print(pipeline(BrokenCamera(FakeCamera(frames), fail_after=3), FakeDetector(), rec, 10))

print("\n== 4. 只把送出換成真的 HTTP（替身平台），其他兩段不動 ==")
PROFILE["event_types"].setdefault("room_anomaly", {"severity": "high", "escalate_after_minutes": 2})
url = platform_mock.start(PROFILE)
print(pipeline(FakeCamera(frames), FakeDetector(), HttpSender(url), 5))
print("平台上的告警：", len(requests.get(f"{url}/alerts", timeout=5).json()), "筆")
print("錯的 key：", pipeline(FakeCamera(frames), FakeDetector(), HttpSender(url, api_key="nope"), 5))

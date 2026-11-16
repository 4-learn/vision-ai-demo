"""V27｜requests＋設定檔：異常偵測的結果變成新的事件類型，送進平台。事件裡放分數、門檻、最熱點，不放影像。

執行（在 repo 根目錄）：
    python demo/V27-new-event/demo.py
"""
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from sklearn.neighbors import NearestNeighbors
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

torch.manual_seed(0)
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "story"))
import platform_mock  # noqa: E402  API 課平台的替身（行為照 v0.12）

ANOMALY = ROOT / "story" / "data" / "care" / "anomaly"   # 健康線：寢室畫面（程式畫的合成畫面）
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
KEY = {"X-API-Key": "dev-device"}

# ---- 先用 V10–V12 的方法（縮小版）算出：分數、門檻、最熱點 ----
extractor = create_feature_extractor(resnet18(weights=WEIGHTS).eval(), {"layer2": "l2", "layer3": "l3"})


def feats(paths):
    with torch.no_grad():
        x = torch.stack([WEIGHTS.transforms()(torch.from_numpy(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
                         for p in paths])
        f = extractor(x)
        l3 = torch.nn.functional.interpolate(f["l3"], size=f["l2"].shape[-2:], mode="bilinear")
        return torch.cat([f["l2"], l3], 1).numpy()


def maps(nn, f):
    return nn.kneighbors(f.transpose(0, 2, 3, 1).reshape(-1, f.shape[1]))[0].reshape(len(f), 28, 28)


def top10(m):
    return np.sort(m.reshape(len(m), -1), axis=1)[:, -10:].mean(1)


goods = sorted((ANOMALY / "train" / "good").glob("*.png"))
m = feats(goods[:40])
bank = m.transpose(0, 2, 3, 1).reshape(-1, m.shape[1])
nn = NearestNeighbors(n_neighbors=1).fit(bank[np.random.default_rng(0).choice(len(bank), 8000, replace=False)])
threshold = float(top10(maps(nn, feats(goods[40:60]))).max())
image = ANOMALY / "test" / "obstacle" / "001.png"
dist = maps(nn, feats([image]))
score = float(top10(dist)[0])
big = cv2.resize(dist[0], (256, 256), interpolation=cv2.INTER_LINEAR)
y, x = np.unravel_index(big.argmax(), big.shape)
mask = cv2.imread(str(ANOMALY / "ground_truth" / "obstacle" / "001_mask.png"), cv2.IMREAD_GRAYSCALE)

print("== 1. 偵測器算出來的東西 ==")
print(f"影像 {image.parent.name}/{image.name}：分數 {score:.3f}、門檻 {threshold:.3f}（校正良品最大值）")
print(f"最熱點（原圖 256×256 的座標）x={x}, y={y}；落在異常遮罩裡：{bool(mask[y, x] > 0)}")


def to_confidence(score, threshold):
    """距離不是機率：剛好在門檻上給 0.5（待確認），超過門檻多少就加多少，最多 1.0。和進度存檔 v27 同一個換法。"""
    return round(float(min(1.0, 0.5 + (score - threshold))), 3)


print("\n== 2. 事件 JSON：分數、門檻、最熱點放進 detail，不放影像 ==")
event = {"event_type": "room_anomaly", "location": "room_301", "source": "camera_room_01",
         "confidence": to_confidence(score, threshold), "occurred_at": "2026-02-02T03:10:00",
         "detail": {"score": round(score, 3), "threshold": round(threshold, 3), "peak_xy": [int(x), int(y)],
                    "image_size": [256, 256], "model": "resnet18-l2l3-knn"}}
for k, v in event.items():
    print(f"{k}: {json.dumps(v, ensure_ascii=False)}")
png = image.read_bytes()
print(f"事件 {len(json.dumps(event).encode())} bytes；這張影像本身 {len(png)} bytes")

print("\n== 3. 設定檔還沒有這個事件類型：平台回 400 ==")
profile = json.loads((ROOT / "story" / "profiles" / "care.json").read_text(encoding="utf-8"))
url = platform_mock.start(profile)
print("設定檔收的事件類型：", list(profile["event_types"]))
r = requests.post(f"{url}/events", json=event, headers=KEY, timeout=5)
print(f"{r.status_code} {r.json()['detail']}")

print("\n== 4. 加進設定檔，再送一次 ==")
before = copy.deepcopy(profile)
profile["event_types"].setdefault("room_anomaly", {"severity": "high", "escalate_after_minutes": 2})
print("設定檔收的事件類型：", list(profile["event_types"]))
print("改了哪一個：", sorted(set(profile["event_types"]) - set(before["event_types"])))
r = requests.post(f"{url}/events", json=event, headers=KEY, timeout=5)
print(r.status_code, r.json())

print("\n== 5. 分數離門檻多遠，決定平台怎麼處理 ==")
rows = []
for s in [threshold - 0.2, threshold + 0.1, threshold + 0.25, threshold + 0.4, threshold + 1.0]:
    body = requests.post(f"{url}/events", json={**event, "confidence": to_confidence(s, threshold),
                                                "detail": {**event["detail"], "score": round(s, 3)}},
                         headers=KEY, timeout=5).json()
    rows.append({"score": round(s, 3), "超過門檻": round(s - threshold, 3), "confidence": to_confidence(s, threshold),
                 "alert_status": body["alert_status"] or "（只記錄）"})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 6. 工廠線也一樣：defect_detected 先加進 factory.json ==")
factory = json.loads((ROOT / "story" / "profiles" / "factory.json").read_text(encoding="utf-8"))
furl = platform_mock.start(factory)
defect = {**event, "event_type": "defect_detected", "location": "aisle_01", "source": "camera_line_01"}
print("加之前 →", requests.post(f"{furl}/events", json=defect, headers=KEY, timeout=5).status_code)
factory["event_types"]["defect_detected"] = {"severity": "medium", "escalate_after_minutes": 10}
print("加之後 →", requests.post(f"{furl}/events", json=defect, headers=KEY, timeout=5).status_code)

print("\n== 7. GET /alerts：健康線平台上的告警 ==")
alerts = pd.DataFrame(requests.get(f"{url}/alerts", timeout=5).json())
print(alerts.to_string(index=False))

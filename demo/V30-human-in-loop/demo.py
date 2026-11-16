"""V30｜人在迴路：值班的人按下的 resolved／dismissed，就是上線後的標籤。

執行（在 repo 根目錄）：python demo/V30-human-in-loop/demo.py
資料是手打的 10 筆告警，以及固定亂數種子產生的 400 筆模擬分數（不需要影像）。
"""
import numpy as np
import pandas as pd

print("== 1. 平台上的 10 筆告警：值班的人處理到哪裡 ==")
alerts = pd.DataFrame({
    "id": range(1, 11),
    "score": [4.9, 4.6, 4.4, 4.4, 4.3, 4.3, 4.2, 4.2, 4.1, 4.1],
    "status": ["resolved", "resolved", "resolved", "dismissed", "resolved",
               "acked", "dismissed", "open", "resolved", "open"],
})
print(alerts.to_string(index=False))
print(alerts["status"].value_counts().to_string())

print("\n== 2. 真告警率：分母只算「已經確認」的 ==")
done = alerts[alerts["status"].isin(["resolved", "dismissed"])]
wrong = (alerts["status"] == "resolved").sum() / len(alerts)
print(f"已確認 {len(done)} 筆：resolved {(done['status'] == 'resolved').sum()}、dismissed {(done['status'] == 'dismissed').sum()}")
print(f"真告警率 = resolved ÷ 已確認 = {(done['status'] == 'resolved').mean():.3f}")
print(f"錯的算法 resolved ÷ 全部 = {wrong:.3f}（把還沒處理的 open、acked 當成誤報）")

print("\n== 3. 標籤只會出現在「有叫的」那一邊 ==")
rng = np.random.default_rng(0)
truth = rng.random(400) < 0.3                                  # 3 成是真的異常
score = np.where(truth, rng.normal(5.0, 0.6, 400), rng.normal(3.6, 0.5, 400))
for th in [4.0, 4.4]:
    sent = score > th
    print(f"門檻 {th}：送出 {sent.sum():3d} 筆，人確認後 resolved {(sent & truth).sum():3d}；"
          f"沒送的 {(~sent).sum()} 筆裡藏著真的 {(~sent & truth).sum():2d} 筆（沒人看見）")

print("\n== 4. 讓系統自己調門檻：每週把 dismissed 太多就往上調 ==")
th = 4.0
print("週  門檻  送出  dismissed  真告警率  看不見的漏報  真正的 recall")
for week in range(1, 6):
    sent = score > th
    dismissed = (sent & ~truth).sum()
    rate = (sent & truth).sum() / sent.sum()
    print(f"{week:>2}  {th:4.1f}  {sent.sum():4d}  {dismissed:9d}  {rate:8.3f}  {(~sent & truth).sum():12d}  {(sent & truth).sum() / truth.sum():12.3f}")
    if rate < 0.98:                                           # 只看得到 dismissed，就只會往上調
        th += 0.2

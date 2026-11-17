"""V22｜pandas shift：去抖動（連續的 True 只算一次）與冷卻（通知後一段時間內不再通知）。

執行（在 repo 根目錄）：python demo/V22-debounce/demo.py
只用骨架資料 story/data/care/pose/poses.csv（程式模擬的骨架），不需要 MVTec。
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POSES = ROOT / "story" / "data" / "care" / "pose" / "poses.csv"

print("== 1. shift：每一格和「上一格」並排 ==")
flag = pd.Series([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=bool)
t = pd.DataFrame({"flag": flag, "上一格": flag.shift(fill_value=False)})
t["開頭"] = t["flag"] & ~t["上一格"]
print(t.astype(int).T.to_string())
print(f"True 的格數 {int(flag.sum())}；連續段的開頭 {int(t['開頭'].sum())} 個，在第 {np.flatnonzero(t['開頭']).tolist()} 格")

print("\n== 2. cumsum：替每一段編號，算每段多長 ==")
# 段編號 0＝不在任何一段裡，drop(0) 把它拿掉
run_id = t["開頭"].cumsum().where(flag)
lengths = flag.groupby(run_id.fillna(0).astype(int)).size().drop(0)
kept = lengths[lengths >= 2]
print("段編號：", run_id.fillna(0).astype(int).tolist())
print("每段長度：", lengths.to_dict())
print("只留至少 2 格的段（去掉一閃而過的）：", kept.index.tolist())

print("\n== 3. 冷卻：通知之後 cooldown 格內，新的開頭不再通知 ==")


def notify(flags, cooldown):
    s = pd.Series(flags).astype(bool)
    starts = np.flatnonzero((s & ~s.shift(fill_value=False)).to_numpy())
    sent, last = [], None
    for i in starts:
        if last is None or i - last >= cooldown:
            sent.append(int(i))
            last = i
    return sent


for cd in [1, 5, 8]:
    print(f"cooldown {cd}：通知在第 {notify(flag, cd)} 格")

print("\n== 4. 真的骨架串流：336 段接成 22 分鐘（10 fps）==")
df = pd.read_csv(POSES, keep_default_na=False, dtype={"alert": str})
order = np.random.default_rng(0).permutation(df["seq_id"].unique())
df = df.set_index("seq_id").loc[order].reset_index()        # 336 段隨機排成一條串流
sx = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
sy = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
hx = (df["left_hip_x"] + df["right_hip_x"]) / 2
hy = (df["left_hip_y"] + df["right_hip_y"]) / 2
df["angle"] = np.degrees(np.arctan2(np.abs(sx - hx), np.abs(hy - sy)))
df["low"] = df["angle"] >= 55                                 # 逐格旗標：這一格是低姿勢
falls = df[df["alert"] == "1"].groupby("seq_id", sort=False).apply(lambda s: (s.index.min(), s.index.max()),
                                                                   include_groups=False)
print(f"{len(df)} 格；跌倒 {len(falls)} 段；逐格旗標為 True 的 {int(df['low'].sum())} 格")


def caught(sent):
    """通知落在某一段跌倒裡（或它結束後 2 秒內），那段就算有叫到人。"""
    return sum(any(a <= i <= b + 20 for i in sent) for a, b in falls)


def false_calls(sent):
    """不在任何一段跌倒裡的通知：誤報。"""
    return sum(not any(a <= i <= b + 20 for a, b in falls) for i in sent)


s = df["low"]
starts = np.flatnonzero((s & ~s.shift(fill_value=False)).to_numpy())
run_id = (s & ~s.shift(fill_value=False)).cumsum().where(s)
long_runs = s.groupby(run_id).size()
held = run_id.map(long_runs >= 10).fillna(False).astype(bool)   # 只留連續 ≥ 10 格（1 秒）的段

plans = [("逐格：True 就通知", np.flatnonzero(s.to_numpy())), ("去抖動：每段開頭通知一次", starts),
         ("＋只留連續 ≥ 1 秒的段", notify(held, 1))]
plans += [(f"＋冷卻 {cd_s} 秒", notify(held, cd_s * 10)) for cd_s in [30, 60, 300]]
rows = [{"做法": name, "通知次數": len(sent), "叫到的跌倒": caught(sent), "誤報": false_calls(sent)} for name, sent in plans]
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 5. 冷卻吞掉的是哪幾段 ==")
base, cool = notify(held, 1), notify(held, 3000)
lost = [(sid, a) for sid, (a, b) in falls.items() if any(a <= i <= b + 20 for i in base)
        and not any(a <= i <= b + 20 for i in cool)]
for sid, a in lost[:4]:
    prev = max(i for i in cool if i < a)
    kind = "誤報" if false_calls([prev]) else "真的"
    print(f"{sid} 在 {a / 600:4.1f} 分跌倒；上一次通知在 {prev / 600:4.1f} 分（{kind}），還在 300 秒冷卻裡")
print(f"共 {len(lost)} 段跌倒，沒冷卻時叫得到、冷卻 300 秒就叫不到")

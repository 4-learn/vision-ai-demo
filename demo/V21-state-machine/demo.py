"""V21｜dict 轉移表：狀態機。把「1 秒內大變化」變成「確定出事了」。

執行（在 repo 根目錄）：python demo/V21-state-machine/demo.py
只用骨架資料 story/data/{factory,care}/pose/poses.csv（程式模擬的骨架），不需要 MVTec。
"""
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "story" / "data"

print("== 1. 轉移表：目前狀態 → {事件: 下一個狀態} ==")
TRANSITIONS = {
    "normal": {"big_change": "watching"},
    "watching": {"still_down": "alert", "recovered": "normal"},
    "alert": {"acked": "normal"},
}

def step(state, event):
    return TRANSITIONS.get(state, {}).get(event, state)


state = "normal"
for event in ["still_down", "big_change", "big_change", "still_down", "recovered", "acked"]:
    new = step(state, event)
    print(f"{state:<9} + {event:<11} → {new}{'' if new != state else '（表上沒有，不變）'}")
    state = new

print("\n== 2. 把骨架串流接上狀態機（為跌倒設計的規則）==")


TORSO = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def load(line):
    df = pd.read_csv(DATA / line / "pose" / "poses.csv", keep_default_na=False, dtype={"alert": str})
    df["seen"] = (df[[f"{p}_c" for p in TORSO]] >= 0.3).all(axis=1)       # 肩、髖四點都看得到
    sx = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    sy = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    hx = (df["left_hip_x"] + df["right_hip_x"]) / 2
    hy = (df["left_hip_y"] + df["right_hip_y"]) / 2
    df["angle"] = np.degrees(np.arctan2(np.abs(sx - hx), np.abs(hy - sy)))   # ML 課 30 節：整欄一次算
    return df


def run(angles, change_min=40, down_min=55, hold=10):
    """逐格：1 秒內變化 ≥ change_min → watching；之後連續 hold 格角度 ≥ down_min → alert。
    回傳每一格的狀態。角度是 NaN（這格看不清楚）就跳過。"""
    window, state, low, history = deque(maxlen=10), "normal", 0, []
    for a in angles:
        if np.isnan(a):
            history.append(state)
            continue
        window.append(a)
        change = max(window) - min(window) if len(window) == window.maxlen else 0.0
        low = low + 1 if a >= down_min else 0
        if state == "normal" and change >= change_min:
            state = step(state, "big_change")
        elif state == "watching" and low >= hold:
            state = step(state, "still_down")
        elif state == "watching" and low == 0 and change < 10:
            state = step(state, "recovered")
        history.append(state)
    return history


care = load("care")
seq = care[care["seq_id"] == care[care["alert"] == "1"]["seq_id"].unique()[1]]
hist = run(seq["angle"].to_numpy())
changes = [(i, s) for i, s in enumerate(hist) if i == 0 or s != hist[i - 1]]
print(f"健康線 {seq['seq_id'].iloc[0]}（{seq['activity'].iloc[0]}，標註跌倒從第 {seq['fall_frame'].iloc[0]} 格開始）")
print("狀態改變的格：", [f"第 {i} 格 {s}" for i, s in changes])
print("每 5 格的角度：", seq["angle"].round(0).astype(int).tolist()[::5])


def score(df, skip_unseen=False, **rule):
    """每一段跑一次狀態機：每段一列，y＝答案、hit＝狀態機有沒有進 alert。"""
    rows = []
    for _, s in df.groupby("seq_id"):
        angles = s["angle"].where(s["seen"]) if skip_unseen else s["angle"]
        rows.append({"activity": s["activity"].iloc[0], "y": s["alert"].max() == "1",
                     "hit": "alert" in run(angles.to_numpy(), **rule)})
    return pd.DataFrame(rows)


print("\n== 3. 同一套規則，兩條線 ==")
factory = load("factory")
for line, df in [("care", care), ("factory", factory)]:
    r = score(df)
    tp, n1 = int((r["y"] & r["hit"]).sum()), int(r["y"].sum())
    print(f"{line:<8} 抓到 {tp:>2}/{n1}，誤報 {int((~r['y'] & r['hit']).sum())} 段")
    print("   各動作被狀態機叫的段數：", r[r["hit"]].groupby("activity").size().to_dict())

print("\n== 4. 誤報從哪來：沒看到的點 ==")
for line, df in [("care", care), ("factory", factory)]:
    r = score(df, skip_unseen=True)
    print(f"{line:<8} 跳過看不清楚的格：抓到 {int((r['y'] & r['hit']).sum()):>2}/{int(r['y'].sum())}，"
          f"誤報 {int((~r['y'] & r['hit']).sum())} 段；各動作 {r[r['hit']].groupby('activity').size().to_dict()}")

print("\n== 5. 為什麼工廠線抓不到：彎腰多久 ==")
bend = factory[factory["alert"] == "1"]
low_run = []
for _, s in bend.groupby("seq_id"):
    a = s["angle"].to_numpy() >= 55
    best = cur = 0
    for v in a:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    low_run.append(best)
print(f"44 段彎腰搬重物，標註 bending 的格數：{bend[bend['posture'] == 'bending'].groupby('seq_id').size().agg(['min', 'median', 'max']).tolist()}")
print(f"角度 ≥ 55 度「連續」最多幾格：中位數 {int(np.median(low_run))}，達到 10 格的只有 {sum(x >= 10 for x in low_run)} 段")

print("\n== 6. 參數改一改：抓到與誤報一起動 ==")
rows = []
for down_min in [45, 55]:
    for hold in [3, 5, 10]:
        row = {"down_min": down_min, "hold": hold}
        for line, df in [("care", care), ("factory", factory)]:
            r = score(df, down_min=down_min, hold=hold)
            row[f"{line} 抓到"] = int((r["y"] & r["hit"]).sum())
            row[f"{line} 誤報"] = int((~r["y"] & r["hit"]).sum())
        rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 7. 換一張轉移表：彎下去、又站起來才算一次搬重物 ==")
LIFT = {
    "normal": {"bent": "bending"},
    "bending": {"stood_up": "alert"},
    "alert": {"acked": "normal"},
}


def run_lift(angles, down_min=45, hold=5, up_max=25):
    state, low = "normal", 0
    for a in angles:
        low = low + 1 if a >= down_min else 0
        if state == "normal" and low >= hold:
            state = LIFT[state]["bent"]
        elif state == "bending" and a < up_max:
            state = LIFT[state]["stood_up"]
    return state == "alert"


rows = []
for hold in [3, 5]:
    hit = factory.groupby("seq_id").apply(lambda s: (s["alert"].max() == "1", run_lift(s["angle"].to_numpy(), hold=hold)),
                                          include_groups=False)
    tp = sum(y and h for y, h in hit)
    fp = sum((not y) and h for y, h in hit)
    rows.append({"hold": hold, "抓到": f"{tp}/44", "誤報": fp})
print(pd.DataFrame(rows).to_string(index=False))

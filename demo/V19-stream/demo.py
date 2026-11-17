"""V19｜Python generator＋deque：骨架變成一格一格流進來的串流。

執行（在 repo 根目錄）：python demo/V19-stream/demo.py
只用骨架資料 story/data/{factory,care}/pose/poses.csv（程式模擬的骨架，沿用 ML 課 29–30 節），不需要 MVTec。
"""
from collections import deque
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "story" / "data"
TORSO = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def load(line):
    return pd.read_csv(DATA / line / "pose" / "poses.csv", keep_default_na=False, dtype={"alert": str})


def torso_angle(row):
    """髖→肩這條線和垂直線差幾度（ML 課 30 節）：0＝直立、90＝躺平。"""
    sx = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    sy = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hx = (row["left_hip_x"] + row["right_hip_x"]) / 2
    hy = (row["left_hip_y"] + row["right_hip_y"]) / 2
    return float(np.degrees(np.arctan2(abs(sx - hx), abs(hy - sy))))


print("== 1. yield：要一格，才拍一格 ==")


def camera(n):
    for i in range(n):
        print(f"  （攝影機拍第 {i} 格）")
        yield i

g = camera(3)
print("建好了，還沒拍任何一格：", type(g).__name__)
print("next(g) →", next(g))
print("next(g) →", next(g))
for x in g:
    print("for 拿到", x)
print("走完之後再 list(g) →", list(g))

print("\n== 2. deque(maxlen=3)：滿了自動丟掉最舊的 ==")
w = deque(maxlen=3)
for x in [10, 20, 30, 40, 50]:
    w.append(x)
    print(f"append({x}) → {list(w)}，滿了：{len(w) == w.maxlen}")


print("\n== 3. 一段彎腰搬重物，一格一格流進來 ==")
factory = load("factory")
seq_id = factory[factory["alert"] == "1"]["seq_id"].iloc[0]
seq = factory[factory["seq_id"] == seq_id]
bending = seq[seq["posture"] == "bending"]["t"]
print(f"{seq_id}：{len(seq)} 格（10 fps、4 秒），標註彎腰的格在 t = {bending.min()}～{bending.max()} 秒")
rows = []


def frames(seq):
    for _, row in seq.iterrows():
        yield row

def angles(stream, min_conf=None):
    for row in stream:
        if min_conf is not None and any(row[f"{p}_c"] < min_conf for p in TORSO):
            continue
        yield row["t"], torso_angle(row)

window = deque(maxlen=10)
for t, a in angles(frames(seq)):
    window.append(a)
    change = max(window) - min(window) if len(window) == window.maxlen else None
    rows.append({"t": t, "角度": round(a, 1), "過去 1 秒變化": None if change is None else round(change, 1)})
print(pd.DataFrame(rows).iloc[::4].to_string(index=False))


def first_big_change(stream, n=10, change_min=40):
    """邊收邊算：視窗滿 n 格之後，最大−最小 ≥ change_min 的第一格。"""
    window = deque(maxlen=n)
    for t, a in stream:
        window.append(a)
        if len(window) == window.maxlen and max(window) - min(window) >= change_min:
            return t, round(max(window) - min(window), 1)
    return None, None


print("第一次「1 秒內變了 40 度以上」：t = %s 秒（變化 %s 度）" % first_big_change(angles(frames(seq))))

print("\n== 4. 沒看到的點是 (0, 0)：先跳過，再判斷 ==")
row0 = seq.iloc[0]
print(f"第 0 格的左肩 ({row0['left_shoulder_x']}, {row0['left_shoulder_y']})，信心值 {row0['left_shoulder_c']}")
print(f"信心值 < 0.3 的格跳過：40 格剩 {sum(1 for _ in angles(frames(seq), 0.3))} 格")
print("第一次「1 秒內變了 40 度以上」：t = %s 秒（變化 %s 度）" % first_big_change(angles(frames(seq), 0.3)))

print("\n== 5. islice：同一段，取樣頻率變低 ==")
for step, name in [(1, "10 fps"), (10, "1 fps"), (30, "每 3 秒一張")]:
    got = [(t, round(a, 1)) for t, a in angles(islice(frames(seq), 0, None, step), 0.3)]
    print(f"{name:<8} 看到 {len(got):>2} 格，最大−最小 {max(a for _, a in got) - min(a for _, a in got):5.1f} 度；"
          f"前幾格 {got[:4]}")

print("\n== 6. 336 段都跑一次：各有幾段「看得到 1 秒內大變化」 ==")
for line in ["factory", "care"]:
    df = load(line)
    rows = []
    for min_conf, how in [(None, "不跳過"), (0.3, "跳過 < 0.3")]:
        for step, name in [(1, "10 fps"), (10, "1 fps"), (30, "每 3 秒一張")]:
            hit = {"1": 0, "0": 0}
            for _, s in df.groupby("seq_id"):
                got = [a for _, a in angles(islice(frames(s), 0, None, step), min_conf)]
                hit[s["alert"].max()] += len(got) > 1 and max(got) - min(got) >= 40
            rows.append({"關鍵點": how, "取樣": name, "要注意的段": hit["1"], "正常的段": hit["0"]})
    n1 = (df.groupby("seq_id")["alert"].max() == "1").sum()
    print(f"{line}：要注意的 {n1} 段、正常的 {336 - n1} 段")
    print(pd.DataFrame(rows).to_string(index=False))

"""V29｜Workshop：骨架串流 MVP。一個人一個監測器（類別記住狀態），只在「剛進入 alert」時發事件。

執行（在 repo 根目錄）：python demo/V29-mvp-pose/demo.py
用健康線的骨架資料 story/data/care/pose/poses.csv（10 fps、每段 4 秒、40 格）。
"""
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
poses = pd.read_csv(ROOT / "story" / "data" / "care" / "pose" / "poses.csv",
                    keep_default_na=False, dtype={"alert": str})


def torso_angle(row):
    """肩膀中點到髖中點的連線，和垂直線差幾度：站直約 0、躺平約 90。"""
    sx = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    sy = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hx = (row["left_hip_x"] + row["right_hip_x"]) / 2
    hy = (row["left_hip_y"] + row["right_hip_y"]) / 2
    return float(np.degrees(np.arctan2(abs(sx - hx), abs(hy - sy))))


class Monitor:
    """一個人的監測器：記住最近 10 格的角度、低姿勢連續幾格、現在的狀態。"""

    def __init__(self, person_id):
        self.person_id = person_id
        self.angles = deque(maxlen=10)
        self.state, self.low = "normal", 0

    def push(self, angle):
        """餵一格，回傳這一格之後的狀態。"""
        self.angles.append(angle)
        change = max(self.angles) - min(self.angles) if len(self.angles) == 10 else 0.0
        self.low = self.low + 1 if angle >= 55 else 0
        if self.state == "normal" and change >= 40:
            self.state = "watching"
        elif self.state == "watching" and self.low >= 10:
            self.state = "alert"
        elif self.state == "watching" and self.low == 0 and change < 10:
            self.state = "normal"
        return self.state


def run(g):
    m = Monitor(g["subject_id"].iloc[0])
    return [m.push(torso_angle(r)) for _, r in g.iterrows()]


falls = poses[poses["alert"] == "1"]
sid = next(s for s, g in falls.groupby("seq_id") if "alert" in run(g))   # 第一段狀態機抓得到的跌倒
seq = falls[falls["seq_id"] == sid]
angles = [torso_angle(r) for _, r in seq.iterrows()]

print(f"== 1. 一段跌倒（{seq['seq_id'].iloc[0]}，{len(seq)} 格）：每 4 格印一次 ==")
mon = Monitor(seq["subject_id"].iloc[0])
states = [mon.push(a) for a in angles]
for i in range(0, len(angles), 4):
    print(f"第 {i:2d} 格  軀幹角度 {angles[i]:5.1f}°  狀態 {states[i]}")

print("\n== 2. 什麼時候發事件：「是 alert 就發」vs「剛進入 alert 才發」==")
level = sum(s == "alert" for s in states)
edge = sum(s == "alert" and (i == 0 or states[i - 1] != "alert") for i, s in enumerate(states))
print(f"狀態是 alert 的格數：{level} → 每格都發，就送 {level} 筆")
print(f"剛進入 alert（上一格不是 alert）：{edge} 筆")

print("\n== 3. 兩個人交錯進來：共用一個監測器 vs 一人一個 ==")
other = poses[(poses["activity"] == "walk") & (poses["subject_id"] != seq["subject_id"].iloc[0])]
other = other[other["seq_id"] == other["seq_id"].iloc[0]]
mixed = []
for (_, a), (_, b) in zip(seq.iterrows(), other.iterrows()):
    mixed += [(a["subject_id"], torso_angle(a)), (b["subject_id"], torso_angle(b))]
shared = Monitor("shared")
shared_states = [shared.push(angle) for _, angle in mixed]
own_states = []
monitors = {}
for pid, angle in mixed:
    monitors.setdefault(pid, Monitor(pid))
    own_states.append((pid, monitors[pid].push(angle)))
print(f"共用一個：最後狀態 {shared_states[-1]}，alert 格數 {shared_states.count('alert')}")
for pid, m in monitors.items():
    print(f"一人一個：{pid} 最後狀態 {m.state}，alert 格數 {sum(1 for p, s in own_states if p == pid and s == 'alert')}")

print("\n== 4. 336 段全部跑一次（每段一個新的監測器）==")
alert_frames = events = real = 0
for _, g in poses.groupby("seq_id"):
    states = run(g)
    alert_frames += states.count("alert")
    n = sum(s == "alert" and (i == 0 or states[i - 1] != "alert") for i, s in enumerate(states))
    events += n
    real += n if (g["alert"] == "1").any() else 0
print(f"alert 格數 {alert_frames}；剛進入 alert 的次數 {events}，其中在真的跌倒段裡 {real}")

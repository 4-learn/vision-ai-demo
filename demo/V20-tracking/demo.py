"""V20｜scipy linear_sum_assignment：同一個畫面好幾個人，逐格配對、給每個人固定的編號。

執行（在 repo 根目錄）：python demo/V20-tracking/demo.py
只用骨架資料 story/data/care/pose/poses.csv（程式模擬的骨架），不需要 MVTec。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
POSES = ROOT / "story" / "data" / "care" / "pose" / "poses.csv"

print("== 1. np.linalg.norm：上一格 2 人 × 這一格 2 人的距離矩陣 ==")
# 上一格：A 在左、B 在右（髖的中點，像素）；這一格：偵測器這次先吐出右邊那個人
prev = np.array([[100.0, 300.0], [400.0, 300.0]])
curr = np.array([[395.0, 305.0], [108.0, 298.0]])
cost = np.linalg.norm(prev[:, None, :] - curr[None, :, :], axis=2)
print("prev[:, None, :] 形狀", prev[:, None, :].shape, "；curr[None, :, :] 形狀", curr[None, :, :].shape)
print(pd.DataFrame(cost.round(1), index=["上一格 A", "上一格 B"], columns=["這一格 #0", "這一格 #1"]).to_string())

print("\n== 2. linear_sum_assignment：總距離最小的配法 ==")
rows, cols = linear_sum_assignment(cost)
pair_cost = cost[rows, cols]
print("rows", rows.tolist(), "cols", cols.tolist())
for r, c in zip(rows, cols):
    print(f"上一格 {'AB'[r]} → 這一格 #{c}，距離 {cost[r, c]:.1f}")
print(f"照偵測器吐出的順序直接對（A→#0、B→#1）：總距離 {cost[0, 0] + cost[1, 1]:.1f}；配對後：{pair_cost.sum():.1f}")

print("\n== 3. 每個人各挑最近的，會搶同一個 ==")
prev3 = np.array([[100.0, 300.0], [160.0, 300.0]])     # A、B 靠得很近
curr3 = np.array([[150.0, 300.0], [220.0, 300.0]])     # 兩個人都往右走了 50～60 像素
cost3 = np.linalg.norm(prev3[:, None, :] - curr3[None, :, :], axis=2)
print(pd.DataFrame(cost3.round(1), index=["上一格 A", "上一格 B"], columns=["這一格 #0", "這一格 #1"]).to_string())
nearest = cost3.argmin(axis=1)          # 各挑最近
r3, c3 = linear_sum_assignment(cost3)   # 一起考慮
print("各挑最近（argmin）：", nearest.tolist(), "← A、B 都挑 #0")
print("linear_sum_assignment：", list(zip(r3.tolist(), c3.tolist())), f"總距離 {cost3[r3, c3].sum():.1f}")

print("\n== 4. 人數不一樣：多出來的沒配到；太遠的不算同一個人 ==")
prev4 = np.array([[100.0, 300.0], [400.0, 300.0]])
curr4 = np.array([[104.0, 302.0], [600.0, 280.0], [398.0, 296.0]])   # 第三個人從右邊走進來
cost4 = np.linalg.norm(prev4[:, None, :] - curr4[None, :, :], axis=2)
r4, c4 = linear_sum_assignment(cost4)
print("距離矩陣形狀", cost4.shape, "→ 配到", list(zip(r4.tolist(), c4.tolist())))
print("沒配到的（新來的人）：", sorted(set(range(len(curr4))) - set(c4.tolist())))
prev5 = np.array([[100.0, 300.0]])
curr5 = np.array([[500.0, 300.0]])                     # 上一格的人走出畫面，另一個人從遠處進來
cost5 = np.linalg.norm(prev5[:, None, :] - curr5[None, :, :], axis=2)
r5, c5 = linear_sum_assignment(cost5)
MAX_MOVE = 80                                           # 10 fps：0.1 秒內走不了 80 像素
print(f"只有一個人時一定配得上：距離 {cost5[r5, c5][0]:.1f}；超過 MAX_MOVE {MAX_MOVE} 就當成新的人")

print("\n== 5. 真的骨架：兩段骨架放進同一個畫面，偵測器每格吐出的順序都會亂 ==")
poses = pd.read_csv(POSES, keep_default_na=False, dtype={"alert": str})
ids = poses["seq_id"].unique()
a = poses[poses["seq_id"] == ids[0]].reset_index(drop=True)
b = poses[poses["seq_id"] == ids[1]].reset_index(drop=True).copy()
b[[c for c in b.columns if c.endswith("_x")]] += 80     # 第二個人往右挪 80 像素


def centers(frame):
    """每副骨架的中心：左右髖的中點；另外回傳「髖有一點沒看到（信心值 < 0.3）」。"""
    c = np.c_[(frame["left_hip_x"] + frame["right_hip_x"]) / 2, (frame["left_hip_y"] + frame["right_hip_y"]) / 2]
    return c, ((frame["left_hip_c"] < 0.3) | (frame["right_hip_c"] < 0.3)).to_numpy()


def track(use_conf, seed=0):
    """回傳：編號錯的格數、編號對調的次數、髖沒看到的格數。truth[k]＝這一格第 k 個框其實是誰（0＝A、1＝B）。"""
    rng = np.random.default_rng(seed)
    wrong, switches, lost_n, ids_prev, c_prev, was_wrong = 0, 0, 0, None, None, False
    for i in range(len(a)):
        truth = rng.permutation(2)                      # 偵測器吐出的順序：每格隨機
        frame = pd.concat([a.iloc[[i]], b.iloc[[i]]]).iloc[truth].reset_index(drop=True)
        c, lost = centers(frame)
        lost_n += int(lost.any())
        if i == 0:
            ids_now = truth.copy()                      # 第 1 格：直接給編號
        else:
            # 配對：這一格的框 col 繼承上一格 r 的編號
            cost = np.linalg.norm(c_prev[:, None, :] - c[None, :, :], axis=2)
            r, col = linear_sum_assignment(cost)
            ids_now = np.empty(2, int)
            ids_now[col] = ids_prev[r]
            is_wrong = bool((ids_now != truth).any())
            wrong += is_wrong
            switches += is_wrong != was_wrong
            was_wrong = is_wrong
        # 記下每個編號這一格在哪裡；髖沒看到的，沿用上一格的位置
        by_id = np.empty((2, 2))
        by_id[ids_now] = c
        if c_prev is not None and use_conf:
            prev_by_id = np.empty((2, 2))
            prev_by_id[ids_prev] = c_prev
            by_id[ids_now[lost]] = prev_by_id[ids_now[lost]]
        c_prev = by_id[ids_now]
        ids_prev = ids_now
    return wrong, switches, lost_n


rng = np.random.default_rng(0)
naive = sum(int((rng.permutation(2) != np.arange(2)).any()) for _ in range(len(a) - 1))
print(f"{ids[0]}（原位）＋ {ids[1]}（往右 80 像素），{len(a)} 格；配對從第 2 格開始，共 {len(a) - 1} 次")
print(f"照吐出順序當編號：{naive} 格編號錯")
w, sw, lost_n = track(use_conf=False)
print(f"有 {lost_n} 格，某個人的髖沒看到（座標是 (0, 0)）")
print(f"用距離配對：{w} 格編號錯，編號對調 {sw} 次")
w, sw, _ = track(use_conf=True)
print(f"用距離配對＋髖沒看到就沿用上一格的位置：{w} 格編號錯，編號對調 {sw} 次")

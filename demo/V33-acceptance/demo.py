"""V33｜驗收報告：標準先寫、再跑。總分通過不代表每一類都好；測試集小時，一張就能改變結論。

執行（在 repo 根目錄）：python demo/V33-acceptance/demo.py
分數是固定亂數種子產生的模擬資料（不需要影像），類型名稱借用健康線。
"""
import json

import numpy as np
import pandas as pd

print("== 1. 先寫標準（跑之前就定好）==")
ACCEPTANCE = {"min_recall": 0.85, "max_false_rate": 0.1, "per_type_min_recall": 0.8}
print(json.dumps(ACCEPTANCE))

rng = np.random.default_rng(0)
plan = {"good": (30, 3.0), "fallen": (25, 5.0), "obstacle": (25, 3.9), "blocked": (15, 6.0)}
df = pd.concat([pd.DataFrame({"type": t, "score": rng.normal(mu, 0.5, n)}) for t, (n, mu) in plan.items()],
               ignore_index=True)
df["y"] = (df["type"] != "good").astype(int)
threshold = 3.6                                   # 門檻是之前在校正資料上定的，這裡不准動


def report(df, threshold, rules):
    pred = df["score"] > threshold
    recall, false_rate = pred[df["y"] == 1].mean(), pred[df["y"] == 0].mean()
    by_type = df.assign(pred=pred).groupby("type")["pred"].agg(["size", "sum"])
    by_type["比例"] = (by_type["sum"] / by_type["size"]).round(3)
    checks = {f"recall {recall:.3f} ≥ {rules['min_recall']}": recall >= rules["min_recall"],
              f"誤判率 {false_rate:.3f} ≤ {rules['max_false_rate']}": false_rate <= rules["max_false_rate"]}
    for t, row in by_type.drop(index="good").iterrows():
        checks[f"{t} recall {row['比例']:.3f} ≥ {rules['per_type_min_recall']}"] = row["比例"] >= rules["per_type_min_recall"]
    return by_type.rename(columns={"size": "張數", "sum": "判異常"}), checks


print("\n== 2. 再跑：照標準逐條打勾 ==")
table, checks = report(df, threshold, ACCEPTANCE)
print(table.to_string())
for k, ok in checks.items():
    print(f"[{'通過' if ok else '不通過'}] {k}")
print("結論：", "通過" if all(checks.values()) else "不通過")

print("\n== 3. 只看總分會漏掉什麼 ==")
only_total = {k: v for k, v in checks.items() if k.startswith(("recall", "誤判率"))}
print("只寫兩條總標準：", "通過" if all(only_total.values()) else "不通過")
print("但 obstacle 漏了", int(table.loc["obstacle", "張數"] - table.loc["obstacle", "判異常"]), "張")

print("\n== 4. 測試集很小：一張就差多少 ==")
good = int((df["y"] == 0).sum())
for n_false in range(2, 5):
    rate = n_false / good
    print(f"正常 {good} 張裡誤判 {n_false} 張 → 誤判率 {rate:.3f}（{'通過' if rate <= ACCEPTANCE['max_false_rate'] else '不通過'}）")

print("\n== 5. 跑完才改標準：數字好看了，但驗收已經沒有意義 ==")
moved = {**ACCEPTANCE, "per_type_min_recall": round(float(table.drop(index="good")["比例"].min()), 2)}
print("改過的標準：", json.dumps(moved))
print("結論：", "通過" if all(report(df, threshold, moved)[1].values()) else "不通過")

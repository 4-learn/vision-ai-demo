"""V13｜pandas groupby：成績表與錯誤分析。依類型分組算張數、抓到幾張，再把漏掉的依「離門檻多近」排出來。

執行（在 repo 根目錄）：python demo/V13-report/demo.py
資料是手打的 14 張螺帽分數（形狀和進度存檔的小格 kNN 分數一樣），不必讀圖、不必跑模型。
"""
import pandas as pd

THRESHOLD = 4.00
scores = pd.DataFrame({
    "image": ["good/01", "good/02", "good/03", "good/04", "bent/01", "bent/02", "bent/03",
              "color/01", "color/02", "color/03", "scratch/01", "scratch/02", "scratch/03", "scratch/04"],
    "score": [3.52, 3.61, 3.70, 4.08, 4.86, 4.52, 4.95, 4.71, 3.98, 3.62, 4.31, 3.96, 3.93, 3.55],
})

print("== 1. 先有一張「一張圖一列」的表 ==")
scores["type"] = scores["image"].str.split("/").str[0]
scores["y"] = (scores["type"] != "good").astype(int)
scores["pred"] = (scores["score"] > THRESHOLD).astype(int)
print(scores.head(6).to_string(index=False))
print(f"門檻 {THRESHOLD:.2f}；全部 {len(scores)} 張，抓到 {int(scores['pred'][scores['y'] == 1].sum())}/{int(scores['y'].sum())}")

print("\n== 2. groupby：依類型分組，每組各算一次 ==")
g = scores.groupby("type")
print(g["pred"].sum())
print(f"groupby 回傳的是 {type(g).__name__}，要再接 .sum()、.size()、.agg() 才會算")

print("\n== 3. agg：一次算好幾欄，欄名自己取 ==")
table = scores.groupby("type").agg(張數=("y", "size"), 判異常=("pred", "sum"), 平均分數=("score", "mean"))
table["抓到率"] = table["判異常"] / table["張數"]
print(table.round(3).to_string())

print("\n== 4. 漏掉的：離門檻多近 ==")
missed = scores[(scores["y"] == 1) & (scores["pred"] == 0)]
missed = missed.assign(差=(missed["score"] - THRESHOLD).round(3)).sort_values("差", ascending=False)
print(missed[["image", "score", "差"]].to_string(index=False))

print("\n== 5. 門檻往下移一點，能多抓幾張、多誤判幾張 ==")
for t in [4.00, 3.95, 3.90, 3.60]:
    pred = scores["score"] > t
    print(f"門檻 {t:.2f}：抓到 {int(pred[scores['y'] == 1].sum())}/{int(scores['y'].sum())}、"
          f"誤判 {int(pred[scores['y'] == 0].sum())}/{int((scores['y'] == 0).sum())}")

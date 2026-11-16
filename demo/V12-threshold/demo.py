"""V12｜scikit-learn roc_curve：異常分數的門檻怎麼選（只有良品 vs. 有少量異常）。"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

# 校正（全是良品）＋少量異常：score 是異常分數，越大越可疑
val = pd.DataFrame({
    "score": [3.1, 3.3, 3.4, 3.6, 3.7, 3.8, 4.0, 4.1, 4.3, 4.6,
              3.9, 4.2, 4.4, 4.8, 5.0, 5.3, 5.6, 6.1, 6.5, 7.2],
    "y":     [0] * 10 + [1] * 10,
})
y, score = val["y"], val["score"]

print("== 1. roc_curve：每個門檻下，異常抓到幾成、良品誤判幾成 ==")
fpr, tpr, thr = roc_curve(y, score)
curve = pd.DataFrame({"threshold": thr, "fpr（良品被誤判）": fpr, "tpr（異常被抓到＝recall）": tpr})
print(curve.to_string(index=False, float_format="%.3f"))
print(f"第一列門檻是 {thr[0]}：比所有分數都高，一張都不判異常（fpr 0、tpr 0 的起點）")

print("\n== 2. 方向：roc_curve 是「分數 >= 門檻」就判異常 ==")
t = 4.4
pred = score >= t
print(f"門檻 {t}：判異常 {int(pred.sum())} 張；異常抓到 {int(pred[y == 1].sum())}/10、良品誤判 {int(pred[y == 0].sum())}/10")
print(f"> 和 >= 的差別只在剛好等於門檻那一張：score == {t} 的有 {int((score == t).sum())} 張")

print("\n== 3. drop_intermediate：預設會省略「不會改變曲線形狀」的門檻 ==")
print(f"預設 {len(roc_curve(y, score)[2])} 個門檻；drop_intermediate=False 有 {len(roc_curve(y, score, drop_intermediate=False)[2])} 個")

print("\n== 4. 選門檻 A：只有良品 → 校正良品的最大值 ==")
t_good = float(score[y == 0].max())
pred = score > t_good
print(f"門檻 {t_good}：抓到 {int(pred[y == 1].sum())}/10、誤判 {int(pred[y == 0].sum())}/10")
print("保證校正良品一張都不誤判；但異常抓到幾張，完全看運氣")

print("\n== 5. 選門檻 B：有少量異常 → recall ≥ 0.9 裡，最高的門檻 ==")
ok = np.where(tpr >= 0.9)[0]
t_val = float(thr[ok[0]])
pred = score >= t_val
print(f"tpr >= 0.9 的第一列：門檻 {t_val}、tpr {tpr[ok[0]]:.3f}、fpr {fpr[ok[0]]:.3f}")
print(f"門檻 {t_val}：抓到 {int(pred[y == 1].sum())}/10、誤判 {int(pred[y == 0].sum())}/10")

print("\n== 6. 和 ML 課 12 節的 precision_recall_curve 比一比 ==")
p, r, pt = precision_recall_curve(y, score)
print(f"precision_recall_curve：{len(pt)} 個門檻、由低到高；precision 會受「異常占多少」影響")
print(f"roc_curve：{len(thr)} 個門檻、由高到低；fpr、tpr 各自只看良品、只看異常")
for ratio in [1, 10]:
    yy = np.r_[np.zeros(10 * ratio), np.ones(10)]
    ss = np.r_[np.tile(score[y == 0], ratio), score[y == 1]]
    f, tp_, th = roc_curve(yy, ss)
    i = np.where(tp_ >= 0.9)[0][0]
    pr, rc, pth = precision_recall_curve(yy, ss)
    j = np.where(pth == th[i])[0][0]
    print(f"良品 {10 * ratio} 張、異常 10 張：門檻 {th[i]} → tpr {tp_[i]:.3f}、fpr {f[i]:.3f}、precision {pr[j]:.3f}、"
          f"AUROC {roc_auc_score(yy, ss):.3f}")

print("\n== 7. 只有 10 張異常時，recall 一格就是 0.1 ==")
for need in [0.85, 0.9, 0.95]:
    i = np.where(tpr >= need)[0][0]
    print(f"要求 recall ≥ {need}：門檻 {thr[i]}、實際 tpr {tpr[i]:.3f}、fpr {fpr[i]:.3f}")
print("0.85 和 0.9 選到同一個門檻；0.95 就得抓到 10/10，門檻一口氣掉到 3.9")

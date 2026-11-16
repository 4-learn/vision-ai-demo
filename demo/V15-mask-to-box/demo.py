"""V15｜OpenCV connectedComponentsWithStats：瑕疵遮罩變成 YOLO 的框。再看 YOLO label 格式、依類型分層切、良品放空 label。

執行（在 repo 根目錄）：python demo/V15-mask-to-box/demo.py
工廠線的螺帽照片要先由教師執行 tools/build_data.py 產生（MVTec AD 衍生，不隨 repo 散佈）。
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANOMALY = ROOT / "story" / "data" / "factory" / "anomaly"
OUT = ROOT / "demo" / "V15-mask-to-box" / "out"

print("== 1. 手畫一張 12×12 的遮罩：兩塊白色、一個孤立的點 ==")
toy = np.zeros((12, 12), np.uint8)
toy[1:4, 1:5] = 255          # 第一塊：寬 4、高 3
toy[6:11, 7:10] = 255        # 第二塊：寬 3、高 5
toy[9, 2] = 255              # 一個點
n, labels, stats, centroids = cv2.connectedComponentsWithStats((toy > 127).astype(np.uint8))
print(f"n = {n}（0 號是背景，所以白色有 {n - 1} 塊）")
print(labels)

print("\n== 2. stats：每一塊的 x, y, 寬, 高, 面積 ==")
table = pd.DataFrame(stats, columns=["x", "y", "w", "h", "area"])
table["cx"], table["cy"] = centroids[:, 0].round(1), centroids[:, 1].round(1)
print(table.to_string())

print("\n== 3. 換成 YOLO label：類別 中心x 中心y 寬 高（都除以圖寬高）==")
H, W = toy.shape
for i in range(1, n):
    x, y, w, h, area = stats[i]
    keep = "保留" if area >= 2 else "太小，丟掉"
    print(f"第 {i} 塊 面積 {area:>2}：0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}  {keep}")

print("\n== 4. 真的遮罩：bent/000 有兩塊 ==")
mask = cv2.imread(str(ANOMALY / "ground_truth" / "bent" / "000_mask.png"), cv2.IMREAD_GRAYSCALE)
print(f"遮罩大小 {mask.shape}，只有兩種值 {np.unique(mask).tolist()}")
n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
for x, y, w, h, area in stats[1:]:
    print(f"x {x:>3} y {y:>3} 寬 {w:>3} 高 {h:>3} 面積 {area:>5} → 0 {(x + w / 2) / 256:.3f} {(y + h / 2) / 256:.3f} {w / 256:.3f} {h / 256:.3f}")
img = cv2.imread(str(ANOMALY / "test" / "bent" / "000.png"))
for x, y, w, h, area in stats[1:]:
    cv2.rectangle(img, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), (0, 0, 255), 1)
OUT.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT / "bent-000-boxes.png"), img)
print("畫好框的圖：demo/V15-mask-to-box/out/bent-000-boxes.png")

print("\n== 5. 各類型的遮罩有幾塊 ==")
rows = []
for kind in ["bent", "color", "flip", "scratch"]:
    for m in sorted((ANOMALY / "ground_truth" / kind).glob("*_mask.png")):
        g = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        k = cv2.connectedComponentsWithStats((g > 127).astype(np.uint8))[0] - 1
        rows.append({"type": kind, "blocks": k})
blocks = pd.DataFrame(rows)
print(blocks.groupby("type")["blocks"].agg(張數="size", 一塊的=lambda s: int((s == 1).sum()), 最多="max").to_string())

print("\n== 6. 依類型分層切 60／20／20：每一類各自切 ==")
kinds = ["bent", "color", "flip", "scratch", "good"]
rng = np.random.default_rng(0)
split_rows = []
for kind in kinds:
    files = sorted((ANOMALY / "test" / kind).glob("*.png"))
    order = rng.permutation(len(files))
    a, b = int(round(len(files) * 0.6)), int(round(len(files) * 0.8))
    for j, i in enumerate(order):
        split_rows.append({"type": kind, "file": files[i].name, "split": "train" if j < a else "val" if j < b else "test"})
split = pd.DataFrame(split_rows)
print(pd.crosstab(split["type"], split["split"])[["train", "val", "test"]].to_string())

print("\n== 7. 良品也要有 label 檔，只是內容是空的 ==")
lab = OUT / "labels"
shutil.rmtree(lab, ignore_errors=True)
lab.mkdir(parents=True)
(lab / "good-000.txt").write_text("")
(lab / "bent-000.txt").write_text("".join(f"0 {(x + w / 2) / 256:.6f} {(y + h / 2) / 256:.6f} {w / 256:.6f} {h / 256:.6f}\n"
                                          for x, y, w, h, _ in stats[1:]))
for p in sorted(lab.glob("*.txt")):
    print(f"{p.name}：{p.stat().st_size} bytes，{len(p.read_text().splitlines())} 行")

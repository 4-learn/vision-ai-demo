"""V26｜解析度、準確率、延遲：替 Pi 5 算一本預算。

執行（在 repo 根目錄）：
    python demo/V26-budget/demo.py
"""
import json
import os
import platform
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torchvision.models import ResNet18_Weights, resnet18

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.manual_seed(0)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
ANOMALY = ROOT / "story" / "data" / "care" / "anomaly"
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
SIZES = [160, 224, 320]


class Patch(torch.nn.Module):
    """和 V23 同一個模型：ResNet18 算到 layer3。"""

    def __init__(self):
        super().__init__()
        r = resnet18(weights=WEIGHTS).eval()
        self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.l2, self.l3 = r.layer2, r.layer3

    def forward(self, x):
        a = self.l2(self.stem(x))
        b = torch.nn.functional.interpolate(self.l3(a), size=a.shape[-2:], mode="bilinear")
        return torch.cat([a, b], 1)


print("== 1. 匯出時讓高、寬也可以變：一個 ONNX 跑三種解析度 ==")
path = OUT / "patch-anysize.onnx"
torch.onnx.export(Patch().eval(), torch.zeros(1, 3, 224, 224), str(path), input_names=["image"],
                  output_names=["features"], opset_version=17, dynamo=False,
                  dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "features": {0: "batch"}})
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
print("輸入 image：", sess.get_inputs()[0].shape)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(path, size):
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


goods = sorted((ANOMALY / "train" / "good").glob("*.png"))
memory = goods[:40]
image = ANOMALY / "test" / "obstacle" / "000.png"
rows = []
for size in SIZES:
    f = sess.run(None, {"image": preprocess(image, size)[None]})[0]
    rows.append({"輸入大小": size, "小格": f"{f.shape[2]}×{f.shape[3]}", "每張小格數": f.shape[2] * f.shape[3],
                 "每格原圖幾像素": round(256 / f.shape[2], 1)})
grid = pd.DataFrame(rows)
print(grid.to_string(index=False))

print("\n== 2. 解析度變大，kNN 要比的次數跟著變大 ==")
banks = {}
for size in SIZES:
    m = np.concatenate([sess.run(None, {"image": preprocess(p, size)[None]})[0] for p in memory])
    banks[size] = m.transpose(0, 2, 3, 1).reshape(-1, m.shape[1])
grid["記憶庫小格（40 張良品）"] = [len(banks[s]) for s in SIZES]
grid["每張要比幾次（百萬）"] = [round(n * len(banks[s]) / 1e6, 1) for n, s in zip(grid["每張小格數"], SIZES)]
print(grid[["輸入大小", "每張小格數", "記憶庫小格（40 張良品）", "每張要比幾次（百萬）"]].to_string(index=False))


def time_ms(fn, n=10, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return np.array(ts)


print("\n== 3. 在這台機器量：特徵＋kNN 一張圖要多久（中位數、p95，毫秒每次執行都不同）==")
for size in SIZES:
    nn = NearestNeighbors(n_neighbors=1).fit(banks[size])
    x = preprocess(image, size)[None]

    def one():
        f = sess.run(None, {"image": x})[0]
        nn.kneighbors(f.transpose(0, 2, 3, 1).reshape(-1, f.shape[1]))

    ts = time_ms(one)
    print(f"{size}：中位數 {np.median(ts):.0f} ms、p95 {np.percentile(ts, 95):.0f} ms")
print(f"這台機器：{platform.machine()}、{os.cpu_count()} 核、onnxruntime {ort.__version__}")

print("\n== 4. 預算：教師機的數字（story/v26-budget/budget.json）放進現場的節奏 ==")
budget = json.loads((ROOT / "story" / "v26-budget" / "budget.json").read_text(encoding="utf-8"))
LIMIT_MS = 3000   # EZCam 約 3 秒拍一張（ML 課 27 節）
SHARE = 0.5       # 只給異常偵測一半的時間：同一台 Pi 5 還要跑骨架串流、送事件
for line in ["factory", "care"]:
    t = pd.DataFrame(budget[line])
    t["Pi 5 最多可以慢幾倍"] = (LIMIT_MS * SHARE / t["每張毫秒（教師機）"]).round(1)
    print(f"{line}（預算 {LIMIT_MS * SHARE:.0f} ms）")
    print(t.to_string(index=False))
print("教師機不是 Pi 5：表上的毫秒只能拿來比大小，Pi 5 的數字要在 Pi 5 上用第 3 段的方法重量")

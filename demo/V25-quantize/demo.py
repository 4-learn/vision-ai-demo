"""V25｜onnxruntime.quantization：int8 動態量化。檔案變小了，但特徵、分數、速度要一個一個量。

執行（在 repo 根目錄）：
    python demo/V25-quantize/demo.py
"""
import logging
import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision.models import ResNet18_Weights, resnet18

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger().setLevel(logging.ERROR)   # quantize_dynamic 會建議先做 pre-processing，這裡先不做
torch.manual_seed(0)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
ANOMALY = ROOT / "story" / "data" / "care" / "anomaly"      # 健康線：寢室畫面（程式畫的合成畫面）
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


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


fp32 = OUT / "patch.onnx"
torch.onnx.export(Patch().eval(), torch.zeros(1, 3, 224, 224), str(fp32), input_names=["image"],
                  output_names=["features"], dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
                  opset_version=17, dynamo=False)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(path):
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (256, 256)), cv2.COLOR_BGR2RGB)
    return ((rgb[16:240, 16:240].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def mb(p):
    return p.stat().st_size / 1e6


print("== 1. quantize_dynamic：權重從 float32 換成 int8 ==")
int8 = OUT / "patch-int8.onnx"
quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
print(f"{fp32.name} {mb(fp32):.1f} MB → {int8.name} {mb(int8):.1f} MB（約 {mb(fp32) / mb(int8):.1f} 分之一）")

print("\n== 2. 打開來看：哪些運算被換掉了 ==")
for p in [fp32, int8]:
    ops = Counter(n.op_type for n in onnx.load(str(p)).graph.node)
    print(f"{p.name}：", "、".join(f"{k} {ops[k]}" for k in ["Conv", "ConvInteger", "DynamicQuantizeLinear"]))

print("\n== 3. 同一張圖，特徵差多少 ==")
s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
x = preprocess(ANOMALY / "test" / "fallen" / "000.png")[None]
a, b = s32.run(None, {"image": x})[0], s8.run(None, {"image": x})[0]
print(f"相關係數 {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.4f}；平均絕對差 {np.abs(a - b).mean():.4f}（特徵平均大小 {np.abs(a).mean():.4f}）")

print("\n== 4. 真正要問的：異常分數還分得開嗎（健康線，小格 kNN）==")
goods = sorted((ANOMALY / "train" / "good").glob("*.png"))
memory, calib = goods[:40], goods[40:60]
test = sorted((ANOMALY / "test").glob("*/*.png"))[::2]
y = np.array([0 if p.parent.name == "good" else 1 for p in test])


def features(sess, paths):
    xs = np.stack([preprocess(p) for p in paths])
    return np.concatenate([sess.run(None, {"image": xs[i:i + 8]})[0] for i in range(0, len(xs), 8)])


def scores(sess):
    """V10 的小格 kNN（縮小版）：回傳（校正良品的分數, 測試圖的分數）。"""
    m = features(sess, memory)
    bank = m.transpose(0, 2, 3, 1).reshape(-1, m.shape[1])
    nn = NearestNeighbors(n_neighbors=1).fit(bank[np.random.default_rng(0).choice(len(bank), 8000, replace=False)])
    out = []
    for paths in [calib, test]:
        f = features(sess, paths)
        d = nn.kneighbors(f.transpose(0, 2, 3, 1).reshape(-1, f.shape[1]))[0]
        out.append(np.sort(d.reshape(len(paths), -1), axis=1)[:, -10:].mean(1))
    return out


(c32, sc32), (c8, sc8) = scores(s32), scores(s8)
print(f"記憶 {len(memory)} 張、校正 {len(calib)} 張良品；測試 {len(test)} 張（異常 {y.sum()}）")
print(f"AUROC：fp32 {roc_auc_score(y, sc32):.3f}、int8 {roc_auc_score(y, sc8):.3f}")
print(f"門檻（校正良品最大值）：fp32 {c32.max():.3f}、int8 {c8.max():.3f}")
for name, sc, thr in [("fp32 分數 × fp32 門檻", sc32, c32.max()), ("int8 分數 × int8 門檻", sc8, c8.max()),
                      ("int8 分數 × fp32 門檻", sc8, c32.max())]:
    hit = sc > thr
    print(f"{name}：抓到 {hit[y == 1].sum()}/{y.sum()}、誤判 {hit[y == 0].sum()}/{(y == 0).sum()}")


def median_ms(sess, n=10):
    for _ in range(2):
        sess.run(None, {"image": x})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {"image": x})
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


print("\n== 5. 檔案變小，有變快嗎（單執行緒，中位數；毫秒每次執行都不同）==")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
t32 = median_ms(ort.InferenceSession(str(fp32), sess_options=opts, providers=["CPUExecutionProvider"]))
t8 = median_ms(ort.InferenceSession(str(int8), sess_options=opts, providers=["CPUExecutionProvider"]))
print(f"fp32 {t32:.0f} ms、int8 {t8:.0f} ms")
print("變小是一定的；變不變快要看這台 CPU 有沒有 int8 的加速指令，要在 Pi 5 上自己量")

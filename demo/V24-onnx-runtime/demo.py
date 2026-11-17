"""V24｜onnxruntime：不靠 PyTorch 跑 ONNX 模型，並且正確地量延遲（暖機、多次、中位數）。

執行（在 repo 根目錄）：
    python demo/V24-onnx-runtime/demo.py
"""
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import ResNet18_Weights, resnet18

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.manual_seed(0)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
IMAGE = ROOT / "story" / "data" / "care" / "anomaly" / "test" / "obstacle" / "000.png"   # 健康線：走道有障礙物


class Patch(torch.nn.Module):
    """和 V23 同一個模型：ResNet18 算到 layer3。這裡只為了做出 patch.onnx。"""

    def __init__(self):
        super().__init__()
        r = resnet18(weights=WEIGHTS).eval()
        self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.l2, self.l3 = r.layer2, r.layer3

    def forward(self, x):
        a = self.l2(self.stem(x))
        b = torch.nn.functional.interpolate(self.l3(a), size=a.shape[-2:], mode="bilinear")
        return torch.cat([a, b], 1)


model = Patch().eval()
path = OUT / "patch.onnx"
torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(path), input_names=["image"], output_names=["features"],
                  dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}}, opset_version=17, dynamo=False)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2RGB)
    crop = rgb[16:240, 16:240].astype(np.float32) / 255
    return ((crop - MEAN) / STD).transpose(2, 0, 1)[None]


print("== 1. InferenceSession：載入 ONNX，看它要什麼輸入 ==")
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
for a in sess.get_inputs():
    print(a.name, a.shape, a.type)
for a in sess.get_outputs():
    print(a.name, a.shape, a.type)
print("使用的 provider：", sess.get_providers())

print("\n== 2. 前處理只用 NumPy：和 torchvision 的結果一樣嗎 ==")
bgr = cv2.imread(str(IMAGE))
x = preprocess(bgr)
ref = WEIGHTS.transforms()(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)).unsqueeze(0).numpy()
print(f"x：shape {x.shape}、dtype {x.dtype}；和 torchvision 最大差 {np.abs(x - ref).max():.6f}")

print("\n== 3. sess.run：餵 dict、拿回 list ==")
outputs = sess.run(None, {"image": x})
feats = sess.run(["features"], {"image": x})[0]
try:
    sess.run(None, {"image": x.astype(np.float64)})
except Exception as e:
    err = type(e).__name__
print(f"sess.run(None, ...) 回傳 {type(outputs).__name__}，{len(outputs)} 個；第一個 shape {outputs[0].shape}")
with torch.no_grad():
    pt = model(torch.from_numpy(x)).numpy()
print(f"和 PyTorch 的相對誤差 {np.abs(feats - pt).max() / np.abs(pt).max():.1e}")
print(f"餵 float64 → {err}：模型要的是 tensor(float)，也就是 float32")


def time_it(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return np.array(ts)


print("\n== 4. 量延遲：第一次、之後 20 次的中位數、最慢一次（毫秒，每次執行都不同）==")
cold = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
t = time.perf_counter()
cold.run(None, {"image": x})
first = (time.perf_counter() - t) * 1000
ts = time_it(lambda: cold.run(None, {"image": x}))
print(f"第一次 {first:.0f} ms；中位數 {np.median(ts):.0f} ms；平均 {ts.mean():.0f} ms；最慢 {ts.max():.0f} ms")

print("\n== 5. 執行緒數：Pi 5 有 4 核，模擬「只給 1 核」和「給 4 核」==")
for n in [1, 4]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n
    s = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"intra_op_num_threads={n}：中位數 {np.median(time_it(lambda: s.run(None, {'image': x}))):.0f} ms")

print("\n== 6. 一次幾張：batch 1 × 4 次 vs batch 4 × 1 次（每張平均）==")
x4 = np.repeat(x, 4, axis=0)
one = np.median(time_it(lambda: [sess.run(None, {"image": x}) for _ in range(4)], n=5)) / 4
four = np.median(time_it(lambda: sess.run(None, {"image": x4}), n=5)) / 4
print(f"一張一張送：每張 {one:.0f} ms；四張一起送：每張 {four:.0f} ms")
print("數字每台機器、每次執行都不同；要比較的是同一台機器、同一次執行裡的相對大小")

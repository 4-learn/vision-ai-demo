"""V23｜torch.onnx.export：把「ResNet18 算到 layer3」包成一個模型，匯出 ONNX，再驗收和 PyTorch 一樣。

執行（在 repo 根目錄）：
    python demo/V23-onnx-export/demo.py
"""
import os
import shutil
import warnings
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")   # ultralytics 不要印進度

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision.models import ResNet18_Weights, resnet18

warnings.filterwarnings("ignore", category=DeprecationWarning)   # torch 2.x 提醒舊版匯出器，這門課先用它
torch.manual_seed(0)
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
IMAGE = ROOT / "story" / "data" / "care" / "anomaly" / "test" / "fallen" / "000.png"   # 健康線：倒地的寢室畫面


class Patch(torch.nn.Module):
    """ResNet18 只算到 layer3：layer2 和放大後的 layer3 疊在一起，就是 V10 的小格特徵。"""

    def __init__(self):
        super().__init__()
        r = resnet18(weights=WEIGHTS).eval()
        self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.l2, self.l3 = r.layer2, r.layer3

    def forward(self, x):
        a = self.l2(self.stem(x))
        b = torch.nn.functional.interpolate(self.l3(a), size=a.shape[-2:], mode="bilinear")
        return torch.cat([a, b], 1)


def count(m):
    return sum(p.numel() for p in m.parameters())


def load_image(path):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return WEIGHTS.transforms()(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)


print("== 1. 把「ResNet18 算到 layer3」包成一個 nn.Module ==")
full = resnet18(weights=WEIGHTS).eval()
print("ResNet18 的零件：", [name for name, _ in full.named_children()])
model = Patch().eval()
print(f"參數：整個 ResNet18 {count(full) / 1e6:.2f} M → 只到 layer3 {count(model) / 1e6:.2f} M")
x = load_image(IMAGE)
with torch.no_grad():
    ref = model(x).numpy()
print(f"輸入 {tuple(x.shape)} → 輸出 {ref.shape}（128＋256＝384 個特徵 × 28×28 小格）")

print("\n== 2. torch.onnx.export：匯出成 ONNX ==")
path = OUT / "patch.onnx"
torch.onnx.export(model, x, str(path), input_names=["image"], output_names=["features"],
                  dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}}, opset_version=17, dynamo=False)
print(f"{path.name}：{path.stat().st_size / 1e6:.1f} MB")

print("\n== 3. 打開 ONNX 檔看看：輸入、輸出、opset ==")
m = onnx.load(str(path))
onnx.checker.check_model(m)
print("onnx.checker.check_model：通過")


def dims(v):
    return [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]


for kind, values in [("輸入", m.graph.input), ("輸出", m.graph.output)]:
    for v in values:
        print(f"{kind} {v.name}：{dims(v)}")
print(f"opset {m.opset_import[0].version}；運算節點 {len(m.graph.node)} 個，種類：{sorted({n.op_type for n in m.graph.node})}")

print("\n== 4. 驗收：同一張圖，ONNX 和 PyTorch 算出來一樣嗎 ==")
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
got = sess.run(None, {"image": x.numpy()})[0]
diff = np.abs(got - ref).max()
rel = diff / np.abs(ref).max()
print(f"特徵最大值 {np.abs(ref).max():.2f}；最大差 {diff:.1e}；相對誤差 {rel:.1e}")
print(f"相對誤差 < 1e-4：{rel < 1e-4}")
print(f"np.allclose(rtol=1e-3, atol=1e-4)：{np.allclose(got, ref, rtol=1e-3, atol=1e-4)}")

print("\n== 5. 沒寫 dynamic_axes：batch 被寫死成 1 ==")
fixed = OUT / "patch-fixed.onnx"
torch.onnx.export(model, x, str(fixed), input_names=["image"], output_names=["features"],
                  opset_version=17, dynamo=False)
print("輸入 image：", dims(onnx.load(str(fixed)).graph.input[0]))
three = np.repeat(x.numpy(), 3, axis=0)
print(f"有 dynamic_axes，一次 3 張 → {sess.run(None, {'image': three})[0].shape}")
try:
    ort.InferenceSession(str(fixed), providers=["CPUExecutionProvider"]).run(None, {"image": three})
except Exception as e:   # onnxruntime 的錯誤類別在 C++ 那一層定義
    print(f"沒有 dynamic_axes，一次 3 張 → {type(e).__name__}")

print("\n== 6. ultralytics：YOLO.export(format='onnx') 一行匯出 ==")
from ultralytics import YOLO  # noqa: E402

shutil.copy(ROOT / "models" / "yolo11n.pt", OUT / "yolo11n.pt")   # 匯出檔會寫在 .pt 旁邊，先複製一份到 out/
onnx_path = YOLO(str(OUT / "yolo11n.pt")).export(format="onnx", imgsz=320, simplify=False)
y = onnx.load(onnx_path)
print(f"{Path(onnx_path).name}：{Path(onnx_path).stat().st_size / 1e6:.1f} MB；opset {y.opset_import[0].version}")
print(f"輸入 {dims(y.graph.input[0])} → 輸出 {dims(y.graph.output[0])}")

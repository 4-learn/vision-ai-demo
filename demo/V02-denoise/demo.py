"""V02｜OpenCV：去噪與銳化。GaussianBlur、medianBlur、fastNlMeansDenoising 去噪，PSNR 打分數，addWeighted 做 unsharp mask。

執行（在 repo 根目錄）：python demo/V02-denoise/demo.py
工廠線的螺帽照片要先由教師執行 tools/build_data.py 產生（MVTec AD 衍生，不隨 repo 散佈）。
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "story" / "data"


def read_gray(rel):
    img = cv2.imread(str(DATA / rel), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(rel)
    return img


def contrast(gray, mask):
    inside, outside = gray[mask > 0], gray[mask == 0]
    return abs(float(inside.mean()) - float(outside.mean()))


nut = read_gray("factory/anomaly/test/scratch/000.png")
scratch = read_gray("factory/anomaly/ground_truth/scratch/000_mask.png")
room = read_gray("care/anomaly/train/good/000.png")


def gaussian_noise(img, sigma):
    """每個像素加上常態分布的亂數：像夜拍、感光元件的顆粒。"""
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def salt_pepper(img, frac):
    """隨機挑 frac 的像素變成全黑或全白：像傳輸壞掉、灰塵反光。"""
    out = img.copy()
    pick = rng.random(img.shape)
    out[pick < frac / 2] = 0
    out[pick > 1 - frac / 2] = 255
    return out


print("== 1. cv2.PSNR：和原圖有多像（dB，越大越像）==")
print(f"原圖 vs 原圖：{cv2.PSNR(nut, nut):.1f} dB（完全一樣）")
rng = np.random.default_rng(0)
noisy = np.clip(nut.astype(np.float32) + rng.normal(0, 15, nut.shape), 0, 255).astype(np.uint8)
psnr = cv2.PSNR(nut, noisy)
print(f"原圖 vs 加雜訊 σ=15：{psnr:.2f} dB")
print(f"原圖 vs 加雜訊 σ=30：{cv2.PSNR(nut, gaussian_noise(nut, 30)):.2f} dB")
print(f"原圖 vs 整張亮 10：{cv2.PSNR(nut, cv2.add(nut, 10)):.2f} dB")

print("\n== 2. 兩種雜訊、三種去噪（PSNR dB）==")
cases = {"螺帽＋顆粒": (nut, gaussian_noise(nut, 15)), "螺帽＋黑白點": (nut, salt_pepper(nut, 0.05)),
         "房間＋顆粒": (room, gaussian_noise(room, 15)), "房間＋黑白點": (room, salt_pepper(room, 0.05))}
print(f"{'':<18}" + "".join(f"{c:>9}" for c in cases))
print(f"{'不處理':<18}" + "".join(f"{cv2.PSNR(clean, dirty):11.2f}" for clean, dirty in cases.values()))
filters = {"GaussianBlur 5×5": lambda g: cv2.GaussianBlur(g, (5, 5), 0),
           "medianBlur 5": lambda g: cv2.medianBlur(g, 5),
           "fastNlMeans h=10": lambda g: cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)}
for name, f in filters.items():
    print(f"{name:<18}" + "".join(f"{cv2.PSNR(clean, f(dirty)):11.2f}" for clean, dirty in cases.values()))

print("\n== 3. fastNlMeansDenoising 的 h：去多少雜訊 ==")
for h in [3, 10, 20, 30]:
    out_nut = cv2.fastNlMeansDenoising(cases["螺帽＋顆粒"][1], None, h=h, templateWindowSize=7, searchWindowSize=21)
    out_room = cv2.fastNlMeansDenoising(cases["房間＋顆粒"][1], None, h=h, templateWindowSize=7, searchWindowSize=21)
    print(f"h={h:<3}螺帽 {cv2.PSNR(nut, out_nut):.2f} dB；房間 {cv2.PSNR(room, out_room):.2f} dB；"
          f"螺帽刮痕亮度差 {contrast(out_nut, scratch):.1f}")
print(f"（原圖刮痕亮度差 {contrast(nut, scratch):.1f}）")

print("\n== 4. 去噪的代價：模糊越重，刮痕越淡 ==")
for k in [3, 5, 9, 15]:
    blur = cv2.GaussianBlur(nut, (k, k), 0)
    print(f"GaussianBlur {k:>2}×{k:<2} 刮痕亮度差 {contrast(blur, scratch):.1f}")

print("\n== 5. cv2.addWeighted：unsharp mask 銳化 ==")
background = nut < 40
blur = cv2.GaussianBlur(nut, (0, 0), 2.0)
for amount in [0.0, 0.5, 1.0, 2.0]:
    sharp = cv2.addWeighted(nut, 1 + amount, blur, -amount, 0)
    print(f"amount={amount:<4}邊緣強度 {cv2.Laplacian(sharp, cv2.CV_64F).std():5.1f}；"
          f"刮痕亮度差 {contrast(sharp, scratch):5.1f}；背景起伏 {sharp[background].std():4.1f}")
sharp_noisy = cv2.addWeighted(noisy, 2.0, cv2.GaussianBlur(noisy, (0, 0), 2.0), -1.0, 0)
print(f"有雜訊的圖直接銳化：PSNR {cv2.PSNR(nut, noisy):.2f} → {cv2.PSNR(nut, sharp_noisy):.2f} dB（雜訊也被放大）")

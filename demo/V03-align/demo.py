"""V03｜OpenCV：把零件轉正。moments 求重心與主軸、getRotationMatrix2D＋warpAffine 旋轉，主軸不可靠時逐角度找最像。

執行（在 repo 根目錄）：python demo/V03-align/demo.py
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


def rotate(img, angle):
    """以圖的中心逆時針轉 angle 度，大小不變；轉出去的角落用邊緣的顏色補。"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def axis_angle(gray):
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.moments(fg, binaryImage=True)
    angle = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    return angle


def pixel_diff(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


print("== 1. cv2.moments：一根長條的重心和主軸 ==")
bar = np.zeros((200, 200), np.uint8)
cv2.rectangle(bar, (40, 90), (160, 110), 255, -1)       # 橫的長條：寬 121、高 21
m = cv2.moments(bar, binaryImage=True)
print(f"m00（面積）{m['m00']:.0f}；重心 x={m['m10'] / m['m00']:.1f}、y={m['m01'] / m['m00']:.1f}")
print(f"mu20 {m['mu20']:.0f}、mu02 {m['mu02']:.0f}、mu11 {m['mu11']:.0f}")
for a in [0, 30, 60, -45]:
    print(f"把長條轉 {a:>3}° → 主軸角度 {axis_angle(rotate(bar, a)):6.1f}°")

print("\n== 2. getRotationMatrix2D：2×3 的轉換矩陣 ==")
M = cv2.getRotationMatrix2D((100, 100), 30, 1.0)
print(np.round(M, 3))
x, y = 160, 100
print(f"點 ({x}, {y}) 轉 30° 之後 → ({M[0, 0] * x + M[0, 1] * y + M[0, 2]:.1f}, {M[1, 0] * x + M[1, 1] * y + M[1, 2]:.1f})")

print("\n== 3. warpAffine：角落補什麼 ==")
nut = read_gray("factory/anomaly/train/good/000.png")
for name, mode in [("BORDER_CONSTANT（補黑）", cv2.BORDER_CONSTANT), ("BORDER_REPLICATE（補邊緣）", cv2.BORDER_REPLICATE)]:
    h, w = nut.shape
    out = cv2.warpAffine(nut, cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0), (w, h), borderMode=mode)
    print(f"{name}：左上角 {out[0, 0]}、shape {out.shape}")
print(f"原圖左上角 {nut[0, 0]}")

print("\n== 4. 螺帽：主軸角度跟不上真的旋轉 ==")
print(f"螺帽 000.png 的主軸角度 {axis_angle(nut):.1f}°")
for a in [10, 20, 45, 90]:
    expect = (axis_angle(nut) - a + 90) % 180 - 90           # 主軸只有 -90～90 度
    print(f"把它轉 {a:>2}° → 主軸角度 {axis_angle(rotate(nut, a)):6.1f}°（主軸可靠的話應該是 {expect:6.1f}°）")
_, fg = cv2.threshold(nut, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mm = cv2.moments(fg, binaryImage=True)
print(f"mu20 {mm['mu20']:.3g}、mu02 {mm['mu02']:.3g}：幾乎一樣大（四爪對稱，沒有「長的方向」）")

print("\n== 5. 逐角度找最像：每隔 3 度轉一次，和參考圖比 ==")
moved = rotate(nut, 37)                                    # 假裝這是一顆歪了 37° 的螺帽
small, ref = cv2.resize(moved, (64, 64)).astype(np.float32), cv2.resize(nut, (64, 64)).astype(np.float32)
diffs = [(float(np.abs(rotate(small, a) - ref).mean()), a) for a in range(0, 360, 3)]
best = min(diffs)[1]
for d, a in sorted(diffs)[:4]:
    print(f"轉 {a:>3}°：平均像素差 {d:.1f}")
print(f"最像的角度 {best}°；轉回去後和原圖的像素差 {pixel_diff(moved, nut):.1f} → {pixel_diff(rotate(moved, best), nut):.1f}")
score = {a: d for d, a in diffs}
print("每隔 90° 都有一個很像的：", "、".join(f"{a}° {score[a]:.1f}" for a in (54, 144, 234, 324)))
print("對照不轉和轉 90°、180°：", "、".join(f"{a}° {score[a]:.1f}" for a in (0, 90, 180)))

print("\n== 6. 健康線：鏡頭固定，不用轉 ==")
rooms = [read_gray(f"care/anomaly/train/good/{i:03d}.png") for i in range(5)]
ref = cv2.resize(rooms[0], (64, 64)).astype(np.float32)
for i, r in enumerate(rooms[1:], start=1):
    s = cv2.resize(r, (64, 64)).astype(np.float32)
    best = min((float(np.abs(rotate(s, a) - ref).mean()), a) for a in range(0, 360, 3))[1]
    print(f"{i:03d}.png（平均亮度 {r.mean():5.1f}）和 000.png：不轉像素差 {pixel_diff(r, rooms[0]):5.1f}；"
          f"逐角度找到 {best:>3}°，轉了之後 {pixel_diff(rotate(r, best), rooms[0]):5.1f}")

"""產生影像 AI 課（2026 v2）兩條線的資料。

    python tools/build_data.py            # 重建（工廠線需要 MVTEC_ROOT）
    python tools/build_data.py --check    # 比對 MANIFEST（只比健康線與骨架；工廠線是 MVTec 衍生，不進 repo）

兩條線同一種格式，程式只有一份：
  story/data/{LINE}/anomaly/train/good/*.png                 只有正常（良品／正常畫面）
  story/data/{LINE}/anomaly/test/{good,<瑕疵類型>}/*.png
  story/data/{LINE}/anomaly/ground_truth/<瑕疵類型>/*_mask.png  瑕疵在哪裡（白色）
  story/data/{LINE}/pose/poses.csv                             骨架序列（沿用 ML 課 25–42 節）

工廠線：MVTec AD 的 metal_nut（CC BY-NC-SA 4.0，教師預先下載到 MVTEC_ROOT；只在課堂使用、不隨 repo 散佈），縮成 256×256。
健康線：程式畫出的寢室畫面（天花板攝影機）；「異常」＝有人倒在地上（fallen）、走道上有東西（obstacle）、鏡頭被遮（blocked）。
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260201
SIZE = 256
MVTEC_ROOT = Path(os.environ.get("MVTEC_ROOT", Path.home() / "datasets/mvtec-ad"))
ML_STORY = Path(os.environ.get("ML_STORY", Path.home() / "workspace/4-learn/python-machine-learning-demo/story/data"))


# ---------------------------------------------------------------- 工廠線：MVTec metal_nut
def build_factory(out: Path) -> dict:
    src = MVTEC_ROOT / "images"
    if not (src / "train" / "metal_nut").exists():
        print("找不到 MVTec（設定 MVTEC_ROOT）：略過工廠線影像")
        return {}
    a = out / "anomaly"
    stats = {}
    for split in ["train", "test"]:
        for d in sorted((src / split / "metal_nut").iterdir()):
            dst = a / split / d.name
            dst.mkdir(parents=True, exist_ok=True)
            for p in sorted(d.glob("*.png")):
                Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS).save(dst / p.name)
            stats[f"{split}/{d.name}"] = len(list(dst.glob("*.png")))
    for d in sorted((MVTEC_ROOT / "masks" / "test" / "metal_nut").iterdir()):
        dst = a / "ground_truth" / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(d.glob("*.png")):
            Image.open(p).convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST).save(dst / p.name)
    return stats


# ---------------------------------------------------------------- 健康線：寢室畫面
def room(rng, night):
    """一張正常的寢室畫面（天花板往下看）。回傳 (img, draw, 地板走道區域)。"""
    floor = np.array([196, 182, 160]) + rng.normal(0, 4, 3)
    img = Image.new("RGB", (SIZE, SIZE), tuple(int(v) for v in floor))
    d = ImageDraw.Draw(img)
    for y in range(0, SIZE, 32):                                   # 地板接縫
        d.line([0, y, SIZE, y], fill=tuple(int(v * 0.93) for v in floor), width=1)
    dx, dy = rng.integers(-3, 4, 2)                                # 攝影機微晃
    d.rectangle([150 + dx, 20 + dy, 236 + dx, 150 + dy], fill=(120, 95, 70))         # 床框
    d.rectangle([156 + dx, 26 + dy, 230 + dx, 144 + dy], fill=(236, 236, 240))       # 床單
    d.rectangle([160 + dx, 30 + dy, 226 + dx, 52 + dy], fill=(250, 250, 252))        # 枕頭
    blanket = [(180, 190, 220), (200, 180, 190), (175, 200, 180)][rng.integers(3)]
    d.rectangle([158 + dx, 70 + dy, 228 + dx, 140 + dy], fill=blanket)              # 被子
    if rng.random() < 0.5:                                          # 床上躺著人（正常）
        d.ellipse([182 + dx, 34 + dy, 204 + dx, 56 + dy], fill=(220, 180, 150))
    d.rectangle([110 + dx, 22 + dy, 140 + dx, 52 + dy], fill=(150, 115, 80))         # 床頭櫃
    d.rectangle([20 + dx, 190 + dy, 70 + dx, 236 + dy], fill=(140, 110, 80))         # 衣櫃
    cx = 40 + int(rng.integers(-6, 7))                              # 椅子（位置每天不同）
    d.rectangle([cx + dx, 30 + dy, cx + 34 + dx, 62 + dy], fill=(110, 80, 60))
    if rng.random() < 0.4:                                          # 地墊
        d.rectangle([150 + dx, 160 + dy, 236 + dx, 186 + dy], fill=(150, 120, 120))
    if rng.random() < 0.3:                                          # 拖鞋（正常的小東西）
        d.ellipse([140 + dx, 150 + dy, 150 + dx, 158 + dy], fill=(90, 90, 140))
    return img, d, (60, 70, 145, 240)


def finish(img, rng, night):
    a = np.asarray(img, np.float32)
    a *= (0.42 if night else rng.uniform(0.92, 1.06))
    a += rng.normal(0, 7 if night else 2.5, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def anomaly(img, d, kind, rng, walk):
    """在畫面上加一個異常，回傳遮罩。"""
    m = Image.new("L", (SIZE, SIZE), 0)
    md = ImageDraw.Draw(m)
    x0, y0, x1, y1 = walk
    if kind == "fallen":                                            # 人倒在地上
        x, y = int(rng.integers(x0, x1 - 78)), int(rng.integers(y0 + 20, y1 - 30))
        cloth = [(150, 120, 160), (120, 140, 170), (170, 120, 110)][rng.integers(3)]
        body = [x + 14, y, x + 62, y + 16]
        d.rectangle(body, fill=cloth); md.rectangle(body, fill=255)
        head = [x, y - 1, x + 16, y + 17]
        d.ellipse(head, fill=(220, 180, 150)); md.ellipse(head, fill=255)
        leg = [x + 62, y + 2, x + 80, y + 14]
        d.rectangle(leg, fill=(70, 70, 80)); md.rectangle(leg, fill=255)
    elif kind == "obstacle":                                        # 走道上的東西（絆倒風險）
        x, y = int(rng.integers(x0, x1 - 30)), int(rng.integers(y0 + 30, y1 - 30))
        w, h = int(rng.integers(16, 30)), int(rng.integers(12, 24))
        col = [(170, 120, 60), (60, 60, 60), (200, 60, 60)][rng.integers(3)]
        d.rectangle([x, y, x + w, y + h], fill=col); md.rectangle([x, y, x + w, y + h], fill=255)
    elif kind == "blocked":                                         # 鏡頭被遮住一部分
        x, y = int(rng.integers(0, SIZE - 90)), int(rng.integers(0, SIZE - 90))
        r = int(rng.integers(40, 70))
        d.ellipse([x, y, x + r * 1.4, y + r], fill=(40, 35, 30)); md.ellipse([x, y, x + r * 1.4, y + r], fill=255)
    return m


def build_care(out: Path, rng) -> dict:
    a = out / "anomaly"
    stats = {}
    plan = [("train", "good", 200), ("test", "good", 30), ("test", "fallen", 25), ("test", "obstacle", 25), ("test", "blocked", 15)]
    for split, kind, n in plan:
        dst = a / split / kind
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            night = rng.random() < 0.25
            img, d, walk = room(rng, night)
            if kind != "good":
                m = anomaly(img, d, kind, rng, walk)
                (a / "ground_truth" / kind).mkdir(parents=True, exist_ok=True)
                m.save(a / "ground_truth" / kind / f"{i:03d}_mask.png")
            img = finish(img, rng, night)
            if rng.random() < 0.2:
                img = img.filter(ImageFilter.GaussianBlur(0.6))
            img.save(dst / f"{i:03d}.png")
        stats[f"{split}/{kind}"] = n
    return stats


def copy_pose(out: Path, line: str) -> dict:
    src = ML_STORY / line / "pose"
    dst = out / "pose"; dst.mkdir(parents=True, exist_ok=True)
    for name in ["poses.csv", "poses-site2.csv"]:
        shutil.copy(src / name, dst / name)
    return {"pose": "copied from ML course"}


def build(base: Path) -> dict:
    stats = {}
    for line in ["factory", "care"]:
        out = base / line
        shutil.rmtree(out / "anomaly", ignore_errors=True)
        s = build_factory(out) if line == "factory" else build_care(out, np.random.default_rng(SEED))
        s.update(copy_pose(out, line))
        stats[line] = s
    return stats


def manifest(base: Path) -> dict:
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob("*")) if p.is_file() and not str(p.relative_to(base)).startswith("factory/anomaly")
            and p.name != "MANIFEST.json"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    target = ROOT / "story" / "data"
    if a.check:
        with tempfile.TemporaryDirectory() as t:
            build(Path(t)); got = manifest(Path(t))
        want = json.loads((target / "MANIFEST.json").read_text())["sha256"]
        bad = [k for k in want if got.get(k) != want[k]]
        print("OK：與 MANIFEST 相同" if not bad else f"不同：{len(bad)} 個，例如 {bad[:3]}"); sys.exit(1 if bad else 0)
    stats = build(target)
    (target / "MANIFEST.json").write_text(json.dumps({"generator": "tools/build_data.py", "seed": SEED, "stats": stats,
        "caveat": "健康線為程式畫出的合成畫面；工廠線為 MVTec AD metal_nut 衍生（CC BY-NC-SA 4.0，不隨 repo 散佈）。",
        "sha256": manifest(target)}, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

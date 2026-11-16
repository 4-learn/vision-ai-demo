"""V35｜sha256sum：交接包。SHA256SUMS 記下每個檔的指紋，接手的人用 sha256sum -c 一次檢查。

執行（在 repo 根目錄）：python demo/V35-handoff/demo.py
會在 demo/V35-handoff/out/ 寫出一個小交接包。需要 sha256sum（Linux、Pi 5、Git Bash 都有）。
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out" / "handoff"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)

print("== 1. 交接包放什麼、不放什麼 ==")
files = {
    "config.json": json.dumps({"line": "care", "event_type": "room_anomaly", "threshold": 3.4171}, indent=1),
    "scores.csv": "image,score,pred,y\nfallen/000.png,4.812,1,1\ngood/000.png,2.903,0,0\nobstacle/003.png,3.210,0,1\n",
    "README.md": "# 交接包\n健康線為合成畫面。工廠線影像為 MVTec AD 衍生（CC BY-NC-SA 4.0），不放進交接包。\n",
}
for name, text in files.items():
    (OUT / name).write_text(text, encoding="utf-8")
skipped = ["fallen/000.png", "good/000.png", "obstacle/003.png"]
print("放進去：", sorted(files))
print("不放：  影像", len(skipped), "張（scores.csv 只記檔名，影像由教師用 tools/build_data.py 重建）")

print("\n== 2. 寫 SHA256SUMS：一行「64 碼指紋＋兩個空白＋檔名」==")


def sha256_file(path, chunk=1 << 16):
    """大檔案一段一段讀，不用整個讀進記憶體。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


lines = [f"{sha256_file(p)}  {p.name}\n" for p in sorted(OUT.iterdir())]
(OUT / "SHA256SUMS").write_text("".join(lines))
for line in lines:
    print(line[:12] + "…" + line[64:].rstrip())
print("和 hashlib.sha256(整個檔).hexdigest() 相同：",
      all(sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()))


def verify():
    r = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=OUT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()


print("\n== 3. 接手的人：sha256sum -c SHA256SUMS ==")
code, out = verify()
print("\n".join(out))
print("exit code", code)

print("\n== 4. 有人改了一個數字 ==")
cfg = OUT / "config.json"
cfg.write_text(cfg.read_text().replace("3.4171", "3.2"), encoding="utf-8")
code, out = verify()
print("\n".join(out))
print("exit code", code)

print("\n== 5. 少了一個檔 ==")
cfg.write_text(files["config.json"], encoding="utf-8")
(OUT / "scores.csv").unlink()
r = subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=OUT, capture_output=True, text=True)
print("沒問題的不印（--quiet），有問題的：")
print(r.stdout.strip())
print("exit code", r.returncode)

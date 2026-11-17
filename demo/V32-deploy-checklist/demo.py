"""V32｜上線清單：logging 寫結構化紀錄（一行一個 JSON）、hashlib 指紋、啟動自檢。

執行（在 repo 根目錄）：python demo/V32-deploy-checklist/demo.py
會在 demo/V32-deploy-checklist/out/ 寫出兩個小檔案。紀錄同時寫到 stderr（含時間，每次不同）。
"""
import hashlib
import io
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

print("== 1. print 和 logging 的差別：logging 預設寫到 stderr ==")
plain = logging.getLogger("plain")
print(f"StreamHandler 預設的去處：{'stderr' if logging.StreamHandler().stream is sys.stderr else 'stdout'}")
print(f"沒設定時的等級：{logging.getLevelName(plain.getEffectiveLevel())}（INFO 不會出現，WARNING 以上才會）")

print("\n== 2. 一行一個 JSON：程式讀得回來、可以篩 ==")


def json_logger(name, *streams):
    """每一筆紀錄排成一行 JSON：時間、等級，msg 放呼叫的人用 json.dumps 寫好的內容。"""
    log = logging.getLogger(name)
    log.handlers.clear()
    for s in streams:
        h = logging.StreamHandler(s)
        h.setFormatter(logging.Formatter('{"ts": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}',
                                         "%Y-%m-%dT%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


buffer = io.StringIO()                       # 示範用：同時寫一份到記憶體，才能讀回來
log = json_logger("vision-care", sys.stderr, buffer)
log.info(json.dumps({"event": "startup", "line": "care", "model": "1361ce3e501a"}))
log.info(json.dumps({"event": "sent", "image": "fallen/000.png", "score": 4.12, "http": 200}))
log.warning(json.dumps({"event": "send_failed", "image": "obstacle/003.png", "error": "ReadTimeout"}))
log.info(json.dumps({"event": "sent", "image": "blocked/001.png", "score": 5.30, "http": 200}))
records = [json.loads(line) for line in buffer.getvalue().splitlines()]
print(f"寫了 {len(records)} 行，每行都有 ts、level、msg：{all(set(r) == {'ts', 'level', 'msg'} for r in records)}")
for r in records:
    print(r["level"], r["msg"])
print("只看 WARNING：", [r["msg"]["image"] for r in records if r["level"] == "WARNING"])
print("送出成功幾筆：", sum(r["msg"].get("http") == 200 for r in records))

print("\n== 2b. 忘了 json.dumps，直接寫一句話 ==")
buffer.seek(0)
buffer.truncate()
log.info("送出 fallen/000.png")
try:
    json.loads(buffer.getvalue())
except json.JSONDecodeError as e:
    print(f"這一行讀不回來 → JSONDecodeError：{e.msg}")

print("\n== 3. hashlib：模型檔的指紋 ==")
model = OUT / "model.bin"
model.write_bytes(bytes(range(256)) * 40)
manifest = {"files": {"model.bin": hashlib.sha256(model.read_bytes()).hexdigest()[:12]}}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(manifest))


# 啟動自檢：任何一項不過就 raise，不准開始送事件。
def startup_check(manifest, profile, event_type):
    for name, expected in manifest["files"].items():
        path = OUT / name
        if not path.exists():
            raise RuntimeError(f"{name} 不存在，拒絕啟動")
        if hashlib.sha256(path.read_bytes()).hexdigest()[:12] != expected:
            raise RuntimeError(f"{name} 的指紋對不上，拒絕啟動")
    if event_type not in profile["event_types"]:
        raise RuntimeError(f"設定檔沒有 {event_type}，平台會回 400，拒絕啟動")
    return True


print("\n== 4. 啟動自檢：三種會被擋下來的情況 ==")
profile = json.loads((ROOT / "story" / "profiles" / "care.json").read_text(encoding="utf-8"))
profile["event_types"]["room_anomaly"] = {"severity": "high", "escalate_after_minutes": 2}
print("全部正常 →", startup_check(manifest, profile, "room_anomaly"))
cases = {
    "模型檔被換過": lambda: model.write_bytes(model.read_bytes()[:-1] + b"\0"),
    "模型檔不見": lambda: model.unlink(),
}
for name, damage in cases.items():
    model.write_bytes(bytes(range(256)) * 40)
    damage()
    try:
        startup_check(manifest, profile, "room_anomaly")
    except RuntimeError as e:
        print(f"{name} → {e}")
model.write_bytes(bytes(range(256)) * 40)
try:
    startup_check(manifest, json.loads((ROOT / "story" / "profiles" / "care.json").read_text(encoding="utf-8")), "room_anomaly")
except RuntimeError as e:
    print(f"設定檔是舊的 → {e}")

"""V34｜dataclasses：Demo 與互審清單。每一項寫清楚「怎麼查」，能自動查的讓程式查，不能的留給人。

執行（在 repo 根目錄）：python demo/V34-review/demo.py
兩份手打的「同學交來的專案摘要」，不需要影像。
"""
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional


@dataclass
class Item:
    name: str
    how: str                                         # 互審的人要怎麼查
    check: Optional[Callable[[dict], bool]] = None   # 能自動查的放函式；None ＝ 要人看
    tags: list = field(default_factory=list)


CHECKLIST = [
    Item("測試集只打開一次", "看紀錄：test 被讀了幾次", lambda p: p["test_opened"] == 1, ["資料"]),
    Item("門檻沒有用測試集定", "看門檻是從哪份資料算出來的", lambda p: p["threshold_from"] == "calib", ["資料"]),
    Item("每一類都有成績", "成績表的類型數 ＝ 資料夾的類型數", lambda p: set(p["report_types"]) == set(p["types"]), ["成績"]),
    Item("漏掉的有逐筆看過", "漏掉的每一張都有一句說明", lambda p: len(p["miss_notes"]) == p["missed"], ["成績"]),
    Item("模型有指紋", "設定檔裡有 sha256，啟動會比對", lambda p: bool(p.get("sha256_12")), ["上線"]),
    Item("資料的限制有寫", "README 有寫合成畫面／MVTec 授權", None, ["文件"]),
    Item("上線後怎麼監控有寫", "誰看真告警率、多久看一次", None, ["文件"]),
]

team_a = {"test_opened": 1, "threshold_from": "calib", "types": ["bent", "color", "flip", "scratch"],
         "report_types": ["bent", "color", "flip", "scratch"], "missed": 3,
         "miss_notes": ["color 很淡", "scratch 太細", "scratch 在邊緣"], "sha256_12": "48eacc113113"}
team_b = {"test_opened": 4, "threshold_from": "test", "types": ["fallen", "obstacle", "blocked"],
       "report_types": ["fallen", "blocked"], "missed": 5, "miss_notes": [], "sha256_12": ""}

print("== 1. @dataclass：一項就是一筆有欄位的資料 ==")
print(CHECKLIST[0].name, "｜", CHECKLIST[0].how, "｜", CHECKLIST[0].tags)
print({k: v for k, v in asdict(CHECKLIST[5]).items() if k != "check"})


def review(project):
    """自動的項目直接算；人工的項目標成「待人看」。回傳 (勾選結果, 待人看的清單)。"""
    rows, manual = [], []
    for item in CHECKLIST:
        if item.check is None:
            manual.append(item)
            rows.append(("?", item.name))
        else:
            rows.append(("x" if item.check(project) else " ", item.name))
    return rows, manual


for who, project in [("A 組", team_a), ("B 組", team_b)]:
    print(f"\n== 2. 互審 {who}的專案 ==")
    rows, manual = review(project)
    for mark, name in rows:
        print(f"[{mark}] {name}")
    failed = [n for m, n in rows if m == " "]
    print(f"自動檢查沒過 {len(failed)} 項；要人看 {len(manual)} 項：{[m.how for m in manual]}")

print("\n== 3. 永遠不會失敗的檢查，不算檢查 ==")
lazy = [("漏掉的有逐筆看過", lambda p: p["missed"] >= 0), ("模型有指紋", lambda p: True)]
for name, fn in lazy:
    print(f"{name}：A 組 {fn(team_a)}、B 組 {fn(team_b)}")
print("B 組一張漏掉的都沒看、也沒有指紋，照樣打勾")

print("\n== 4. 依 tags 分組：Demo 時照這個順序講 ==")
order = ["資料", "成績", "上線", "文件"]
for tag in order:
    print(f"{tag}：{[i.name for i in CHECKLIST if tag in i.tags]}")

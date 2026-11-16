"""把 master_vision.py 切成 v01–v27 各版的 vision.py。"""
import re, sys
from pathlib import Path
src = Path(sys.argv[1]).read_text().splitlines()
root = Path(sys.argv[2])
names = {1: "v01-clahe", 2: "v02-denoise", 3: "v03-align", 4: "v04-edges", 5: "v05-hough", 6: "v06-rule-baseline", 7: "v07-augment", 8: "v08-pixel-baseline", 9: "v09-global-knn", 10: "v10-patch-knn", 11: "v11-heatmap", 12: "v12-threshold", 13: "v13-report", 14: "v14-yolo-cls", 15: "v15-mask-to-box", 16: "v16-yolo-train", 17: "v17-yolo-eval", 18: "v18-compare", 19: "v19-stream", 20: "v20-tracking", 21: "v21-state-machine", 22: "v22-debounce", 23: "v23-onnx-export", 24: "v24-onnx-runtime", 25: "v25-quantize", 26: "v26-budget", 27: "v27-new-event", 28: "v28-mvp-anomaly", 29: "v29-mvp-pose", 30: "v30-human-in-loop", 31: "v31-mock", 32: "v32-deploy-checklist", 33: "v33-acceptance", 34: "v34-review", 35: "v35-handoff"}
for v, name in names.items():
    keep, mode = [], ("all", None)
    for line in src:
        m = re.match(r"#@(all|v(\d\d)|main((?: v\d\d)+))$", line.strip())
        if m:
            if m.group(1) == "all": mode = ("all", None)
            elif m.group(2): mode = ("ver", int(m.group(2)))
            else: mode = ("main", [int(x[1:]) for x in m.group(3).split()])
            continue
        if mode[0] == "all" or (mode[0] == "ver" and v >= mode[1]) or (mode[0] == "main" and v in mode[1]):
            keep.append(line)
    text = "\n".join(keep).replace("{VERSION}", name.split("-")[0]) + "\n"
    text = re.sub(r"\n{3,}(?=def |class |# |[A-Z_]+ = )", "\n\n\n", text)
    d = root / name; d.mkdir(parents=True, exist_ok=True); (d / "vision.py").write_text(text)
print("ok")

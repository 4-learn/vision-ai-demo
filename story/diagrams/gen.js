// 架構圖產生器（影像 AI 淬鍊實戰專案 2026 v2）。繪圖引擎沿用 fastapi-workshop 與 ML 課的 gen.js。
// since "00"＝之前就有（告警平台、ML 課的判斷器與偵測器），永遠灰色。
// 用法：node gen.js all

const VERS = ["00", ...Array.from({ length: 35 }, (_, i) => String(i + 1).padStart(2, "0"))];
const TITLES = {
  "01": ["打光不均", "OpenCV CLAHE"], "02": ["去噪與銳化", "OpenCV fastNlMeansDenoising"],
  "03": ["把零件轉正", "OpenCV warpAffine"], "04": ["找邊緣", "OpenCV Canny"],
  "05": ["霍夫與量測", "OpenCV HoughCircles"], "06": ["規則 baseline", "OpenCV 量測＋規則"],
  "07": ["資料增強", "torchvision transforms.v2"], "08": ["像素差 baseline", "NumPy＋roc_auc_score"],
  "09": ["預訓練特徵與最近鄰", "feature_extraction＋NearestNeighbors"], "10": ["切成小格比", "PatchCore 簡化版"],
  "11": ["熱圖：異常在哪裡", "OpenCV applyColorMap"], "12": ["門檻", "roc_curve"],
  "13": ["成績表與錯誤分析", "pandas"], "14": ["YOLO 分類模式", "ultralytics classify"],
  "15": ["遮罩變成框", "connectedComponentsWithStats"], "16": ["訓練 YOLO 偵測", "ultralytics train"],
  "17": ["看懂 val() 與 mAP", "ultralytics val"], "18": ["有標註 vs 只有正常", "比較"],
  "19": ["骨架串流", "generator＋deque"], "20": ["追蹤", "linear_sum_assignment"],
  "21": ["狀態機", "dict 轉移表"], "22": ["去抖動與冷卻", "pandas shift"],
  "23": ["匯出 ONNX", "torch.onnx.export"], "24": ["onnxruntime 與延遲", "InferenceSession"],
  "25": ["量化", "onnxruntime.quantization"], "26": ["Pi 5 的預算", "解析度與延遲"],
  "27": ["新事件類型", "requests＋設定檔"], "28": ["MVP：異常偵測接平台", "Workshop"],
  "29": ["MVP：骨架串流接平台", "Workshop"], "30": ["人在迴路", "真告警率"],
  "31": ["沒有攝影機也能測", "Mock"], "32": ["上線清單", "logging＋hashlib"],
  "33": ["驗收報告", "先寫標準"], "34": ["Demo 與互審", "清單"], "35": ["交接包", "SHA256SUMS"],
  full: ["全貌：做完這門課會長這樣", "接告警平台與 ML 課的判斷器、偵測器"],
};
let FULL = false;
const vi = (v) => VERS.indexOf(v);
const W = 1340, H = 700;
const SANS = "'PingFang TC','Noto Sans TC','Noto Sans CJK TC','Microsoft JhengHei',sans-serif";
const MONO = "'SF Mono',Menlo,Consolas,'DejaVu Sans Mono',monospace";
const PALETTE = {
  old: { fill: "#f4f5f7", stroke: "#a3a8b0", text: "#3a3f47", line: "#8a9099" },
  new: { fill: "#e3f7e8", stroke: "#1f9d4c", text: "#0f5132", line: "#1f9d4c" },
  changed: { fill: "#fff4dc", stroke: "#d9901a", text: "#7a4b00", line: "#d9901a" },
  error: { fill: "#fdecec", stroke: "#d33a3a", text: "#9b1c1c", line: "#d33a3a" },
};
const E = [];
function el(id, kind, x, y, w, h, lines, since, opts = {}) {
  if (typeof lines[0] === "string") lines = [[since, lines]];
  E.push({ id, kind, x, y, w, h, lines, since, until: null, changed: [], badge: since !== "00", lock: null, ...opts });
}
const ED = [];
function edge(pts, since, opts = {}) {
  let label = opts.label || null;
  if (typeof label === "string") label = [[since, label]];
  ED.push({ pts, since, until: null, changed: [], dashed: false, error: false, arrow: true, lpos: null, lanchor: "middle", ...opts, label });
}

// 外框
el("f_pi", "frame", 186, 100, 500, 580, ["Pi 5 上的偵測器（這門課做的）"], "23", { dashed: true });
el("f_vis", "frame", 196, 112, 480, 536, ["更強的偵測器"], "01", { badge: false });
el("f_platform", "frame", 730, 112, 300, 330, ["告警平台"], "00", { badge: false });
el("f_ml", "frame", 730, 470, 300, 178, ["ML 課做好的"], "00", { badge: false });

el("a_cam", "actor", 24, 112, 140, 120, ["攝影機", "產線／寢室"], "00", { badge: false });
el("a_ezcam", "actor", 24, 260, 140, 60, ["骨架串流", "10 fps"], "19");
el("a_handler", "actor", 1070, 112, 240, 200, ["值班人員", "主管／護理師"], "00", { badge: false });
el("a_next", "external", 1070, 560, 240, 60, ["交接", "客戶與維運"], "35");

// 前處理（01–07）
el("v_pre", "service", 212, 150, 140, 46, [["01", ["前處理", "CLAHE"]], ["02", ["前處理", "CLAHE＋去噪"]], ["03", ["前處理", "＋轉正"]]], "01", { changed: ["02", "03"] });
el("v_rule", "service", 364, 150, 140, 46, [["04", ["傳統量測", "Canny"]], ["05", ["傳統量測", "＋霍夫、輪廓"]], ["06", ["規則 baseline", "超出範圍就判"]]], "04", { changed: ["05", "06"] });
el("v_aug", "service", 516, 150, 144, 46, ["資料增強", "transforms.v2"], "07");
// 異常偵測（08–13）
el("v_pix", "service", 212, 216, 140, 46, ["像素差", "baseline"], "08");
el("v_feat", "service", 364, 216, 140, 46, [["09", ["預訓練特徵", "整張 kNN"]], ["10", ["小格 kNN", "PatchCore 簡化"]]], "09", { changed: ["10"] });
el("v_heat", "service", 516, 216, 144, 46, ["熱圖", "異常在哪裡"], "11");
el("v_thr", "service", 212, 282, 140, 46, ["門檻", "校正＋少量異常"], "12");
el("v_board", "service", 364, 282, 140, 46, [["13", ["成績表", "依類型"]], ["18", ["成績表", "＋YOLO 對照"]]], "13", { changed: ["18"] });
// YOLO（14–18）
el("v_cls", "service", 516, 282, 144, 46, ["YOLO 分類", "good／anomaly"], "14");
el("v_box", "service", 212, 348, 140, 46, ["遮罩變成框", "YOLO 資料集"], "15");
el("v_det", "service", 364, 348, 140, 46, [["16", ["YOLO 偵測", "訓練"]], ["17", ["YOLO 偵測", "val＋mAP"]]], "16", { changed: ["17"] });
// 串流（19–22）
el("v_stream", "service", 212, 414, 140, 46, [["19", ["骨架串流", "最近 1 秒"]], ["20", ["骨架串流", "＋追蹤"]]], "19", { changed: ["20"] });
el("v_sm", "service", 364, 414, 140, 46, [["21", ["狀態機", "normal→alert"]], ["22", ["狀態機", "＋冷卻"]]], "21", { changed: ["22"] });
// 部署（23–26）
el("v_onnx", "service", 212, 480, 140, 46, [["23", ["ONNX", "匯出"]], ["24", ["ONNX", "＋onnxruntime"]], ["25", ["ONNX", "＋int8"]]], "23", { changed: ["24", "25"] });
el("v_budget", "service", 364, 480, 140, 46, ["延遲預算", "解析度取捨"], "26");
// 接平台（27–35）
el("v_send", "service", 516, 414, 144, 46, [["27", ["送事件", "新事件類型"]], ["28", ["送事件", "＋異常 MVP"]], ["29", ["送事件", "＋骨架 MVP"]]], "27", { changed: ["28", "29"] });
el("v_mock", "service", 516, 480, 144, 46, ["Mock 測試", "不用攝影機"], "31");
el("v_check", "service", 212, 546, 140, 46, ["上線清單", "指紋＋紀錄"], "32");
el("v_accept", "service", 364, 546, 140, 46, [["33", ["驗收報告", "先寫標準"]], ["34", ["驗收報告", "＋互審"]]], "33", { changed: ["34"] });
el("v_hand", "store", 516, 546, 144, 46, ["交接包", "SHA256SUMS"], "35");

// 平台（灰）
el("p_events", "port", 746, 158, 130, 30, ["POST /events"], "00", { badge: false });
el("p_prof", "chip", 886, 160, 130, 26, [["00", ["profiles"]], ["27", ["profiles＋新事件"]]], "00", { badge: false, changed: ["27"] });
el("p_sm", "service", 746, 210, 270, 60, ["告警狀態機", "open → resolved／dismissed"], "00", { badge: false });
el("p_alerts", "port", 746, 290, 270, 30, [["00", ["GET /alerts、ack、resolve、dismiss"]], ["30", ["GET /alerts＋真告警率"]]], "00", { badge: false, changed: ["30"] });
el("n_platform", "note", 750, 360, 0, 0, ["值班人員的確認＝上線後的標籤"], "00", { badge: false, anchor: "start", size: 10.5 });
// ML 課（灰）
el("m_judge", "service", 746, 510, 130, 50, ["判斷器", "ML 01–15"], "00", { badge: false });
el("m_det", "service", 886, 510, 130, 50, ["偵測器", "ML 16–24"], "00", { badge: false });
el("m_extra", "service", 746, 576, 270, 50, ["ROI、骨架、串流、pytest", "ML 25–42"], "00", { badge: false });

edge([[164, 160], [212, 160]], "01", { label: "影像", lpos: [188, 152] });
edge([[164, 290], [190, 290], [190, 437], [212, 437]], "19");
edge([[660, 437], [700, 437], [700, 173], [746, 173]], "27", { label: "requests.post", lpos: [704, 300], lanchor: "start" });
edge([[1016, 240], [1070, 240]], "00", { label: "通知", lpos: [1043, 234] });
edge([[588, 592], [588, 610], [1050, 610], [1050, 590], [1070, 590]], "35");
// --------------------------------------------------
// 繪製
// --------------------------------------------------
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const isAscii = (s) => [...s].every((c) => c.charCodeAt(0) < 128);

function stateOf(item, t) {
  if (FULL) return "old";
  if (item.since === t) return "new";
  if (item.changed.includes(t)) return "changed";
  return "old";
}

function visible(item, t) {
  if (vi(item.since) > vi(t)) return false;
  if (item.until && vi(t) > vi(item.until)) return false;
  return true;
}

function atVersion(variants, t) {
  let cur = variants[0][1];
  for (const [since, value] of variants) if (vi(since) <= vi(t)) cur = value;
  return cur;
}

function text(x, y, s, o = {}) {
  const size = o.size || 13, fill = o.fill || "#333", weight = o.weight || "normal", anchor = o.anchor || "middle";
  const mono = o.mono === undefined ? isAscii(s) : o.mono;
  return `<text x="${x}" y="${y}" font-family="${mono ? MONO : SANS}" font-size="${size}" fill="${fill}" ` +
    `font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
}

function lockIcon(x, y, color) {
  return `<g transform="translate(${x},${y})" fill="none" stroke="${color}" stroke-width="1.4">` +
    `<rect x="-5" y="-1" width="10" height="8" rx="1.5" fill="${color}" stroke="none"/>` +
    `<path d="M-3,-1 v-3 a3,3 0 0 1 6,0 v3"/></g>`;
}

function badge(x2, y, v, st) {
  const label = `v${v}`;
  const w = label.length * 6.4 + 10;
  const [bg, fg] = st === "new" ? ["#1f9d4c", "#fff"] : ["#e9ebee", "#5b616b"];
  return `<rect x="${x2 - w}" y="${y - 8}" width="${w}" height="15" rx="7.5" fill="${bg}"/>` +
    text(x2 - w / 2, y + 3.5, label, { size: 10, fill: fg, mono: true });
}

function render(t) {
  FULL = t === "full";
  const titleKey = t;
  if (FULL) t = "35";
  if (!VERS.includes(t)) throw new Error(`沒有這個版本：${t}`);
  const out = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`, "<defs>"];
  for (const [name, p] of Object.entries(PALETTE)) {
    out.push(`<marker id="arr-${name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" ` +
      `orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${p.line}"/></marker>`);
  }
  out.push("</defs>");
  out.push(`<rect width="${W}" height="${H}" fill="#ffffff"/>`);

  // 標題和圖例
  const [title, chapter] = TITLES[titleKey];
  out.push(text(24, 40, FULL ? title : `v${t}　${title}`, { size: 22, fill: "#1f2328", weight: "bold", anchor: "start", mono: false }));
  out.push(text(24, 64, FULL ? `${chapter}　｜　箭頭 = 資料往哪裡走` : `這一節：${chapter}　｜　箭頭 = 資料往哪裡走　｜　小標籤 = 第幾節出現`,
    { size: 13, fill: "#6b7280", anchor: "start", mono: false }));
  let lx = 760;
  if (!FULL) {
  for (const [st, label] of [["old", "之前就有"], ["new", "這一節新增"], ["changed", "這一節改了"]]) {
    const p = PALETTE[st];
    out.push(`<rect x="${lx}" y="27" width="16" height="16" rx="3" fill="${p.fill}" stroke="${p.stroke}" stroke-width="1.5"/>`);
    out.push(text(lx + 22, 40, label, { size: 12, fill: "#444", anchor: "start", mono: false }));
    lx += 112;
  }
  }

  const badges = [];
  const order = { frame: 0, group: 1 };
  const shown = E.filter((e) => visible(e, t)).sort((a, b) => (order[a.kind] ?? 2) - (order[b.kind] ?? 2));
  for (const e of shown) {
    const st = stateOf(e, t);
    const p = PALETTE[e.kind === "error" ? "error" : st];
    const ls = atVersion(e.lines, t);
    const { x, y, w, h, kind: k } = e;
    const sw = st !== "old" ? 2 : 1.3;

    if (k === "frame") {
      const dash = e.dashed ? ' stroke-dasharray="7 5"' : "";
      const fill = e.dashed ? "none" : "#fbfbfc";
      const stroke = st !== "old" ? p.stroke : "#9aa0a8";
      out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="16" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash}/>`);
      if (e.dashed) {
        out.push(text(x + 14, y + h - 7, ls[0], { size: 12, fill: p.text, anchor: "start", mono: false }));
        badges.push([x + 14 + ls[0].length * 12.5 + 46, y + h - 11, e.since, st]);
      } else {
        out.push(text(x + 16, y + 28, ls[0], { size: 16, fill: "#1f2328", weight: "bold", anchor: "start", mono: false }));
      }
      continue;
    }

    if (k === "group") {
      if (e.solid) {
        out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${p.fill}" stroke="${p.stroke}" stroke-width="${sw}"/>`);
        out.push(text(x + 12, y + 22, ls[0], { size: 13, fill: p.text, weight: "bold", anchor: "start" }));
      } else {
        out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="none" stroke="${p.stroke}" stroke-width="${sw}" stroke-dasharray="5 4"/>`);
        out.push(text(x + 10, y + 14, ls[0], { size: 10.5, fill: p.text, anchor: "start" }));
      }
      if (e.badge) badges.push([x + w - 4, y, e.since, st]);
      continue;
    }

    if (k === "note") {
      const fill = st !== "old" ? PALETTE[st].text : "#6b7280";
      out.push(text(x, y, ls[0], { size: e.size || 12, fill, anchor: e.anchor || "middle", mono: false }));
      continue;
    }

    const rx = { port: h / 2, chip: h / 2, error: h / 2, actor: 12, external: 12 }[k] ?? 8;
    const fill = (["port", "actor", "external"].includes(k) && st === "old") ? "#ffffff" : p.fill;
    const dash = k === "external" ? ' stroke-dasharray="6 4"' : "";
    if (k === "store" && !e.labelTop) {
      out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${p.stroke}" stroke-width="${sw}"/>` +
        `<path d="M${x},${y + 8} h${w}" stroke="${p.stroke}" stroke-width="1"/>`);
    } else {
      out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${p.stroke}" stroke-width="${sw}"${dash}/>`);
    }

    let cx = x + w / 2;
    if (e.lock && visible({ since: e.lock }, t)) {
      const lk = e.lock === t ? "new" : "old";
      out.push(lockIcon(x + 15, y + h / 2 - 1, PALETTE[lk].line));
      cx += 7;
    }

    if (e.labelTop) {
      out.push(text(x + 10, y + 20, ls[0], { size: 13, fill: p.text, weight: "bold", anchor: "start" }));
    } else {
      const big = { actor: 15, external: 15, service: 13.5 }[k] ?? 10.5;
      const small = 10.5;
      if (ls.length === 1) {
        const size = k !== "chip" ? big : (e.id.startsWith("st_") ? 9.5 : 10.5);
        const bold = ["actor", "external", "service"].includes(k);
        out.push(text(cx, y + h / 2 + big * 0.36, ls[0], { size, fill: p.text, weight: bold ? "bold" : "normal" }));
      } else {
        const gap = ["actor", "external"].includes(k) ? 16 : 14;
        const top = y + h / 2 - (ls.length - 1) * gap / 2 + 4;
        ls.forEach((s, i) => {
          const first = i === 0 && ["actor", "external", "service", "store"].includes(k);
          out.push(text(cx, top + i * gap, s, { size: first ? big : small, fill: p.text, weight: first ? "bold" : "normal" }));
        });
      }
    }
    if (e.badge) badges.push([x + w + 4, y, e.since, st]);
  }

  for (const d of ED) {
    if (!visible(d, t)) continue;
    const st = d.error ? "error" : stateOf(d, t);
    const p = PALETTE[st];
    const pts = d.pts.map(([a, b]) => `${a},${b}`).join(" ");
    const dash = (d.dashed || d.error) ? ' stroke-dasharray="5 4"' : "";
    const marker = d.arrow ? ` marker-end="url(#arr-${st})"` : "";
    const sw = (st === "new" || st === "changed") ? 1.8 : 1.4;
    out.push(`<polyline points="${pts}" fill="none" stroke="${p.line}" stroke-width="${sw}"${dash}${marker}/>`);
    if (d.label) {
      const [lx2, ly2] = d.lpos;
      out.push(`<text x="${lx2}" y="${ly2}" font-family="${SANS}" font-size="10.5" fill="${p.text}" ` +
        `text-anchor="${d.lanchor}" stroke="#fff" stroke-width="4" paint-order="stroke">${esc(atVersion(d.label, t))}</text>`);
    }
  }

  for (const [bx, by, v, st] of badges) out.push(badge(bx, by, v, st));
  out.push("</svg>");
  return out.join("\n") + "\n";
}

if (typeof process !== "undefined" && process.argv && process.argv[1] && process.argv[1].endsWith("gen.js")) {
  const arg = process.argv[2] || "35";
  if (arg === "all") {
    const fs = require("fs");
    const path = require("path");
    for (const v of VERS.slice(1)) fs.writeFileSync(path.join(__dirname, `v${v}.svg`), render(v));
    fs.writeFileSync(path.join(__dirname, "full.svg"), render("full"));
  } else {
    process.stdout.write(render(arg));
  }
}

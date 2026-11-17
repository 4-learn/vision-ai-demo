# 影像 AI 淬鍊實戰專案（2026 v2）— 範例程式與進度存檔

接在「Python 機器學習」之後：替告警平台做**更強的偵測器，並把它裝到現場的 Pi 5 上**。

| | 工廠線 | 健康線 |
| --- | --- | --- |
| 異常偵測 | 螺帽瑕疵（MVTec AD metal_nut） | 寢室畫面異常（倒地、障礙物、鏡頭被遮） |
| 骨架串流 | 彎腰搬重物 | 跌倒 |

- `demo/VNN-*/demo.py`：講解用的小例子；`story/vNN-*/vision.py`：進度存檔，在該資料夾執行 `LINE=factory python vision.py`。
- **教師開課前**：`python tools/build_data.py`（工廠線需先下載 MVTec AD 並設 `MVTEC_ROOT`）→ `python tools/train_models.py` → `python tools/build_tables.py`。
- 工廠線影像是 MVTec AD（CC BY-NC-SA 4.0）衍生，只在課堂使用、**不隨 repo 散佈**；健康線為程式畫的合成畫面。ultralytics 為 AGPL-3.0。

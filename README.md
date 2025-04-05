# lane-detection
A deep learning-based lane detection system with performance optimization

# Lane Detection System

## 📌 專案簡介

本程式透過 OpenCV 進行車道辨識，針對影片中車道線進行偵測、追蹤與區域填色。程式可即時處理影片影格，辨識出左右車道線，計算並標示車道中心與車輛中心位置，最終以綠色半透明方式將車道區塊填滿，提升辨識結果的可視性與精準度。程式特別限制偵測範圍於畫面下半部，減少干擾並提升偵測效能，適用於各類駕駛輔助系統或自動駕駛前處理模組。

---

## 🚀 動機與目標

**開發動機**
隨著智慧車輛與自動駕駛技術的蓬勃發展，穩定且即時的車道辨識技術已成為先進駕駛輔助系統（ADAS）中不可或缺的一環。本專案的初衷，是希望建立一個簡潔、高效且適合嵌入式平台的車道辨識模組，能在不依賴深度學習模型的情況下完成關鍵辨識任務，適用於早期開發階段或資源有限的場域應用。

**開發目標**
**準確偵測車道線**：透過邊緣偵測與 Hough 轉換演算法，準確找出左右兩側車道線的位置。\n
**限制辨識區域**：僅在畫面下半部設定 ROI，有效排除遠方干擾與非目標區域。\n
**車道區域鋪色填滿**：將左右車道線圍出的區域填色，使結果更清楚且可用於後續判斷。\n
**標示車道中心與車輛中心**：輔助進行偏移量判斷或自動駕駛決策依據。\n
**具即時處理能力**：可直接套用至影片或影像串流，做為實驗平台或實際部署的前端模組。\n

---

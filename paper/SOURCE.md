# 出處說明（Source）

本目錄收錄 FlashMap 論文的**純文字版全文**，取自 arXiv 的 HTML 版本並剝除標記。

| 項目 | 內容 |
|---|---|
| 論文標題 | FlashMap: Flow Map Speculative Decoding |
| arXiv 編號 | arXiv:2610.24365v1 [cs.CL] |
| 上傳日期 | 2026-10-17 |
| 作者 | Jeff Shen, Zhejian Zhou, Ryan Lee, Jonathan May, Shirley Ho |
| 機構 | Polymathic AI / Princeton University / USC Information Sciences Institute / NYU / Flatiron Institute |
| 授權 | CC BY 4.0 |
| 原文 | https://arxiv.org/abs/2610.24365 ｜ https://arxiv.org/html/2610.24365v1 |
| 專案頁 | https://jshen.net/flashmap |

## 處理方式

- 來源為 `https://arxiv.org/html/2610.24365v1` 的 HTML 全文（492 KB）。
- 以腳本剝除 `<script>` / `<style>` / 標籤與空白壓縮後，存成 `flashmap_paper.txt`（約 61 KB）。
- **數學式以純文字形式呈現**（如 `I_t = (1-t)·x₀ + t·x₁`），部分符號（上下標、希臘字母）
  在剝除過程中可能失真；對照時請以 arXiv 原文（HTML 或 PDF）為準。
- 原始 HTML 檔未收錄於本 repo（480 KB 的網頁標記，無實用價值）；需要時請自 arXiv 取得。

## 使用方式

依 CC BY 4.0，允許自由使用與散布（含商業用途），但須保留出處與作者資訊。
引用時建議直接指向 <https://arxiv.org/abs/2610.24365>。

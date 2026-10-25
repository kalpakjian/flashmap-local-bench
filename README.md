# FlashMap 本機實驗與論文拆解

> 針對 **FlashMap: Flow Map Speculative Decoding**（[arXiv:2610.24365](https://arxiv.org/abs/2610.24365)，2026-10-17）
> 的完整機制拆解、數據核對，以及一套可在 **Omarchy / Arch Linux + RTX 4090** 上直接跑的本機實驗腳本。

FlashMap 是一個**推測解碼（speculative decoding）的 drafter**：用連續流的 flow map 在**一次前傳**內
把整塊 16 個 token「整體」產生出來（保留 token 間相依性），交給 target 一次前傳驗證。
論文宣稱在 Qwen3-8B 上最高 **6.14× 無損加速**。

本 repo 提供三件事：

1. **📓 論文拆解筆記** — 公式、機制、shuffle test 數據、與 DFlash/EAGLE-3 的逐項比較、限制條款
2. **🧪 可執行實驗腳本** — 在自己的機器上量出接受長度 γ、無損性、端到端 tok/s
3. **🔍 媒體報導查核** — 「6.1 倍」這個數字到底代表什麼、哪些限定條件被省略了

---

## 快速開始

### Linux（完整流程）

```bash
git clone https://github.com/kalpakjian/flashmap-local-bench.git
cd flashmap-local-bench/lab

bash setup.sh      # 建環境 + 下載模型（Qwen3-8B + FlashMap drafter，約 17.5 GB）
bash run_all.sh    # Phase 1: γ + 無損性；Phase 2: 端到端 tok/s
```

### Windows 原生（只跑 Phase 1）

```powershell
git clone https://github.com/kalpakjian/flashmap-local-bench.git
cd flashmap-local-bench\lab

powershell -ExecutionPolicy Bypass -File setup.ps1     # 建環境 + 下載模型
powershell -ExecutionPolicy Bypass -File run_all.ps1    # 跑 Phase 1
```

> **Windows 只能跑一半**：接受長度與無損性（Phase 1）可以原生執行；
> 端到端 tok/s（Phase 2）需要 vLLM，而 vLLM / SGLang 在 PyPI **只有 manylinux wheel**，
> 必須走 Linux（WSL2 或 Omarchy）。詳見 [`lab/README.md`](lab/README.md) §1b。

詳細說明、錯誤處理、量測設計理由見 **[`lab/README.md`](lab/README.md)**。

---

## 目錄結構

| 路徑 | 內容 |
|---|---|
| [`notes/flashmap-paper-notes.md`](notes/flashmap-paper-notes.md) | 完整筆記：背景公式、FlashMap 三個關鍵設計、與 DFlash/EAGLE-3 比較、全部數據表、限制與本機部署評估 |
| [`lab/`](lab/) | 實驗腳本（Linux + Windows 兩套）：`setup.sh`/`setup.ps1`、`run_all.sh`/`run_all.ps1`、`acceptance.py`、`bench.py`、`env_check.py`、`selftest.py`、`prompts.json` |
| [`paper/flashmap_paper.txt`](paper/flashmap_paper.txt) | arXiv 論文全文（剝除 HTML 後的純文字版，CC BY 4.0） |

---

## 核心結論（皆可在筆記中對到原文章節）

| # | 結論 | 依據 |
|---|---|---|
| 1 | **6.14× 是 18 格實驗中的單一最高格，不是平均值**。9 項 benchmark 平均：T=0 **4.36×**、T=1 **3.48×**；聊天類只有 **2.5–3.0×** | Figure 1(c) |
| 2 | **「無損」來自推測解碼的接受規則**，不是 FlashMap 本身；greedy 下輸出逐字相同。媒體說法正確，但因果要講清楚 | §2.1 |
| 3 | **joint structure 是「積分」製造的，不是模型自帶的**：teacher flow 在 1 NFE 時 Δ=0.00，積分 16 步後 Δ=0.82；FlashMap 1 NFE 只保留約 39% | §6.2 shuffle test |
| 4 | **在 Qwen3-4B 上 FlashMap 沒有全面勝出**（平均 3.78× vs DFlash 3.81×）。摘要的「比 DFlash 快」只對 Qwen3-8B 成立 | Appendix C, Table 3 |
| 5 | **量測條件是 H200 NVL、concurrency=1、thinking off**；批次服務作者明確列為未評估 | §4、Limitations |

---

## 實驗設計要點

- **Phase 1（`acceptance.py`）** 不起 server，直接用 flashmap 套件的 transformers 參考實作，
  量出 `acceptance_length`（γ），並把 FlashMap 輸出與 target greedy 輸出**逐字比對**驗證無損。
- **Phase 2（`bench.py`）** 依序啟動兩個 vLLM server，**唯一差異是 `--speculative-config`**，
  同一組 prompt、concurrency=1、temperature=0、thinking off — 與論文量測條件對齊。
- 腳本會從 vLLM 的 `/metrics` 抓 spec-decode counter，交叉驗證接受率。

**預期量級**（RTX 4090 24 GB）：Qwen3-8B 逐字約 40–55 tok/s；
若 FlashMap 達到論文比例（數學/程式 4–6×），應落在 150–250 tok/s，聊天約 100–150 tok/s。

---

## 硬體與軟體需求

| 項目 | 需求 |
|---|---|
| OS | Linux（本 repo 以 **Omarchy / Arch** 為主要目標，Ubuntu/Debian 亦可） |
| GPU | NVIDIA，24 GB VRAM 以上（Qwen3-8B bf16 16.4 GB + drafter 1.0 GB） |
| 驅動 | 需支援 CUDA 13（torch 2.13.0 的 cu13 wheel） |
| 磁碟 | ≥ 35 GB（模型 17.5 GB + venv 約 12 GB + 快取） |
| Python | **3.12**（用 `uv venv --python 3.12` 隔離，不動系統 Python） |

套件版本（2026-10-26 由 PyPI API 查證，非猜測）：

- `flashmap` 0.1.0 — `requires-python >=3.10`
- `vllm` **0.31.0**（唯一符合 flashmap 的 `>=0.31,<0.32`）— 鎖 `torch==2.13.0`、`transformers>=5.10.4,<5.18.0`
- `sglang` 0.5.20（備用路徑）— 鎖 `torch==2.13.0`

---

## 出處與授權

| 項目 | 來源 | 授權 |
|---|---|---|
| FlashMap 論文 | [arXiv:2610.24365](https://arxiv.org/abs/2610.24365) | CC BY 4.0（作者：Jeff Shen, Zhejian Zhou, Ryan Lee, Jonathan May, Shirley Ho） |
| FlashMap 程式碼 | [github.com/al-jshen/flashmap](https://github.com/al-jshen/flashmap) | MIT |
| FlashMap 權重 | [j-shen/FlashMap-Qwen3-8B](https://huggingface.co/j-shen/FlashMap-Qwen3-8B) | Apache-2.0 |
| 本 repo 的筆記與腳本 | 本 repo | CC BY 4.0（見 [LICENSE](LICENSE)） |

`paper/flashmap_paper.txt` 為 arXiv 論文的純文字版，收錄於本 repo 依 CC BY 4.0 授權，
出處說明見 [`paper/SOURCE.md`](paper/SOURCE.md)。

## 免責聲明

- 本 repo **不是** FlashMap 官方 repo，與原作者無關聯。
- 腳本的 Linux 路徑（pacman / uv / NVIDIA 驅動 / vLLM plugin 載入）尚未在真機上執行過，
  首次執行若有問題，請把 `vllm_*.log` 一併回報。
- 筆記與腳本由 AI 輔助整理，**所有數字均取自 arXiv 原文並在文中標註對應章節與表格**。

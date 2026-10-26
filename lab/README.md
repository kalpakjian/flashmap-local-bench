# FlashMap 本機實驗（Omarchy / RTX 4090）

重現 FlashMap 論文（[arXiv:2610.24365](https://arxiv.org/abs/2610.24365)）的加速效果，
在這台機器上量測 **Qwen3-8B + j-shen/FlashMap-Qwen3-8B** 的：

1. **接受長度 γ**（acceptance length）與 **無損性**（FlashMap 輸出是否與逐字生成完全相同）
2. **端到端 tok/s**（vLLM，基準 vs FlashMap）

> 對應筆記：`../notes/flashmap-paper-notes.md`（§7 有本機部署評估與 VRAM 試算）

---

## 0. 前置條件

| 項目 | 需求 | 檢查方式 |
|---|---|---|
| OS | Omarchy（Arch 系） | `grep PRETTY_NAME /etc/os-release` |
| GPU | RTX 4090 24 GB | `nvidia-smi` |
| 驅動 | 需支援 CUDA 13（本機 Windows 側驅動 621.01 / CUDA UMD 13.5，Omarchy 側請用相近或更新版） | `nvidia-smi` 頂端有 `CUDA Version` |
| 磁碟 | 建議 ≥ 35 GB 可用（模型 17.5 GB + venv 約 12 GB + 快取） | `df -h ~` |
| 記憶體 | 64 GB 足夠（模型載入用 VRAM，CPU 記憶體主要給編譯快取） | `free -h` |

Omarchy 沒裝 NVIDIA 驅動時：

```bash
omarchy install nvidia && sudo mkinitcpio -P && reboot
```

---

## 1. 取得這些腳本

### 方式 A：從 GitHub clone（建議）

```bash
git clone https://github.com/kalpakjian/flashmap-local-bench.git
cd flashmap-local-bench/lab
```

### 方式 B：從 Windows 分割區複製（網路不通時）

Omarchy 通常不會自動掛載 Windows 分割區。先找出它：

```bash
lsblk -f | grep -i ntfs
```

假設是 `/dev/nvme0n1p2`（**實際代號以你的輸出為準**）：

```bash
sudo mkdir -p /mnt/win
sudo mount -o ro /dev/nvme0n1p2 /mnt/win
cp -r /mnt/win/c/Users/<你的帳號>/flashmap-local-bench/lab ~/
cd ~/flashmap-local-bench/lab
```

> **換行符**：Windows 寫出的檔案可能是 CRLF，若 bash 報 `invalid option` 或 `\r` 錯誤，執行
> `sed -i 's/\r$//' *.sh` 修正（本 repo 的檔案已確認為 LF，但保險起見）。

---

## 2. 三步執行

```bash
cd ~/omarchy-flashmap

# ① 建環境（冪等；會下載 ~17.5GB 模型，約 10–30 分鐘）
bash setup.sh

# ② 跑實驗（Phase 1 接受長度+無損性，Phase 2 端到端 tok/s）
bash run_all.sh
```

`setup.sh` 做什麼：

- 檢查 pacman / nvidia-smi / 磁碟空間
- 裝 `uv`（若沒有）→ `uv venv --python 3.12`（**用 uv 管理的 3.12，不動系統 Python**）
- 裝 `vllm==0.31.0` + `flashmap[vllm]` + `huggingface_hub`
  （vLLM 只有 `0.31.0` 符合 flashmap 的 `>=0.31,<0.32`；vLLM 0.31.0 與 flashmap 都鎖 `torch==2.13.0`）
- CUDA bf16 smoke test
- 下載 `Qwen/Qwen3-8B` 與 `j-shen/FlashMap-Qwen3-8B` 到 `$HF_HOME`（預設 `~/flashmap-lab/hf-cache`）

---

## 3. 產出與怎麼讀

| 檔案 | 內容 |
|---|---|
| `results_acceptance_step1.json` | 每個 prompt 的 γ、是否無損、target/spec 各自秒數 |
| `results_vllm.json` | 兩種設定的逐筆延遲、TTFT、token 數、server `/metrics` |
| `report.md` | 彙總表 + 與論文數字的對照 + 解讀注意事項 |
| `vllm_baseline.log` / `vllm_flashmap.log` | 兩個 server 的完整 log（出錯時先看這裡） |

**判讀順序**

1. `report.md` 的「ALL」列 → 加速比（這是本實驗的主指標）
2. `report.md` 的「伺服器端輸出比對」→ **這才是使用者真正在意的無損性**：
   同一個 vLLM、temperature=0，開關 FlashMap 前後回答有沒有變。
3. `results_acceptance_step1.json` 的 `summary.ALL.all_lossless` → **這一項在 bf16 下不可信**。
   `torch.equal` 逐 token 比對，而 bf16 的 logits 解析度只有 ~2⁻⁸ ≈ 0.0039（在 logit≈1 處），
   top-2 logit 差距小於 bf16 評測誤差時，argmax 會由四捨五入決定 ——
   這不是演算法錯誤。**嚴格檢查請用 `--dtype float32`**（上游 `examples/benchmark.py`
   明文建議），此時同一組程式在 4B 上 12/12 逐 token 完全相同。
   詳見 `FINDINGS.md` 與 `pass3_bf16_proof.py` / `pass4_bf16_shape.py` / `pass5_fp32_cpu_truth.py`。
4. 接受長度 γ 對照論文（Qwen3-8B, T=0, 1 step）：數學 ~6.0–7.7、程式 ~5.0–6.7、聊天 ~3.1–3.7。
   本機若明顯偏低，先確認 `--max-tokens` 沒太小（太短會讓接受長度統計失真）。
5. 聊天類加速本來就低於數學/程式（開源性文字有很多合理續寫），這與論文一致。
6. `lossless_prefix_ok` 與 `spec_overshoot`：最後一次驗證 pass 會衝過 `max_new_tokens`
   （上游文件明載），此時兩者互為前綴、在 `max_new_tokens` 內完全一致，
   不應算成無損性失敗。

---

## 4. 常見錯誤處理

| 症狀 | 原因 | 處理 |
|---|---|---|
| `vllm serve` 啟動時報 `/usr/local/cuda/bin/nvcc: No such file` | FlashInfer 需要 nvcc 才能 JIT 編譯 sampling kernel | `export VLLM_USE_FLASHINFER_SAMPLER=0`（腳本已預設），或 `bash setup.sh --with-cuda` 裝 CUDA Toolkit |
| `ImportError: undefined symbol` / `GLIBC` 相關 | Arch 的 glibc 太新或太舊 | 用 uv 的 Python 3.12（腳本已指定）；不要用 Arch 系統 Python |
| `No module named flashmap` / plugin 未載入 | venv 沒啟用或安裝時漏了 extra | `source ~/flashmap-lab/.venv/bin/activate`；重跑 `uv pip install "vllm==0.31.0" "flashmap[vllm]"` |
| CUDA out of memory | 24 GB 卡塞不下 | 降 `--gpu-memory-utilization`（bench.py 內，預設 0.90 → 0.80）、降 `--max-model-len`、或用 Qwen3-4B drafter |
| `transformers` 版本衝突 | vLLM 0.31.0 要求 `>=5.10.4,<5.18.0` | 別手動升級 transformers，讓 vLLM 決定 |
| server 起來但加速比 <1 | drafter 沒真的啟用 | 查 `vllm_flashmap.log` 是否有 speculative decoding 字樣；確認 `--speculative-config` 格式正確 |
| 只想快速驗證無損性 | 不需要起 server | `python acceptance.py --max-new-tokens 256 --dtype float32` |
| `all_lossless=false` 但想知道是不是真的壞了 | bf16 的近似並列 | 跑 `python acceptance.py --dtype float32`；若 12/12 相同就是數值假象（見 `FINDINGS.md`） |

---

## 5. 檔案清單

| 檔案 | 用途 |
|---|---|
| `setup.sh` | 冪等環境建置：檢查 → 裝 uv → py3.12 venv → 裝 vllm/flashmap → CUDA smoke → 下載模型 |
| `run_all.sh` | 一鍵跑完 Phase 1 + Phase 2 並列出產出 |
| `acceptance.py` | Part 1：接受長度 γ + 無損性（transformers 參考實作，不需要 server） |
| `bench.py` | Part 2：端到端 tok/s（vLLM，基準 vs FlashMap）＋報告產生＋伺服器端輸出比對 |
| `pass3_bf16_proof.py` | 機制驗證：每個分歧點在共享前綴上的 fp32 真值 vs bf16 解析度 |
| `pass4_bf16_shape.py` | 機制驗證：基準呼叫簽章（2×2）＋前向形狀（batch 1 vs 17）對 bf16 argmax 的影響 |
| `pass5_fp32_cpu_truth.py` | 機制驗證：8B 的 fp32 真值（CPU 上做，繞過 24 GB 限制） |
| `selftest.py` | 不需 GPU 的邏輯自測，重開機後可先跑 |
| `prompts.json` | 12 題固定測試集（數學 5 / 程式 4 / 聊天 3） |
| `FINDINGS.md` | 驗證結論：Phase 1 的 `all_lossless=false` 是 bf16 數值假象的完整證據鏈 |
| `FINDINGS-easy.md` | 同上，但沒有數字與公式的白話版（給不想讀表格的人） |
| `README.md` | 本文件 |

---

## 6. 設計說明（為什麼這樣量）

**為什麼 Phase 1 不用 server？** `flashmap` 套件附 transformers 參考實作，直接呼叫
`draft.spec_generate(...)` 就能拿到 `acceptance_length`，而且能逐字比對 target 的 greedy 輸出，
是驗證「無損」最乾淨的方式（server 層可能因採樣實作差異產生雜訊）。

**為什麼要兩個 server 分開跑？** 同一個 prompt 集合、同樣的參數，唯一的差別是
`--speculative-config`。這樣加速比只反映推測解碼本身的貢獻。

**為什麼 temperature=0？** 論文的 T=0 結果是「無損」的強形式（輸出逐字相同）；
T=1 需要 `draft_sample_method: probabilistic` 才保證分布一致，vLLM 端的行為較難驗證。

**為什麼 concurrency=1？** 論文的量測就是 concurrency=1；批次服務的加速作者明確列為未評估，
若本機先量到 concurrency=1 的數字，才對得上論文。

**預期量級**：RTX 4090 上 Qwen3-8B 逐字生成約 40–55 tok/s；
FlashMap 在數學/程式若達到論文比例（4–6×），應落在 150–250 tok/s；聊天約 100–150 tok/s。
若實測遠低於此，優先懷疑 drafter 未啟用（查 `vllm_flashmap.log`）或 GPU 被別的程式占用。



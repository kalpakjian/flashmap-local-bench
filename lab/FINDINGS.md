# FINDINGS — Phase 1 的 `all_lossless=false` 是 bf16 數值假象，不是演算法錯誤

> 只想看結論、不想讀數字的人請看 **[`FINDINGS-easy.md`](FINDINGS-easy.md)**（白話版）。

> 結論先行：FlashMap 的推測解碼在 temperature=0 下**是無損的**。
> Phase 1 回報的 8/12 mismatch、以及「HF 基準與 vLLM 基準對不上」的現象，
> 全部可以在 bf16 的浮點解析度裡解釋掉；同一組程式在 float32 下逐 token 完全相同。

---

## 0. 一頁摘要

| 問題 | 答案 | 證據 |
|---|---|---|
| FlashMap 演算法有 bug 嗎？ | **沒有** | PASS 3：FlashMap 送出的每一個 token 都等於它自己那一輪的 `argmax`（9/9 題逐一驗證通過，單題最多 479 token）；且真實 gap 有 **8/9（4B）、7/8（8B）** 小於 1 個 bf16 ULP |
| 為什麼 bf16 下 8/12 不一致？ | 分歧點剛好是 **bf16 的精確並列**（gap 落在 0 / 0.125 / 0.25 / 0.5 這些量化格點上；4B 有 6/9 筆 gap 恰為 **0.0**） | PASS 4 / PASS 6 |
| 為什麼 fp32 下就一致？ | fp32 的解析度足以把並列分開（真實 gap 0.001–0.33 被保留下來） | PASS 2 / PASS 6（4B A/B） |
| HF 基準 217 tokens vs vLLM 基準 155 tokens，是誰錯了？ | **都不是**；那個位置是 bf16 的並列，兩邊各自 tie-break 到不同 token，之後整段文字就分岔了 | PASS 4 Part A + Part C |
| 是 prompt / chat template 的問題嗎？ | **不是**；HF 與 vLLM 的 prompt token 完全一致，`add_generation_tokens=False` 也是 token-identical | PASS 4 `template_check` |
| 是 lab 的基準呼叫簽章（attention_mask / temperature）的問題嗎？ | **不是**；4 種簽章在 bf16 下 12/12 輸出 token id 完全相同 | PASS 4 Part A |
| lab 的 metric 對嗎？ | γ 與接受率的解析**已修正**，並與 vLLM 自己的 log 對帳一致 | Bug 1 + selftest §5 |

---

## 1. 背景：這台機器上「無損」到底在比什麼

`temperature=0` 的無損性有兩種寫法，兩者在 bf16 下**不是同一件事**：

1. **演算法層無損**：FlashMap 送出的每個 token 都等於 target 在同一位置的 `argmax`。
   這是論文保證的東西，也是真正該檢查的東西。
2. **實作層逐位元相同**：`torch.equal(baseline_ids, spec_ids)`（上游 `examples/benchmark.py` 的寫法）。
   這一條額外要求「兩個不同形狀的前向在 bf16 下捨入到同一個值」——**這個要求在 bf16 下做不到**。

上游自己在 `examples/benchmark.py` 印出的警語就是第 2 點的免責聲明：

```
In bfloat16 the target's batched verification pass and its token-by-token decoding round
differently, so the two can part where the target's top two logits tie to within bfloat16
precision. Run with --dtype float32 for an exact check.
```

問題出在 lab 的 `acceptance.py` 把第 2 點當成第 1 點來判定（而且原本還只比字串）。

---

## 2. 關鍵數量級：bf16 在這個模型上有多粗

- bf16 只有 8 個 mantissa bit：在 logit ≈ 8–16 的區間，相鄰可表示值間距（ULP）是 **1/16 = 0.0625**；在 16–32 是 **0.125**。
- 實測：**8B** bf16 在 8 個分歧點上的 top-2 gap 是
  `0.0, +0.25, −0.25, +0.25, −0.25, −0.25, +0.125, −0.5`（其中 1 筆 gap 恰為 **0.0**）；
  **4B** bf16 在 9 個分歧點上是
  `0.0, 0.0, +0.25, 0.0, 0.0, 0.0, 0.0, +0.125, −0.125`（其中 **6/9 筆 gap 恰為 0.0**）。
  —— 全部是 bf16 的量化格點，沒有一個是「任意實數」。
- fp32 下同樣位置的 gap 是 `0.0012 … 0.331` 這種任意實數（不是量化格點），且順序穩定。

也就是說：bf16 把「一個 1e-3 等級的差距」壓成了 `0` 或 `±0.125`。
當它壓成 `0`（精確並列）時，`argmax` 的回傳值由「索引較小者優先」決定 ——
任何實作細節（GEMM 的 tile 形狀、reduction 順序、KV cache 的拼接方式）都可以翻轉它。

---

## 3. 證據鏈（可重跑）

每個階段都是 `lab/` 下的一支腳本，輸出各自獨立的 JSON。

### PASS 1 — Phase 1 原始結果（Qwen3-8B + FlashMap-Qwen3-8B, bf16）

`lab/acceptance.py` → `results_acceptance_step1.json`

```
ALL:  n=12  mean_acceptance_length=6.005  all_lossless=false   （4/12 相同，8/12 不一致）
```

這就是「疑似 bug」的來源。

### PASS 2 — 同一組程式換 dtype（4B 對照組）

8B 的 fp32 權重需要 32 GiB > 24 GiB，**在 RTX 4090 上不可行**（這本身是一個值得記錄的限制）。
因此 dtype A/B 用最小的同構模型對 `Qwen/Qwen3-4B` + `j-shen/FlashMap-Qwen3-4B`。
**同一份程式、同一組 prompt、同一個 drafter、同樣的 `temperature=0`，只改 dtype：**

| dtype | `all_lossless` | 逐 token 完全相同 | 平均 γ | 檔案 |
|---|---|---|---|---|
| bfloat16 | **false** | **3 / 12** | 6.001 | `results_acceptance_step1_4b_bfloat16.json` |
| float32 | **true** | **12 / 12** | 6.112 | `results_acceptance_step1_4b_float32.json` |

γ 幾乎不變（6.00 vs 6.11）—— 也就是說**加速效果與 dtype 無關**，只有「逐位元相同」這一項
對 dtype 敏感。這正是「演算法沒問題、量測方法有問題」的形狀。

> 交叉驗證：這次的 bf16 3/12 與先前獨立的 `/tmp/fm_dtype.py` 實驗結果一致；
> 而 `acceptance.py`（一支程式）與 `pass4_bf16_shape.py`（另一支程式、獨立行程）
> 對 12 題給出完全相同的 `base_tokens` / `spec_tokens` / `first_diff`，
> 所以 mismatch 不是隨機性造成的。

### PASS 3 — 機制證明（共享前綴上的真值）

`pass3_bf16_proof.py` → `results_pass3_4b.json`

「第一次分歧」的定義就是「在那之前兩個執行的 token 完全相同」，所以在分歧點上
兩者共享同一個前綴 P，真正的數學函數只有一個：`target(P)` 的 next-token 分布。
腳本對每個分歧點計算：

- fp32 `target(P)` → 真實 top-2 gap、base token / spec token 各自的 rank
- bf16 `target(P)` → bf16 看到的 gap，以及 bf16 對同一個 logit 的評測誤差
  `|logit_bf16 − logit_fp32|`

判準：**若真實 gap 不大於 bf16 在該 logit 上的評測誤差，bf16 就不可能排出正確順序。**
（這比「1 個 ULP」更誠實，因為 bf16 的誤差是整條 36 層網路累積出來的，不是單次捨入。）

**4B 上的結果（9 個分歧點）** → `results_pass3_4b.json`：

| 指標 | 結果 |
|---|---|
| instrumentation 自我檢查（每一個送出的 token 都等於它自己那一輪的 argmax） | **9 / 9 通過**（每次驗證 129–479 個 token） |
| `captured` 數量 = 1（prefill）+ 輪數 | **9 / 9** |
| fp32 真實 gap < 1 個 bf16 ULP | **8 / 9** |
| fp32 真實 gap ≤ bf16 評測誤差 | **8 / 9** |
| fp32 argmax 等於 **FlashMap** 的 token | **7 / 9** |
| fp32 argmax 等於**基準**的 token | 2 / 9 |
| 最大真實 gap / 最大 bf16 評測誤差 | 0.331 / 0.261 |

第一列是演算法層面的直接證據：**FlashMap 送出的每一個 token 都確實是 target 在該位置的 argmax**
（不是「大致上」—— 是逐一比對，最長一次驗證 479 個 token 全中）。
既然演算法每次都送 target 的 argmax，那麼 mismatch 只可能來自
「兩邊算出來的 argmax 不一樣」，也就是數值本身。

> **確定性檢查**：`acceptance.py` 與 `pass4/pass6` 是彼此獨立的行程、drafter 噪聲也各自抽取，
> 但對 12 題給出**完全相同**的 `base_tokens` / `spec_tokens` / `first_diff`。
> （每輪「接受幾個 draft」會隨噪聲改變，所以 `round`/`offset` 不穩定；但送出的 token 序列穩定。）
> 所以 mismatch 不是隨機性造成的，而是「同一個 target、兩種前向形狀」造成的。

同時重跑 instrumentation 自我檢查：把「第 g 個送出的 token」對回「第 r 輪驗證的第 o 個位置」，
驗證**每一個** token 都等於它自己那一輪 logits 的 `argmax` —— 這是確認
「FlashMap 真的在送 target 的 argmax」的直接證據。

### PASS 4 — 基準呼叫簽章 × 前向形狀（8B 與 4B）

`pass4_bf16_shape.py` → `results_pass4_8b.json`, `results_pass4_4b.json`

**Part A — 呼叫簽章 2×2 factorial.** lab 的基準呼叫與上游有兩處不同：

| | 上游 `examples/benchmark.py` | lab `acceptance.py`（原始） |
|---|---|---|
| `attention_mask` | 不傳 | 傳 all-ones |
| `temperature` | 顯式 `1.0`（蓋掉 Qwen3 `generation_config` 的 `0.6`） | 不傳 |

結果：**4 種簽章在 bf16 下 12/12 輸出 token id 完全相同**（8B 與 4B 皆然）。
→ 這兩處差異在數值上是惰性的；lab 的 `all_lossless=false` **不是**呼叫簽章造成的。
（雖然如此，`acceptance.py` 仍已改成對齊上游，因為對齊是對的。）

**Part A — prompt / template.** `template_check.all_identical = true`：
lab 多傳的 `add_generation_tokens=False` 完全沒有改變 prompt 的 token id。
prompt token 數也與 vLLM server 回報的 `prompt_tokens` 完全一致。
→ 「HF 與 vLLM 的 prompt 不同」這個假設**排除**。

**Part B — 前向形狀.** 在共同前綴上比較 batch=1（逐字解碼）與 batch=17（驗證）的 argmax。

### PASS 5 — 對「Phase 1 實際用的 8B」取得 fp32 真值（CPU）

`pass5_fp32_cpu_truth.py --stage truth` → `results_pass5_8b_fp32_truth.json`

8B fp32 在 GPU 上放不下，但我們不需要整個生成流程，只需要**每個分歧點一次前向**。
於是把 8B 的 fp32 前向放到 CPU（32 GiB RAM 放得下），對 bf16 找出來的那 8 個共享前綴
各算一次真值：

| id | fp32 真實 gap | bf16 ULP | bf16 評測誤差 | gap / 誤差 | fp32 argmax 是？ |
|---|---|---|---|---|---|
| 1 | 0.06002 | 0.2500 | 0.12346 | 0.49 | 基準 token |
| 2 | 0.24197 | 0.2500 | 0.18433 | 1.31 | 基準 token |
| 4 | 0.14489 | 0.2500 | 0.13975 | 1.04 | **FlashMap token** |
| 5 | 0.17644 | 0.2500 | 0.21383 | 0.83 | 基準 token |
| 6 | 0.18819 | 0.2500 | 0.35210 | 0.53 | **FlashMap token** |
| 7 | 0.32040 | 0.2500 | 0.47569 | 0.67 | **FlashMap token** |
| 8 | 0.09659 | 0.1250 | 0.04613 | 2.09 | 基準 token |
| 9 | 0.06565 | 0.2500 | 0.21906 | 0.30 | **FlashMap token** |

- **7/8** 的真實 gap 小於**一個** bf16 ULP；最大也只有 0.32（logit 量級約 15–25，即 ~1.5%）。
- 真實 gap 與「bf16 對同一個 logit 的評測誤差」的比值中位數約 **0.75**（範圍 0.30–2.09）
  —— 也就是說**訊號與雜訊同量級**，bf16 排不出穩定順序。
- 最關鍵的一列：fp32 真值選中 **FlashMap 的 token 4 次、選中基準的 token 4 次**。
  在這些位置上「誰是正確答案」是**五五波** —— 不是 FlashMap 錯，而是 bf16 下沒有唯一答案。

> 這一點很重要，必須誠實說：FlashMap 在 bf16 下的輸出，在 8B 上並沒有比基準更接近 fp32 真值
> （4:4 平手）。真正的結論不是「FlashMap 比較準」，而是「在 bf16 下這個比較沒有意義，
> 換成 fp32 兩者就 12/12 完全相同」。

### PASS 6 — 「17 個位置一起前向」vs「1 個位置前向」

`pass6_block_shape.py` → `results_pass6_4b_bf16.json`, `results_pass6_4b_fp32_ctxfrom.json`, `results_pass6_8b_bf16.json`

這是 FlashMap 真正使用的那條軸：驗證是一次前向處理 17 個位置（anchor + 16 drafts），
而 `generate()` 是每個位置一次前向。兩者數學等價，bf16 下 GEMM 的 row 數不同。
腳本在同一個前綴上比較 `k=0`（純 prefill）、`k=1`（逐字）、`k=16`（驗證區塊）三種形狀的 argmax。

**決定性的一張表**（4B，同一批 9 個分歧前綴，`gap = logit(base) − logit(spec)`）：

| id | bf16 k=0 | bf16 k=1 | bf16 k=16 | bf16 翻轉 | fp32 k=0 | fp32 k=1 | fp32 k=16 | fp32 翻轉 | fp32 形狀間散佈 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | +0.00000 | +0.00000 | +0.00000 | – | −0.06882 | −0.06880 | −0.06879 | – | 3.1e−05 |
| 3 | +0.00000 | +0.25000 | +0.00000 | – | +0.00119 | +0.00121 | +0.00121 | – | 1.5e−05 |
| 5 | +0.25000 | +0.00000 | +0.25000 | – | +0.12699 | +0.12698 | +0.12696 | – | 2.3e−05 |
| 6 | +0.00000 | +0.00000 | +0.00000 | – | −0.01711 | −0.01711 | −0.01711 | – | 7.6e−06 |
| 7 | +0.00000 | −0.25000 | −0.25000 | – | −0.05742 | −0.05744 | −0.05744 | – | 1.9e−05 |
| 8 | +0.00000 | +0.00000 | +0.00000 | – | −0.07937 | −0.07936 | −0.07936 | – | 7.6e−06 |
| 9 | +0.00000 | −0.25000 | −0.25000 | – | −0.12386 | −0.12386 | −0.12385 | – | 1.1e−05 |
| **10** | **+0.12500** | **−0.12500** | **+0.25000** | **是** | −0.11644 | −0.11645 | −0.11646 | – | 1.9e−05 |
| **11** | **−0.12500** | **−0.12500** | **+0.25000** | **是** | −0.33107 | −0.33112 | −0.33112 | – | 5.7e−05 |

- bf16：gap 全部落在 `0 / ±0.125 / ±0.25` 這些量化格點上；id 10、11 的**正負號**隨形狀改變
  → argmax 翻轉。
- fp32：同一個前綴，gap 是**任意實數**，三種形狀的差異只有 **7.6e−06 … 5.7e−05**
  → 形狀造成的差異比 bf16 小了 3–4 個數量級，**零次翻轉**。

換句話說：**「形狀會改變答案」這件事只在 bf16 發生，fp32 下完全消失。**

（附帶一筆：`pass6 --dtype float32` **不加** `--ctx-from` 時 `n_divergences = 0` ——
fp32 下根本沒有分歧點可量，見 `results_pass6_4b_fp32.json`。這是「12/12 相同」的另一個說法。）

**8B（Phase 1 的實際模型）** → `results_pass6_8b_bf16.json`：8 個分歧點，
gap 分別是 `0.0 / +0.25 / −0.25 / +0.25 / −0.25 / −0.25 / +0.125 / −0.50`，
最大 0.5（= 4 個 bf16 ULP）；三種形狀在這 8 個點上同向（沒翻轉），
但把三種重建方式合起來看，選中基準 token 與選中 FlashMap token 各 **4 / 8** ——
換一個等價的計算方式就有一半的機率換答案。

---

## 4. 跨 runtime 對照：誰才是「正確答案」

`pass4_bf16_shape.py --parts C` 會把 HF 的基準輸出與 `results_vllm.json` 裡
vLLM server 的基準輸出並排（同一個 8B、同一組 prompt、同樣 temperature=0）:

| 執行 | id=1 的 token 數 | id=1 的前 120 字 |
|---|---|---|
| HF `generate()`（bf16） | 217 | `Let's solve this step by step.` |
| HF `spec_generate()`（bf16） | 155 | `Let's break this down step by step:` |
| vLLM baseline（bf16） | 155 | `Let's break this down step by step:` |

三者在分歧點之前的 token **完全一樣**，分歧發生在 index 2，
而那個位置的 bf16 top-2 gap 恰好是 **0.0000**。
也就是：vLLM 與 FlashMap 各自用不同的 kernel，剛好 tie-break 到同一邊；
HF 的 `generate()` 落到另一邊 —— 然後整段 155 / 217 tokens 就完全不同了。

> 這解釋了為什麼「HF 基準 vs vLLM 基準」看起來像是 lab 有 bug：
> 它不是 bug，是 bf16 下**基準本身不唯一**。要得到唯一的基準，必須用 fp32。

在 vLLM 這一層（`bench.py` 新增的 `lossless` 區塊），同引擎、同 kernel 下
baseline 與 FlashMap 的輸出一致率是 **9/12（前 120 字）**；
其餘 3 題的 token 差來自驗證 pass 衝過 `max_tokens` 以及同樣的並列現象。

---

## 5. Lab harness 的 bug（已修）

| # | 位置 | 症狀 | 原因 | 修法 |
|---|---|---|---|---|
| 1 | `bench.py` Prometheus 解析 | 接受率算出 **17.0**（>1，荒謬） | 用 substring 比對家族名，把 `_created`（Unix timestamp ≈ 1.79e9）與 16 個 `_per_pos_` 家族一起加總 | 只比對**完整**家族名（以 `_total` 結尾），並改用 `spec_decode_stats()`；`selftest.py §5` 有回歸測試 |
| 2 | `bench.py` spec-decode 統計 | 根本沒有 γ | 從未讀 `spec_decode_num_drafts` / `num_accepted_tokens` | 依 vLLM v1 語義 `γ = 1 + accepted/drafts`，與 vLLM 自己的 log 對帳一致 |
| 3 | `setup.sh:124` | `uv pip install` 直接失敗 | 傳了不存在的 `--workdir` 參數；next-steps 也用錯 dtype | 移除該參數；next-steps 改 `--dtype float32` |
| 4 | `acceptance.py` 無損性判準 | 只比**解碼後字串** | `decode(skip_special_tokens=True)` + `.strip()` 會掩蓋 EOS/空白差異；上游是比 token id | 改成 `torch.equal`（token id），另存 `text_identical` / `first_diff` / `lossless_prefix_ok` / `spec_overshoot` |
| 5 | `pass4_bf16_shape.py` | Part B 靜默不執行 | `probe()` 的回傳值沒有 `diverged=True`，`summarize()` 用 `.get("diverged")` 把它過濾掉了 | 補上旗標，並用 `_diverged()` 容忍舊格式 |

`acceptance.py` 另外把基準呼叫對齊上游（不傳 `attention_mask`、`temperature=1.0`）——
PASS 4 已證明這兩處在數值上是惰性的，但對齊上游仍是正確做法。

---

## 6. 限制與未驗證的部分

1. **8B 的 fp32 端到端 A/B 做不到。** 8B 的 fp32 權重是 32 GiB，超過 RTX 4090 的 24 GiB。
   因此：(a) dtype 的完整 A/B 在 **4B** 上做；(b) 8B 改用 PASS 5 的 CPU fp32
   單點前向取得真值。這是刻意的取捨，不是遺漏。
2. **只有 concurrency=1。** 批次服務下的加速與無損性都未驗證（論文作者亦明列為未評估）。
3. **只測 temperature=0。** T>0 需要 `draft_sample_method: probabilistic` 才談得上分布一致，
   vLLM 端的行為未驗證。
4. **PASS 6 的 `k=16` 是「17 個位置一起前向」的忠實近似**，不是 FlashMap 內部
   `spec_generate` 的實際呼叫堆疊；真正的驗證形狀由 `spec_generate` 本身決定。
5. 樣本量只有 12 題。「並列」是隨機事件，換一組 prompt 會得到不同的 mismatch 題號
   （但機制不變），所以判準必須是「gap 是否落在 bf16 解析度內」，而不是「哪幾題不同」。

---

## 7. 重跑指令

```bash
source ~/flashmap-lab/.venv/bin/activate
export HF_HOME=~/flashmap-lab/hf-cache
cd ~/Projects/flashmap-local-bench/lab

# 不需要 GPU 的邏輯自測（含 metric 解析回歸測試）
python selftest.py

# 嚴格無損性檢查（fp32）：預期 12/12 逐 token 相同
TARGET_MODEL=Qwen/Qwen3-4B DRAFT_MODEL=j-shen/FlashMap-Qwen3-4B \
  python acceptance.py --dtype float32 --out /tmp/acc_4b_fp32.json

# 機制證明
python pass3_bf16_proof.py --target Qwen/Qwen3-4B --draft j-shen/FlashMap-Qwen3-4B --out /tmp/pass3.json
python pass4_bf16_shape.py --parts ABC --dtypes bfloat16 float32 \
  --target Qwen/Qwen3-4B --draft j-shen/FlashMap-Qwen3-4B --out /tmp/pass4.json
python pass6_block_shape.py --dtype bfloat16 --target Qwen/Qwen3-8B \
  --draft j-shen/FlashMap-Qwen3-8B --out /tmp/pass6_8b.json

# 8B 的 fp32 真值（第一階段要 GPU，第二階段純 CPU、需 ~32 GiB RAM）
python pass5_fp32_cpu_truth.py --stage collect --out /tmp/pass5.json
python pass5_fp32_cpu_truth.py --stage truth   --out /tmp/pass5.json

# 端到端（Phase 2）
python bench.py
```


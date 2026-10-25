# FlashMap：Flow Map Speculative Decoding — 論文筆記與機制拆解

> 查核對象：`arXiv:2610.24365v1`（FlashMap: Flow Map Speculative Decoding，2026-10-17）、
> `github.com/al-jshen/flashmap`（MIT）、`j-shen/FlashMap-Qwen3-8B` / `-4B`（Apache-2.0）
> 查核方式：抓取 arXiv HTML 全文逐節比對、讀 GitHub README 與 `pyproject.toml`、查 Hugging Face models API
> 整理日期：2026-10-26
> 延伸閱讀：FMLM (Lee et al., 2026)、Flow Maps (Boffi et al., 2025a/b)、DFlash (Chen et al., 2026)、
> EAGLE-3 (Li et al., 2025)、DeepSpec、Discrete Flow Maps (Potaptchik et al., 2026)

---

## 一、一句話結論

FlashMap 是一個**推測解碼（speculative decoding）的 drafter**：用一個連續流（continuous flow）模型
的 flow map，在**一次前傳**內把整塊 K=16 個 token「整體」產生出來（保留 token 間相依性），
交給 target 一次前傳驗證，在 Qwen3-8B 上取得**最高 6.14× 的無損加速**（GSM8K, greedy）。

三個必須記住的限定：

1. **6.14× 是 18 個實驗格中的最高格，不是平均**。9 項 benchmark 平均：T=0 約 **4.36×**、T=1 約 **3.48×**；
   聊天類（MT-Bench / Alpaca / Arena-Hard）只有 **2.5～3.0×**。
2. **「無損」是數學保證**：推測解碼的接受規則讓輸出分布與 target 完全相同；greedy 下輸出逐字一致。
   FlashMap 只改「怎麼猜」，不改「誰做決定」。
3. **只發布了 Qwen3-4B / 8B 兩個 target 的 drafter**，且量測是 concurrency=1、單卡（H200 NVL）、
   thinking 關閉。批次服務與更大模型**都還沒評估**。

---

## 二、FlashMap 在解決什麼問題

### 2.1 推測解碼的成本模型

每輪：drafter 提出 K 個 token `ŷ₁:K`，target 一次前傳算完 K 個位置的 logits，
用 Leviathan et al. (2023) 的接受規則**由左至右**接受：第一個被拒的位置由 target 從殘餘分布重抽，
其後整塊丟棄。該規則保證輸出分布與 target 的 `p` **完全相同**（這就是「無損」的來源）。
T=0 時規則退化成「保留與 target argmax 最長相符的前綴」。

- **acceptance length γ ∈ [1, K+1]**：每輪實際提交的 token 數（含 target 自己貢獻的那 1 個）。
- **每 token 延遲 = (T_draft + T_verify) / γ**。
  `T_verify` 大致固定，因為 decode 是 memory-bound（受權重讀取頻寬限制，非算力）。

所以加速有兩個槓桿：**γ 越大越好**、**T_draft 越小越好**。EAGLE-3 贏前者的尾巴，DFlash 贏後者，
FlashMap 想兩者兼得。

### 2.2 DFlash 的 factorization error

DFlash 用 block discrete diffusion：attend target 多層 hidden states，**一次前傳**把 fully masked 的整塊
去噪完（所以 T_draft 與 K 無關）。token embedding 與 LM head 直接複製 target 並凍結，只訓練幾層 transformer。

問題出在「離散 diffusion 每一步都在取樣 token」。若要一步取多個 token，需要 `V^K` 大小的 joint 分布，
指數級不可行 → 只能退到 **factorized approximation**：

```
q(y₁:K | c) = Πₖ qₖ(yₖ | c)        (Eq. 1)
```

per-slot cross-entropy 的最優解就是 target 的**邊際分布** `qₖ = p(yₖ|c)`，
而剩下的差距是 **total correlation** `KL( p(y₁:K|c) ‖ Πₖ p(yₖ|c) )`——**再多的模型容量也消不掉**。

論文用一個經典例子說明（Figure 1(b)）：若後續是 `"new york"` 或 `"san diego"` 各半，
factorized drafter 有一半的機率生出 `"new diego"` 或 `"san york"`——每個 slot 單看都合理，合起來是廢話。
這正是「前後不連貫 → 後半段被拒 → γ 變小」的根源。

### 2.3 Flow over one-hot tokens（FMLM）

Flow 的隨機性**只有初始噪聲**一處，一條確定性軌跡把整塊一起搬到資料上：

```
I_t = (1-t)·x₀ + t·x₁ ,  x₀ ~ N(0, I)          (Eq. 2)
```

序列用 one-hot 編碼 `x₁ ∈ R^{L×|V|}` 表示，最後 argmax 解碼。
FMLM 選擇**預測 clean data 而非 velocity**（clean data 低秩，velocity 含雜訊而全秩）：

```
v_t(x) = ( D_t(x) - x ) / (1-t)                  (Eq. 3)
```

denoiser 用 softmax 參數化、以 cross-entropy 訓練，唯一最小解就是真 denoiser（Eq. 4）。
因為 one-hot 向量是 simplex 的頂點，真 denoiser `D_t(x)^l = p(y_l=· | I_t=x)` 恆落在 simplex 上。

直接用 flow 起草需要數值積分，**O(100) 次前傳**，太貴。

### 2.4 Flow map

Flow map `X_{s,t}` 是 flow 這個 ODE 的**解算子**：把時間 s 的狀態送到時間 t 的狀態。
兩個性質：

- `X_{s,s} = id`
- **semigroup**：`X_{u,t} ∘ X_{s,u} = X_{s,t}`（跳 s→u→t 等於直接跳 s→t）

FMLM 用 two-time denoiser 參數化：

```
X_{s,t}(x) = ((1-t)/(1-s))·x + ((t-s)/(1-s))·δ_{s,t}(x)     (Eq. 5)
```

對角線上退化成 denoiser（`δ_{s,s} = D_s`）。**單次評估** `X_{0,1}(x₀) = δ_{0,1}(x₀)` 就把噪聲映射成資料，
而且因為 map 作用在**整塊**上，它能表達 token 間相依性——這就是 FlashMap 的立足點。

> **關鍵洞見（§6.2 shuffle test 的結論）**：joint structure 不是「模型本身自帶」的，而是
> **積分過程**產生的。teacher flow 在 1 NFE 時 Δ=0.00（完全沒有 joint structure），
> 積分 16 步後 Δ 才長出來。所以「一次前傳就有整體一致性」這句話**只對被蒸餾後的 2 NFE 版本大致成立**。

---

## 三、FlashMap 的三個關鍵設計

### 3.1 架構：conditional DiT

- backbone：conditional diffusion transformer（Peebles & Xie 2023）。
  每個 block = **slot 間 self-attention** + **對 target context 的 cross-attention** + MLP，
  三者各自受 **adaLN-Zero** 以時間 `t` 調控。
- context = target 第 **1, 9, 17, 25, 33** 層的 hidden states，在 anchor 位置拼接後線性投影，
  再加上 anchor token `y_a` 的 embedding。
- **詞表投影（輸入 E、輸出 LM head）複製自 target 並凍結** → 每個 slot 做 softmax 就得到 `D^θ_t`。
- Qwen3-8B target 的設定：**6 個 DiT block、d=2048、16 heads、QK-norm、GELU**，
  可訓練參數 **500M**（DFlash 1049M、EAGLE-3 302M）。

### 3.2 訓練目標

在 target 生成的回應上取樣、cache target features、隨機取 anchor 位置（同 DFlash）。
每個 block 有**自己的**時間 `t` 與噪聲 `x₀`：

```
I_t = (1-t)·x₀ + t·onehot(y_{a+1:a+K})
```

slot loss 以指數衰減權重加權：`w_k = e^{-(k-1)/κ}`，**κ = 7**——前面的 slot 出錯會讓後面全部作廢。

### 3.3 輸入特徵化（Appendix A）

直覺做法是 `x^k E`（把 state 過 target 的 embedding 矩陣）。但 **clean token 只貢獻一項，其餘全是雜訊**，
訊號會被淹沒。改成用 noisy state 下 clean token 的**後驗分布**：

```
π_t(x^k) = softmax( t·x^k / (1-t)² )
h^k = RMSNorm( π_t(x^k) · E )
```

### 3.4 時間重參數化

詞表 151,936 太大，state 只在 `t→1` 的窄窗內才揭露 token；均勻抽 t 會讓大部分訓練花在
「什麼都沒揭露」或「已完全揭露」的時間點。改用**解碼誤差**當時間軸：

```
P_e(t) = 1 - E_{z~N(0,1)}[ Φ(z + t/(1-t))^{|V|-1} ]          (Eq. 6)
τ(t)   = 1 - P_e(t)/P_e(0)      ← 已消除的初始誤差比例
```

訓練抽 `τ ~ U[0,1]` 再反解 t；推論用網格 `t_n = τ^{-1}(n/N)`。

### 3.5 Flow Map Distillation：semigroup distillation

把 teacher（flow）蒸餾成 student（two-time denoiser `δ^φ_{s,t}`）。
訓練目標直接來自 semigroup 性質——「s→t 必須落在 s→u→t 的同一點」：

```
δ̄_{s,t} = sg( λ·δ^φ_{s,u}(I_s) + (1-λ)·δ^φ_{u,t}( X^φ_{s,u}(I_s) ) )
λ = (1-t)(u-s) / ((1-u)(t-s))                                   (Eq. 7)
```

- **s = t（零長度跳躍）**：目標是 teacher 的 `D_s(I_s)` → 強制 `δ_{s,s} = D_s`（tangent condition），
  把 student 錨在 teacher 上。
- **s < t**：目標由 student **自己**在中點分割產生（stop-gradient）→ 從短跳躍自舉到長跳躍。
- 訓練時 **1/16 的跳躍是 full jump (0,1)**；一半的 block 訓練零長度跳躍。

### 3.6 快速多步驟起草（Appendix B）

多跳躍時每個 slot 都要算詞表大小的 state（`x₀E`、`π_u(x_u)E`），是 O(|V|) 的昂貴操作。
優化：把每個 token 的權重拆成「只有噪聲的部分」與「top-|Tᵏ| 的修正項」；
前者只依賴 `x₀` 與 `α = u/(1-u)`，可**預先為 8,192 個 noise draws 預計算**，
起草時每 slot 隨機抽一筆 bank，只需重抽 top-16 的噪聲。
→ 第二次跳躍相對一跳躍的**額外成本降低約 70%**（Qwen3-8B），接受長度幾乎不變。

---


## 四、與 DFlash / EAGLE-3 的差異

### 4.1 三者定位

| | EAGLE-3 | DFlash | FlashMap |
|---|---|---|---|
| drafter 的生成模型 | 自回歸（逐 token 預測分布） | block discrete diffusion | **continuous flow + flow map** |
| 產出 K 個 token 的 NFE | K（或 draft tree） | 1 | **1**（或 2） |
| T_draft 是否與 K 有關 | **是**（線性成長） | 否 | 否 |
| slot 間相依性 | 有（自回歸天然自洽） | **無**（factorized） | **有**（map 作用於整塊） |
| 接受長度 γ | 最高（尤其 T=1，6/9 格第一） | 中 | 接近 EAGLE-3 |
| 總加速 | 最差（延遲高） | 次佳 | **最高（18/18 格）** |
| Qwen3-8B 參數 | 302M | 1049M | 500M |

一句話：**EAGLE-3 拿「猜得準」換「猜得慢」；DFlash 拿「猜得快」換「猜不連貫」；
FlashMap 試圖兩者兼得**——它繼承 DFlash 的 target-conditioning 與單次前傳，
再把生成模型換成能表達 joint 分布的 flow map。

### 4.2 FlashMap 真正贏在哪裡

1. **速度上全面勝出**：Qwen3-8B 上 1 NFE 在**全部 18 格**（9 benchmark × 2 溫度）都是最快，
   比同設定重訓的 DFlash 平均快 **7.4%**。
2. **不是靠模型小**：FlashMap(1 NFE) 只有 DFlash 一半參數，接受長度卻在 **18 格中 15 格高於 DFlash**。
   → 贏的是演算法，不是延遲優勢。
3. **2 NFE 接受長度最高但加速反而略低**：多一次 drafter 前傳的延遲 > 多接受 0.6 個 token 的收益。
   實務上：互動式用 1 step，要長篇數學推論再試 2 step。
4. **conditional acceptance 曲線形狀不同**（§6.1）：
   - FlashMap **1 NFE**：第 1 個 slot 全面勝過 DFlash，但後續 slot 的條件接受率**衰減較快**。
   - FlashMap **2 NFE**：**條件接受率在整塊內不衰減，甚至上升**，三個領域皆然。
   - 這意味著 2 NFE 可能在更長的 block（>16）上還能再賺，但需要用更大 K 重訓才能驗證。
5. **1 NFE 為何仍比 DFlash 整體快**：早期 slot 的收益會被「所有走到該 slot 的 block」放大，
   晚期 slot 的損失只影響少數 block——前者權重遠大於後者。

### 4.3 Shuffle test：把「邊際」與「相依性」拆開

做法：把同一份 draft 的各 slot 換成**獨立噪聲**（`shuffled`），保留每個 slot 的邊際分布、
只抽掉 token 間相依性；Δ = γ_shared − γ_shuffled 即為 joint structure 帶來的接受長度增益。
DFlash 這類與噪聲無關的 drafter 結構上 Δ ≡ 0。

**Qwen3-8B, T=0（accepted draft tokens per block）**

| Drafter | Math Δ | Code Δ | Chat Δ |
|---|---|---|---|
| DFlash | 0.00 | 0.00 | 0.00 |
| Flow（1 step） | 0.00 | 0.00 | 0.00 |
| Flow（16 steps，teacher） | **0.82** | **0.59** | **0.29** |
| FlashMap（1 step） | 0.32 | 0.13 | 0.08 |
| FlashMap（2 steps） | 0.78 | 0.47 | 0.25 |

三個結論：

1. **積分才製造 joint structure**：teacher flow 在 1 NFE 時 Δ=0.00，積分 16 步後 Δ=0.82。
2. **1 NFE 只保留 teacher 的 ~39%**（0.32/0.82）；**2 NFE 保留 ~95%**（0.78/0.82）。
3. 因此「一次 forward pass 就有前後一致性」在 1 NFE **只部分成立**；媒體描述略嫌樂觀。

### 4.4 文獻定位與作者坦承的不足

FlashMap 建立在三條線上：FMLM（連續流的語言模型）、Flow Maps（解算子蒸餾 / semigroup）、
DFlash（block diffusion drafter 的設計）。作者群在 §6.3 坦承 1 NFE 的 flow map 遠落後
16 NFE 的 teacher（連 shuffled 設定也落後 → **per-slot 邊際也估得比較差**），
並引用 Potaptchik et al. (2026)、Tang & Wang (2026) 指出：one-step flow map 模型
有 generative perplexity 過高或 entropy collapse 的已知問題——即 §3.5 的蒸餾策略仍有很大改進空間。

---


## 五、完整數據

### 5.1 Qwen3-8B 加速倍率（Figure 1(c)）

**T = 0（greedy）**

| 方法 | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|
| EAGLE-3 (16) | 3.20 | 3.09 | 2.82 | 2.57 | 2.87 | 2.90 | 1.80 | 1.78 | 1.92 |
| DFlash | 5.77 | 5.59 | 4.67 | 3.74 | 4.50 | 4.88 | 2.48 | 2.34 | 2.67 |
| **FlashMap (1 step)** | **6.14** | **5.89** | **4.89** | **4.02** | **4.73** | **5.26** | **2.73** | **2.56** | **3.04** |
| FlashMap (2 steps) | 5.89 | 5.57 | 4.61 | 3.79 | 4.53 | 5.06 | 2.47 | 2.33 | 2.76 |

**T = 1（採樣）**

| 方法 | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|
| EAGLE-3 (16) | 2.94 | 2.69 | 2.24 | 2.37 | 2.53 | 2.51 | 1.66 | 1.57 | 1.55 |
| DFlash | 4.89 | 4.24 | 3.23 | 3.42 | 3.82 | 3.54 | 2.24 | 2.06 | 2.05 |
| **FlashMap (1 step)** | **5.12** | **4.39** | **3.56** | **3.47** | **3.95** | **3.85** | **2.44** | **2.28** | **2.30** |
| FlashMap (2 steps) | 5.07 | 4.15 | 3.28 | 3.43 | 3.86 | 3.57 | 2.29 | 2.14 | 2.14 |

### 5.2 Qwen3-8B acceptance length γ（Table 1）

**T = 0**

| 方法 | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|
| EAGLE-3 (16) | 6.75 | 6.43 | 5.96 | 5.38 | 6.04 | 6.16 | 3.75 | 3.71 | 3.97 |
| DFlash | 7.43 | 7.09 | 5.92 | 4.77 | 5.74 | 6.35 | 3.14 | 2.96 | 3.38 |
| FlashMap (1 step) | 7.69 | 7.24 | 6.00 | 4.95 | 5.82 | 6.65 | 3.34 | 3.14 | 3.71 |
| FlashMap (2 steps) | **8.31** | **7.69** | **6.36** | **5.24** | **6.26** | **7.19** | **3.40** | **3.20** | **3.80** |

**T = 1**

| 方法 | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|
| EAGLE-3 (16) | 6.15 | 5.55 | 4.61 | 4.94 | 5.29 | 5.37 | **3.43** | **3.25** | **3.20** |
| DFlash | 6.30 | 5.40 | 4.10 | 4.37 | 4.88 | 4.62 | 2.85 | 2.62 | 2.61 |
| FlashMap (1 step) | 6.40 | 5.40 | 4.37 | 4.29 | 4.87 | 4.90 | 3.00 | 2.81 | 2.83 |
| FlashMap (2 steps) | **7.11** | **5.74** | 4.52 | **4.75** | **5.35** | **5.06** | 3.16 | 2.96 | 2.95 |

### 5.3 平均值（由上表自行計算）

| 設定 | FlashMap(1) 平均 | DFlash 平均 | 差 |
|---|---|---|---|
| Qwen3-8B, T=0, 全部 9 項 | **4.36×** | 4.07× | +7.2% |
| Qwen3-8B, T=1, 全部 9 項 | **3.48×** | 3.28× | +6.3% |
| Qwen3-8B, T=0, 僅聊天 3 項 | 2.78× | 2.50× | +11.2% |
| Qwen3-8B, T=0, 僅數學 3 項 | 5.64× | 5.34× | +5.5% |

（論文自述「比 DFlash 平均快 7.4%」，與上表由四捨五入後數字算出的 7.2% 吻合。）

### 5.4 Qwen3-4B：FlashMap 並沒有全面勝出（Appendix C, Table 3）

**T = 0 加速倍率**

| 方法 | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|
| EAGLE-3 (16) | 2.66 | 2.64 | 2.47 | 2.20 | 2.33 | 2.44 | 1.49 | 1.37 | 1.64 |
| DFlash | **5.18** | **5.23** | **4.56** | 3.72 | 4.00 | 4.27 | 2.43 | 2.28 | 2.58 |
| FlashMap (1 step) | 5.14 | 5.02 | 4.47 | 3.71 | **4.04** | **4.28** | **2.46** | **2.28** | **2.66** |
| FlashMap (2 steps) | 4.75 | 4.57 | 4.00 | 3.42 | 3.76 | 3.87 | 2.16 | 2.00 | 2.34 |

**這是摘要沒有強調、但很重要的一點**：摘要宣稱的「比 DFlash 快」只對 **Qwen3-8B** 成立。
在 **Qwen3-4B** 上 DFlash 在三個數學項全面領先；FlashMap(1) 的 9 項平均是 **3.78× vs DFlash 3.81×**——
**打成平手且略輸**。FlashMap 的優勢與 target 規模相關，並非與 target 無關的通用改進。

---

## 六、限制、成熟度與注意事項

### 6.1 論文自列的 Limitations

- 只評估**同一家族的兩個 target**（Qwen3-4B / 8B），其他家族與更大的 target 留待未來。
- 訓練資料只用了 Open-PerfectBlend 的 **500K 樣本子集** → 絕對接受長度可能低於用完整資料訓練者。
- 加速在 **concurrency=1、單卡**下量測，**未評估批次服務**。
- 驗證預算固定 K=16（EAGLE-3 以 TTT=7、top-k=10、tree size=16 對齊）。
- 未評估 DFlash 後續工作（draft trees、autoregressive postprocessing、low-rank latent mixtures），
  但 FlashMap 與 draft tree **相容**（Appendix D）。
- 基準涵蓋英文 math/code/chat，T=0 與 T=1。
- **可能訓練不足**：作者 train 39K + distill 23K steps，Lee et al. (2026) 是 1M + 100K steps。

### 6.2 訓練與量測成本（Appendix G）

- 訓練：**4 個節點 × 8×H200 SXM（共 32 GPU）**。
  Qwen3-8B flow teacher ≈ 25 小時、蒸餾 ≈ 9 小時；DFlash 重訓 ≈ 27 小時；EAGLE-3 重訓 ≈ 21 小時。
  **全部實驗合計約 4,600 GPU-hours。**
- 量測：主結果在 **NVIDIA H200 NVL**；multi-sample 表（Table 4）在 1×H100 SXM。
- 軟體：DeepSpec（`005e03b`）+ SGLang 0.5.20、PyTorch、concurrency 1、max new tokens 2048、thinking 關閉。

### 6.3 成熟度與授權

| 項目 | 狀態 |
|---|---|
| 論文 | arXiv:2610.24365v1，2026-10-17（CC BY 4.0） |
| 程式碼 | github.com/al-jshen/flashmap，**MIT**，41 commits |
| 模型權重 | `j-shen/FlashMap-Qwen3-8B`（500,111,361 參數 bf16，~1.0 GB）、`-4B`（474,940,929），**Apache-2.0** |
| 社群 | 發布約一週：37 stars / 3 forks / 672 downloads（8B）→ 數字應視為 preliminary |
| 框架整合 | vLLM `>=0.31,<0.32` general plugin；SGLang `==0.5.20` plugin；另有 transformers 參考實作 |

### 6.4 其他注意事項

- **T>0 時必須用 probabilistic draft sampling**（vLLM 的 `draft_sample_method: "probabilistic"`）才保證分布一致。
- drafter 需要 target 第 1/9/17/25/33 層 hidden states → 對 target 架構有相依性，換家族需重訓。
- 倫理面：lossless 代表 target 的偏見／有害／錯誤輸出風險原樣保留；加速也降低大規模濫用的成本。

---


## 七、本機部署評估（RTX 4090 24 GB / 64 GB RAM / i7-13700K / Windows）

> 環境快照（2026-10-26）：RTX 4090 24 GB（~1008 GB/s）、64 GB RAM、i7-13700K 16C/24T、
> Windows、Python **3.15.1**、`uv` 已裝、CUDA Toolkit 目錄存在；
> **torch / transformers / vllm / sglang 全部未安裝**。

### 7.1 三個實際阻礙

1. **vLLM 0.31.x 與 SGLang 0.5.20 只有 Linux wheel**，Windows 原生無法使用。
   WSL2 功能可用但**未安裝任何 distro**（`wsl -l -v` 顯示無發行版）。
2. **Python 3.15.1 太新**：vLLM/SGLang wheel 目標 ≤3.13，官方 README 建議 **3.12**。
3. FlashMap 套件本身在 Windows 上可裝（`requires-python >= 3.10`、`Operating System :: OS Independent`），
   有 transformers 參考實作，但速度慢，只適合驗證正確性。

### 7.2 VRAM 試算（可行但緊）

| 項目 | 大小 |
|---|---|
| Qwen3-8B bf16 權重 | ~16.4 GB |
| FlashMap-Qwen3-8B drafter（bf16） | ~1.0 GB |
| 合計 | **~17.4 GB / 24 GB** |
| 剩餘給 KV cache + activation | ~6 GB |

Qwen3-8B（GQA、36 層、8 KV heads、head_dim 128）KV ≈ **147 KB/token** → 約可放 **40K context**。
必要時開 FP8 KV cache。

### 7.3 加速比能否外推到 4090？

判斷：**大致可以**，因為加速比主要取決於 γ 與相對成本，而非硬體：

- 每輪記憶體讀取 ≈ 16.4 GB（target 驗證）+ 1.0 GB（drafter）= **17.4 GB**；
  AR 每 token 讀 16.4 GB → 理論加速 ≈ **γ × 0.94**，與硬體無關（前提：memory-bound 且算力有餘裕）。
- 4090 有 1008 GB/s；16-token 驗證的一次前傳約 256 GFLOP，遠低於其算力上限 → 仍是 memory-bound，條件成立。
- 實測校準：GSM8K 上 6.14 / γ=7.69 = **0.80** → 每輪約 20% 框架開銷。
- **估算（未實測）**：AR Qwen3-8B 約 40–55 tok/s → FlashMap 在數學/程式約 **150–250 tok/s**、
  聊天約 **100–150 tok/s**。4090 上 Python/scheduler 固定開銷占比更高，實際倍數可能略低於 H200 NVL。

### 7.4 「本地跑 Opus 等級」這句話要修正

FlashMap **只降延遲、不改能力上限**。要拿到接近 Opus 的品質，現實路線是**量化 MoE**
（例如 Qwen3-30B-A3B，Q4_K_M ≈ 17 GB 可塞進 24 GB；llama.cpp GPU offload 下 decode 只讀
active experts，約 300–500 tok/s）。但目前**沒有對應 30B target 的 FlashMap drafter**，
兩條路線暫時無法疊加。

FlashMap 真正的價值是：**讓一個跑得動的小模型（8B）互動體感像大模型**。
Windows 原生可行的折衷是 llama.cpp 內建的 speculative decoding
（prompt lookup / draft model / self-speculative），不需要 FlashMap 這套連續流基礎設施。
### 7.5 實作環境：本機已裝 Omarchy（Arch Linux）

**2026-10-26 更新**：使用者已在本機安裝 **Omarchy**（Arch 系），因此 §7.1 的三個阻礙
全部解除 —— 不需要 WSL2、不需要 Windows 原生 wheel、直接就是原生 Linux：

| 原阻礙 | Omarchy 下的解法 |
|---|---|
| vLLM/SGLang 只有 Linux wheel | Arch 本來就是 Linux；vLLM 0.31.0 是 `manylinux_2_28_x86_64` wheel，Arch 的 glibc 相容 |
| Python 3.15.1 太新 | 用 `uv venv --python 3.12` 建獨立環境，不動 Arch 系統 Python（Arch 是 rolling release，系統 Python 常比 wheel 支援的還新） |
| WSL2 未裝 distro | 不需要了，直接開機選單選 Omarchy |

**版本綁定（2026-10-26 由 PyPI API 查證）**：

- `flashmap` 0.1.0（2026-10-16 上傳）：`requires-python >=3.10`，py3-none-any wheel，MIT。
- `vllm`：**只有 `0.31.0`** 符合 flashmap 的 `>=0.31,<0.32`；鎖 `torch==2.13.0`、`transformers>=5.10.4,<5.18.0`、`flashinfer-python==0.7.0.post1`。
- `sglang` 0.5.20（flashmap 的 `[sglang]` extra 鎖定）：cp310–cp313 manylinux_2_34 wheel，鎖 `torch==2.13.0`、`transformers==5.12.1`。
- 兩條路徑的 torch 綁定一致（2.13.0），與 flashmap 官方測試矩陣（torch 2.9 / 2.13）相容。
- 本機 Windows 側驅動 621.01 / CUDA UMD 13.5 → 可支援 cu13 的 torch wheel；Omarchy 側驅動需相當或更新。

**已備妥的實驗腳本包**：`lab/`（重開機後從 Windows 分割區掛載取用）

| 檔案 | 用途 |
|---|---|
| `setup.sh` | 冪等建置：檢查 → 裝 uv → py3.12 venv → `vllm==0.31.0` + `flashmap[vllm]` → CUDA smoke → 下載 17.5 GB 模型 |
| `run_all.sh` | 一鍵：Phase 1（γ + 無損性）→ Phase 2（端到端 tok/s） |
| `acceptance.py` | 用 transformers 參考實作量 γ，並逐字比對 target greedy 輸出以驗證無損 |
| `bench.py` | vLLM 兩個 server（基準 vs FlashMap）同 prompt 集比較，抓 `/metrics` 的 spec-decode counter |
| `selftest.py` | 不需 GPU 的邏輯自測（本機已驗證通過） |
| `prompts.json` | 12 題固定測試集（數學 5 / 程式 4 / 聊天 3） |

量測條件刻意對齊論文：**concurrency=1、temperature=0（greedy）、thinking off**，
兩個 server 除 `--speculative-config` 外參數完全相同，避免不公平比較。

### 7.6 Windows 原生可行性（2026-10-26 實測查證）

| 階段 | Windows 原生 | 證據 |
|---|---|---|
| Phase 1：接受長度 γ + 無損性 | ✅ **可以** | 已在本機實跑 `setup.ps1`：torch 2.14.1+cu130、CUDA runtime 13.0、RTX 4090 sm_89、bf16 matmul 通過 |
| Phase 2：端到端 tok/s | ❌ **不行** | vLLM 0.31.0 / SGLang 0.5.20 在 PyPI **只有 manylinux wheel，無 `win_amd64`**（PyPI API 逐一列出檔名確認） |

**為什麼 Phase 1 可以**：`flashmap` wheel 拆成兩塊——

- `flashmap/modeling_flashmap.py`（核心參考實作）：只 import `math / warnings / contextlib / dataclasses / typing / torch / torch.nn.functional`，
  **無 triton、無 torch.compile、無 flash-attn** → 純 PyTorch，跨平台。
- `flashmap/vllm/*`、`flashmap/sglang/*`（server 外掛）：含 triton kernel 與 `torch.compile` → Linux only。

而 `flashmap/__init__.py` 只註冊 `AutoConfig`/`AutoModel`，**不會在匯入時拉進外掛** → Windows 上 `import flashmap` 安全。

**Windows 專屬的三個坑（已寫進 `lab/setup.ps1`）**：

1. **PyPI 的 `torch` 在 Windows 上是 CPU-only 版**（實測裝到 `2.14.1+cpu`，`cuda.is_available()` 為 False）。
   CUDA 版在 PyTorch 自己的 index：`--index-url https://download.pytorch.org/whl/cu130`（`2.14.1+cu130-cp312-win_amd64`）。
   必須**先 `uv pip uninstall torch`** 再裝，否則 uv 認為需求已滿足而沿用 CPU 版。
   Linux 相反：PyPI 的 torch 預設就是 CUDA build，不要加 `--index-url cpu`。
2. **PowerShell 5.1 讀無 BOM 的 UTF-8 `.ps1` 會解碼錯亂** → 中文內容導致語法錯誤（"字串缺少終止符"）。
   所有 `.ps1` 必須存成 **UTF-8 with BOM**。
3. **PowerShell 5.1 把字串傳給原生程式時會吃掉內嵌雙引號** → `python -c "... getattr(m, "__version__", ...) ..."` 會變成
   `NameError: name '__version__' is not defined`。解法：python 片段寫成獨立檔案（`lab/env_check.py`）再呼叫。
   另外原生工具的進度輸出會寫到 stderr → `$ErrorActionPreference` 必須用 `'Continue'` 搭配明確檢查 `$LASTEXITCODE`，
   用 `'Stop'` 會讓腳本在第一個進度輸出就中止。

### 7.7 本機實測結論：bf16 的「假性 MISMATCH」

> 完整證據鏈見 repo 的 [`lab/FINDINGS.md`](https://github.com/kalpakjian/flashmap-local-bench/blob/main/lab/FINDINGS.md)，
> 白話版見 [`lab/FINDINGS-easy.md`](https://github.com/kalpakjian/flashmap-local-bench/blob/main/lab/FINDINGS-easy.md)。

Phase 1 在 bf16 下回報 8/12 不一致，一開始看起來像演算法有 bug。**實際結論：演算法沒有問題，是量測判準錯了。**

| 問題 | 答案 |
|---|---|
| FlashMap 演算法有 bug 嗎 | **沒有**。PASS 3 的 instrumentation 自我檢查證明：FlashMap 送出的每一個 token 都等於它自己那一輪的 `argmax`（9/9 題通過，單題最多 479 token 全中） |
| 為什麼 bf16 下不一致 | 分歧點的 top-2 logit gap 全部落在 bf16 的量化格點上（`0 / ±0.125 / ±0.25 / ±0.5`），4B 有 **6/9 筆 gap 恰為 0.0**（位元完全相同的並列）→ `argmax` 由索引順序決定，任何實作細節都可能翻轉 |
| fp32 下呢 | 同一組程式、同一組 prompt、只改 dtype：**bf16 3/12 相同，fp32 12/12 相同**；γ 幾乎不變（6.00 vs 6.11）→ **加速效果與 dtype 無關，只有「逐位元相同」對 dtype 敏感** |
| 「HF 基準 217 tokens vs vLLM 基準 155 tokens」是誰錯了 | **都不是錯**。分歧發生在 index 2，該處 bf16 gap 恰為 0.0，vLLM 與 FlashMap 各自 tie-break 到同一邊，HF 的 `generate()` 落到另一邊 → **bf16 下基準本身不唯一** |
| 是 prompt / chat template 的問題嗎 | 不是。HF 與 vLLM 的 prompt token 完全一致；4 種 `attention_mask`/`temperature` 簽章在 bf16 下 12/12 輸出 token id 相同 |
| 前向形狀會翻轉答案嗎 | 會——**但只在 bf16**。把前向形狀從 1 個位置改成 17 個位置，bf16 的 gap 會翻轉（id 10/11 符號相反）；fp32 三種形狀的差異只有 7.6e−06~5.7e−05，**零次翻轉** |

**這對「無損性」這個詞的意義**：論文保證的是**演算法層無損**（輸出分布與 target 相同，greedy 下每個 token 都是 argmax），
**不是**「兩種不同形狀的前向在 bf16 下捨入到同一個值」。上游 `examples/benchmark.py` 自己也印了這段警語並建議
用 `--dtype float32` 做 exact check——而我原本的 `acceptance.py` 把它當成演算法無損來判定。

**VRAM 的硬限制**：8B 的 fp32 權重是 32 GiB，**超過 RTX 4090 的 24 GiB**，所以完整的 dtype A/B 只能在 **4B** 對照組做；
8B 改以 `pass5_fp32_cpu_truth.py` 在 CPU 上取單點真值（32 GiB RAM 放得下）。

**要修正的既有認知**：§2.1 我寫的「greedy 下規則退化成保留與 target argmax 最長相符的前綴」在**演算法層**正確，
但實作層還多了一層 bf16 解析度的干擾；§7.2 的 VRAM 試算（17.4 GB）則沒有考慮 fp32 嚴格檢查需要 2 倍權重。



---

## 八、速記卡（11 條）

1. FlashMap = **flow map 版 drafter**：一次前傳產出整塊 K=16 token，保留 token 間相依性。
2. 「無損」的來源是**推測解碼的接受規則**，不是 FlashMap 本身；greedy 下**演算法層**每個 token 都是 target 的 argmax。
   （實作層要「逐位元相同」必須用 fp32，bf16 的近似並列會造成假性 mismatch → §7.7）
3. **DFlash 的病根是 factorization error**：只剩 per-slot 邊際，差距是 total correlation，容量救不了。
4. **FlashMap 的病根是 flow map 的 1 NFE 蒸餾不夠好**：只保留 teacher 約 39% 的 joint structure。
5. **joint structure 是「積分」產生的**，teacher flow 在 1 NFE 時 Δ=0.00。
6. **1 NFE 加速最高、2 NFE 接受長度最高**；實務上先跑 1 step。
7. **6.14× 是單格最高值**；9 項平均 4.36×（T=0）、聊天只有 ~2.78×。
8. **在 Qwen3-4B 上 FlashMap 沒有全面贏**（平均 3.78× vs DFlash 3.81×）——摘要只講 8B。
9. **量測條件**：H200 NVL、concurrency 1、SGLang 0.5.20、thinking off；批次服務未評估。
10. 目前只有 **Qwen3-4B / 8B** 的 drafter；權重 Apache-2.0、程式碼 MIT、論文 CC BY 4.0。
11. **Windows 只能跑 Phase 1**（接受長度＋無損性），端到端 tok/s 需要 vLLM → 只有 Linux wheel → 走 WSL2 或 Omarchy。

---

## 九、參考連結

| 用途 | 連結 |
|---|---|
| 論文 | https://arxiv.org/abs/2610.24365 （v1, 2026-10-17, CC BY 4.0） |
| 專案頁 | https://jshen.net/flashmap |
| 程式碼（MIT） | https://github.com/al-jshen/flashmap |
| Drafter（8B, Apache-2.0） | https://huggingface.co/j-shen/FlashMap-Qwen3-8B |
| Drafter（4B, Apache-2.0） | https://huggingface.co/j-shen/FlashMap-Qwen3-4B |
| FMLM（Lee et al., 2026） | arXiv:2511.13720 |
| Flow Maps / semigroup distillation（Boffi et al., 2025b） | arXiv:2505.18825 |
| DFlash（Chen et al., 2026） | arXiv:2602.06036 |
| 推測解碼接受規則（Leviathan et al., 2023） | arXiv:2211.17192 |
| DeepSpec（drafter 實作框架） | https://github.com/deepseek-ai/DeepSpec |

### 本地留存的原始資料

- `paper/flashmap_paper.txt` — 論文字元版全文（61 KB，剝除 HTML 後；CC BY 4.0）
- arXiv HTML 原始檔未收錄（480 KB 網頁標記），可自 https://arxiv.org/abs/2610.24365 取得
- 專案頁原始 HTML 未收錄，可自 https://jshen.net/flashmap 取得

---

*本筆記的每個數字都取自 arXiv HTML 全文對應章節與表格；「估算」段落已明確標註為未實測。*

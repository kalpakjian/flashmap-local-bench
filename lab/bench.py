#!/usr/bin/env python3
"""
FlashMap 實驗 Part 2：端到端吞吐量（vLLM）

依次啟動兩個 server，跑同一組 prompt，比較：
  baseline : vllm serve Qwen/Qwen3-8B
  flashmap : 同上 + --speculative-config '{"model":"j-shen/FlashMap-Qwen3-8B","num_speculative_tokens":16}'

除 speculative-config 外，兩個 server 的參數完全相同（避免不公平比較）。
量測 concurrency=1、temperature=0、greedy，對齊論文的量測條件。

用法：python bench.py --prompts prompts.json --max-tokens 512
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TARGET_MODEL = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "j-shen/FlashMap-Qwen3-8B")
PORT = int(os.environ.get("BENCH_PORT", "8000"))
LOGDIR = Path(os.environ.get("BENCH_LOGDIR", "."))


def server_env():
    env = os.environ.copy()
    # FlashInfer 需要 nvcc 才能 JIT 編譯 sampling kernel；沒有 CUDA Toolkit 時退回 PyTorch sampler
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def start_server(tag, extra_args=None, port=PORT, max_model_len=8192, gpu_util=0.90):
    """啟動 vllm serve，等到 /health 回 200 才回傳。"""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGDIR / f"vllm_{tag}.log"
    cmd = [
        "vllm", "serve", TARGET_MODEL,
        "--served-model-name", TARGET_MODEL,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_util),
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ] + (extra_args or [])

    print(f"[srv:{tag}] {' '.join(cmd)}", flush=True)
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=server_env())

    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[srv:{tag}] 啟動失敗 (exit={proc.returncode})，log 尾端：", flush=True)
            print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
            raise RuntimeError(f"vllm server {tag} 啟動失敗，詳見 {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    print(f"[srv:{tag}] ready on :{port}  ({time.time() and ''})", flush=True)
                    return proc, log_path
        except Exception:
            time.sleep(3)
    proc.kill()
    raise TimeoutError(f"vllm server {tag} 900 秒內未 ready，詳見 {log_path}")


def stop_server(proc):
    """溫柔關閉 vLLM（SIGTERM 讓它釋放 GPU）。"""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
    time.sleep(5)


def chat_stream(port, prompt, max_tokens, model=TARGET_MODEL, extra_body=None,
                use_template_kwargs=True):
    """送一個 streaming chat request，回傳 (ttft, 總時間, 內容, usage)。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if use_template_kwargs:
        # Qwen3 預設開 thinking；論文的設定是 thinking off
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if extra_body:
        payload.update(extra_body)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft, parts, usage = None, [], None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            delta = (obj.get("choices") or [{}])[0].get("delta", {}) or {}
            content = delta.get("content")
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(content)
    total = time.perf_counter() - t0
    return ttft, total, "".join(parts), usage
def fetch_metrics(port=PORT):
    """抓 vLLM 的 /metrics，擷取 speculative decoding 相關統計。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        return {"error": str(e)}
    out = {}
    for line in text.splitlines():
        if "spec_decode" not in line or line.startswith("#"):
            continue
        try:
            name, value = line.rsplit(" ", 1)
            out[name.strip()] = float(value)
        except ValueError:
            continue
    return out


def metric_total(metrics, key_part):
    """把指定 counter family 的數值加總（Prometheus 可能有多個 label 組合）。

    Prometheus 對每個 counter 都會產生兩條 series：
      vllm:<name>_total    真正的累計值
      vllm:<name>_created  該 series 的建立時間戳（單位是秒，不是計數）
    用 substring 比對會同時誤抓：
      1. `_created` 的時間戳（數值約 1.79e9，直接把總和灌爆）
      2. 兄弟 family，例如 spec_decode_num_accepted_tokens_per_pos_total
    因此這裡改為「去掉 label 之後 family 名稱完全相等」的比對。
    """
    total, hit = 0.0, False
    for k, v in metrics.items():
        family = k.split("{", 1)[0]
        if family.startswith("vllm:"):
            family = family[len("vllm:"):]
        if family == key_part or family == key_part + "_total":
            total += v
            hit = True
    return total if hit else None


def run_suite(port, prompts, max_tokens, tag):
    """依序送出所有 prompt（concurrency=1），回傳每筆延遲記錄。"""
    records = []
    use_tk = True
    for i, item in enumerate(prompts):
        prompt, category = item["prompt"], item.get("category", "misc")
        try:
            ttft, total, text, usage = chat_stream(port, prompt, max_tokens, use_template_kwargs=use_tk)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if use_tk and "chat_template_kwargs" in body:
                print("[warn] server 不接受 chat_template_kwargs，改用預設 chat template 重試", flush=True)
                use_tk = False
                ttft, total, text, usage = chat_stream(port, prompt, max_tokens, use_template_kwargs=False)
            else:
                raise
        rec = {
            "id": i, "category": category, "prompt": prompt,
            "ttft_sec": round(ttft, 3) if ttft else None,
            "total_sec": round(total, 3),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "output_head": text[:120],
            "output_text": text,
        }
        records.append(rec)
        print(f"  [{tag}][{i+1:2d}/{len(prompts)}] {category:6s} "
              f"tok={rec['completion_tokens']} ttft={rec['ttft_sec']}s total={rec['total_sec']}s", flush=True)
    return records


def summarize(records):
    """整體與分 category 的吞吐量彙總（completion_tokens / 總牆鐘時間）。"""
    out = {}
    groups = {"ALL": records}
    for cat in sorted({r["category"] for r in records}):
        groups[cat] = [r for r in records if r["category"] == cat]
    for name, rs in groups.items():
        wall = sum(r["total_sec"] for r in rs)
        toks = sum(r["completion_tokens"] or 0 for r in rs)
        ttfts = [r["ttft_sec"] for r in rs if r["ttft_sec"]]
        out[name] = {
            "n": len(rs),
            "wall_sec": round(wall, 2),
            "completion_tokens": toks,
            "output_tok_per_sec": round(toks / wall, 2) if wall > 0 else None,
            "mean_ttft_sec": round(statistics.mean(ttfts), 3) if ttfts else None,
            "mean_tok_per_sec": round(
                statistics.mean([(r["completion_tokens"] or 0) / r["total_sec"] for r in rs]), 2),
        }
    return out
def spec_decode_stats(metrics):
    """從 Prometheus counter 算出真正的 spec-decode 統計量。

    vLLM v1 的語義（見 vllm/v1/spec_decode/metrics.py）：
      num_drafts          每個 decode step 的「草稿輪次」數（+1 per step）
      num_draft_tokens    總共草擬出多少個 token
      num_accepted_tokens 總共被接受多少個草稿 token

    因此：
      γ（每個 target forward 實際送出的 token 數）= accepted/drafts + 1
          （+1 是 target 自己 bonus 產生的那個 token）
      草稿接受率（被接受的草稿 token 佔草稿 token 的比例）= accepted/draft_tokens
    """
    if not metrics or "error" in metrics:
        return {}
    accepted = metric_total(metrics, "spec_decode_num_accepted_tokens")
    drafts = metric_total(metrics, "spec_decode_num_drafts")
    draft_tokens = metric_total(metrics, "spec_decode_num_draft_tokens")
    out = {}
    if accepted is not None and drafts:
        out["accepted_tokens"] = accepted
        out["draft_steps"] = drafts
        out["gamma_tokens_per_step"] = round(accepted / drafts + 1, 3)
    if accepted is not None and draft_tokens:
        out["draft_tokens"] = draft_tokens
        out["acceptance_fraction"] = round(accepted / draft_tokens, 4)
    if draft_tokens is not None and drafts:
        out["draft_tokens_per_step"] = round(draft_tokens / drafts, 3)
    return out


def compare_outputs(base_records, fm_records):
    """伺服器端無損性：同一台 vLLM、同一組 prompt、temperature=0，比對兩次執行的輸出。

    這是「使用者真正在意的」那一層：FlashMap 開起來之後，回答有沒有變。
    比對兩層：
      identical_text  完整字串相同（最嚴格）
      identical_head  前 120 字相同（容忍最後一次驗證 pass 衝過 max_tokens 造成的一兩個尾字差異）
    """
    base = {r["id"]: r for r in base_records}
    fm = {r["id"]: r for r in fm_records}
    rows = []
    for i in sorted(base):
        b, f = base[i], fm[i]
        bt, ft = b.get("output_text") or b.get("output_head", ""), f.get("output_text") or f.get("output_head", "")
        hb, hf = b.get("output_head", ""), f.get("output_head", "")
        n_min = min(len(bt), len(ft))
        first = next((k for k in range(n_min) if bt[k] != ft[k]), None)
        rows.append({
            "id": i, "category": b["category"],
            "identical_text": bt == ft,
            "identical_head": hb == hf,
            "first_char_diff": first,
            "base_tokens": b.get("completion_tokens"),
            "fm_tokens": f.get("completion_tokens"),
            "token_delta": (f.get("completion_tokens") or 0) - (b.get("completion_tokens") or 0),
        })
    return {
        "n": len(rows),
        "identical_text": sum(1 for r in rows if r["identical_text"]),
        "identical_head": sum(1 for r in rows if r["identical_head"]),
        "rows": rows,
    }


def build_report(env, base_sum, fm_sum, fm_metrics, out_md, lossless=None):
    """產生人可讀的 markdown 報告（含與論文數字的對照）。"""

    def row(label, b, f):
        bt = (b or {}).get("output_tok_per_sec")
        ft = (f or {}).get("output_tok_per_sec")
        sp = round(ft / bt, 3) if bt and ft else None
        return (f"| {label} | {(b or {}).get('completion_tokens', '-')} | "
                f"{bt if bt else '-'} | {ft if ft else '-'} | {sp if sp else '-'} |")

    sd = spec_decode_stats(fm_metrics)

    lines = [
        "# FlashMap 本機實驗報告",
        "",
        f"產生時間：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 環境",
        "",
        f"- GPU：{env.get('gpu')}",
        f"- 驅動：{env.get('driver')}",
        f"- vLLM：{env.get('vllm')} / torch：{env.get('torch')}",
        f"- target：`{env.get('target')}`　drafter：`{env.get('draft')}`",
        "- 量測條件：concurrency=1、temperature=0（greedy）、thinking off",
        "",
        "## 結果（vLLM 端到端）",
        "",
        "| 項目 | 基準 output tokens | 基準 tok/s | FlashMap tok/s | 加速比 |",
        "|---|---|---|---|---|",
        row("ALL", base_sum.get("ALL"), fm_sum.get("ALL")),
    ]
    for cat in sorted({k for k in base_sum if k != "ALL"}):
        lines.append(row(cat, base_sum.get(cat), fm_sum.get(cat)))
    lines += [
        "",
        ("- Spec-decode 統計（由 `/metrics` 的 `_total` counter 計算）：" if sd
         else "- （未能從 /metrics 取得 spec-decode counter）"),
    ]
    if sd:
        lines += [
            f"  - γ（每個 target forward 送出的 token 數）= **{sd['gamma_tokens_per_step']}**"
            f"　（accepted {sd['accepted_tokens']:.0f} / steps {sd['draft_steps']:.0f} + 1 bonus）",
            f"  - 草稿接受率（accepted / drafted tokens）= **{sd['acceptance_fraction'] * 100:.2f}%**"
            f"　（每步草擬 {sd['draft_tokens_per_step']:.1f} 個 token）",
        ]
    if lossless:
        lines += [
            "",
            "## 伺服器端輸出比對（無損性，temperature=0）",
            "",
            f"- 完整輸出字串相同：**{lossless['identical_text']}/{lossless['n']}**",
            f"- 前 120 字相同：**{lossless['identical_head']}/{lossless['n']}**",
            "",
            "| id | 類別 | 基準 tokens | FlashMap tokens | 差異 | 前120字相同 | 完整相同 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in lossless["rows"]:
            lines.append(
                f"| {r['id']} | {r['category']} | {r['base_tokens']} | {r['fm_tokens']} | "
                f"{r['token_delta']:+d} | {'✔' if r['identical_head'] else '✘'} | "
                f"{'✔' if r['identical_text'] else '✘'} |")
        lines += [
            "",
            "  `+1` 的 token 差異來自最後一次驗證 pass 會衝過 `max_tokens`"
            "（上游文件明載），不是內容不同。",
            "  temperature=0 下兩次執行理論上應完全一致；在 bf16 下，target 的「逐字解碼」與"
            "「區塊驗證」前向形狀不同，",
            "  在 top-2 logit 差距小於 bf16 解析度時 argmax 會翻轉 —— "
            "詳見 `pass6_block_shape.py` 與 `FINDINGS.md`。",
        ]
    lines += [
        "",
        "## 與論文對照（FlashMap 論文 Table 1 / Figure 1(c)，Qwen3-8B，T=0，1 step）",
        "",
        "| 領域 | 論文 γ | 論文加速 (H200 NVL, SGLang) | 本機 tok/s (RTX 4090, vLLM) |",
        "|---|---|---|---|",
        f"| 數學 | 6.00–7.69 | 4.89–6.14× | {fm_sum.get('math', {}).get('output_tok_per_sec', '-')} |",
        f"| 程式 | 4.95–6.65 | 4.02–5.26× | {fm_sum.get('code', {}).get('output_tok_per_sec', '-')} |",
        f"| 聊天 | 3.14–3.71 | 2.56–3.04× | {fm_sum.get('chat', {}).get('output_tok_per_sec', '-')} |",
        "",
        "## 解讀注意事項",
        "",
        "1. 論文在 **H200 NVL** 量測，本機是 RTX 4090；加速比取決於 γ 與 drafter/target 的",
        "   相對成本，理論上與硬體無關，但 Python/scheduler 固定開銷在高速率下占比更高，",
        "   實際倍數通常略低於論文值。",
        "2. 論文用 **SGLang 0.5.20**，本實驗用 **vLLM 0.31.0**，kernel 與排程不同，",
        "   絕對 tok/s 不可直接比較，只看同一台機器上的加速比。",
        "3. 聊天類加速本來就低於數學/程式（開源性文字有很多合理續寫）。",
        "4. 本實驗為 concurrency=1；**批次服務下的加速未驗證**，高併發時 drafter 會成為額外負擔。",
        "",
    ]
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(Path(__file__).with_name("prompts.json")))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out-json", default="results_vllm.json")
    ap.add_argument("--out-md", default="report.md")
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    print(f"[info] {len(prompts)} prompts, max_tokens={args.max_tokens}, port={PORT}", flush=True)

    import torch
    import vllm
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "driver": os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "target": TARGET_MODEL,
        "draft": DRAFT_MODEL,
    }
    print(f"[env] {env}", flush=True)

    results = {"env": env, "prompts_file": args.prompts, "max_tokens": args.max_tokens}

    print("\n===== A. 基準（無推測解碼）=====", flush=True)
    proc, _ = start_server("baseline")
    try:
        base_records = run_suite(PORT, prompts, args.max_tokens, "base")
        base_metrics = fetch_metrics(PORT)
    finally:
        stop_server(proc)
    base_sum = summarize(base_records)
    print(f"[baseline] {base_sum['ALL']}", flush=True)

    print("\n===== B. FlashMap（推測解碼）=====", flush=True)
    spec_cfg = json.dumps({"model": DRAFT_MODEL, "num_speculative_tokens": 16})
    proc, _ = start_server("flashmap", extra_args=["--speculative-config", spec_cfg])
    try:
        fm_records = run_suite(PORT, prompts, args.max_tokens, "flash")
        fm_metrics = fetch_metrics(PORT)
    finally:
        stop_server(proc)
    fm_sum = summarize(fm_records)
    print(f"[flashmap] {fm_sum['ALL']}", flush=True)

    results.update({"baseline": {"records": base_records, "summary": base_sum, "metrics": base_metrics},
                    "flashmap": {"records": fm_records, "summary": fm_sum, "metrics": fm_metrics}})
    lossless = compare_outputs(base_records, fm_records)
    results["lossless"] = lossless
    print(f"[lossless] identical_text={lossless['identical_text']}/{lossless['n']} "
          f"identical_head={lossless['identical_head']}/{lossless['n']}", flush=True)
    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 原始結果 → {args.out_json}")
    print(build_report(env, base_sum, fm_sum, fm_metrics, args.out_md, lossless))
    print(f"[done] 報告 → {args.out_md}")


if __name__ == "__main__":
    sys.exit(main())



#!/usr/bin/env python3
"""
FlashMap 實驗 Part 1：接受長度 (acceptance length γ) + 無損性驗證

用 flashmap 套件的 transformers 參考實作（不需要啟 server），對每個 prompt 做兩件事：
  A. target 逐字 greedy 生成，作為基準輸出（呼叫簽章對齊上游 examples/benchmark.py）
  B. FlashMap spec_generate (temperature=0) 生成，比對「token id」是否完全相同 → 驗證「無損」
     並讀取 output.acceptance_length → 接受長度 γ

無損性判準與上游一致：torch.equal(baseline_ids, spec_ids)，不看解碼後的字串。
（字串比對會被 skip_special_tokens / strip() 掩蓋 EOS 與空白的差異。）
另外單獨記錄 lossless_prefix_ok：最後一次驗證 pass 可能衝過 max_new_tokens，
此時兩者互為前綴、在 max_new_tokens 內完全一致，不應算成無損性失敗。

用法：python acceptance.py --prompts prompts.json --max-new-tokens 512 [--dtype bfloat16|float32]

注意：--dtype float32 是「無損性」的嚴格檢查模式。bf16 下 logits 差 < 一個 ULP 的近似並列
      (near-tie) 會因四捨五入而讓 argmax 翻轉，造成假性的 MISMATCH；上游 examples/benchmark.py
      亦明確建議以 --dtype float32 做 exact 比對。
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

TARGET_MODEL = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-8B")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "j-shen/FlashMap-Qwen3-8B")

DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def build_inputs(tokenizer, prompt, enable_thinking=False):
    """套用 Qwen3 chat template；enable_thinking=False 對齊論文的 thinking off 設定。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_tokens=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
        return_dict=True,
    )


def measure_acceptance(prompts, max_new_tokens=512, num_steps=1, target_model=TARGET_MODEL,
                       draft_model=DRAFT_MODEL, verbose=True, dtype="bfloat16"):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, DTYPES[dtype])
    print(f"[load] target={target_model} dtype={dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    target = AutoModelForCausalLM.from_pretrained(
        target_model, dtype=torch_dtype
    ).cuda().eval()

    print(f"[load] drafter={draft_model} dtype={dtype}", flush=True)
    draft = AutoModel.from_pretrained(
        draft_model, trust_remote_code=True, dtype=torch_dtype
    ).cuda().eval()

    records = []
    for i, item in enumerate(prompts):
        prompt, category = item["prompt"], item.get("category", "misc")
        enc = build_inputs(tokenizer, prompt)
        input_ids = enc["input_ids"].cuda()
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.cuda()

        # A. target 逐字 greedy 基準。
        #    與上游 examples/benchmark.py 的基準保持一致：
        #      options = dict(do_sample=False, temperature=1.0)
        #    也就是「不傳 attention_mask、顯式把 temperature 釘在 1.0（蓋掉 Qwen3
        #    generation_config 的 0.6）」。PASS 4 已驗證這兩處差異在數值上是惰性的
        #    （4 種簽章在 bf16 下輸出 token id 完全相同），但對齊上游仍是正確做法。
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = target.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        base_ids = out[0, input_ids.shape[1]:]
        base_text = tokenizer.decode(base_ids, skip_special_tokens=True)
        base_time = time.perf_counter() - t0

        # B. FlashMap greedy（無損）+ 接受長度
        t0 = time.perf_counter()
        try:
            res = draft.spec_generate(
                target, input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                num_steps=num_steps,
            )
        except TypeError:
            # 部分版本需要同時傳 attention_mask
            res = draft.spec_generate(
                target, {"input_ids": input_ids, "attention_mask": attn},
                max_new_tokens=max_new_tokens, temperature=0.0, num_steps=num_steps,
            )
        spec_ids = res.sequences[0, input_ids.shape[1]:]
        acc = float(getattr(res, "acceptance_length", float("nan")))
        spec_text = tokenizer.decode(spec_ids, skip_special_tokens=True)
        spec_time = time.perf_counter() - t0

        # 無損性判準：與上游 examples/benchmark.py 一致，比對「token id」而不是解碼字串
        # （字串比對會被 skip_special_tokens 掩蓋 EOS 差異，也會被 strip() 抹掉空白差異）。
        base_list, spec_list = base_ids.tolist(), spec_ids.tolist()
        n_min = min(len(base_list), len(spec_list))
        first_diff = next((k for k in range(n_min) if base_list[k] != spec_list[k]), None)
        identical = first_diff is None and len(base_list) == len(spec_list)
        # 最後一次驗證 pass 可能衝過 max_new_tokens（上游文件明載），此時兩者互為前綴、
        # 在 max_new_tokens 之前完全一致 —— 這不是無損性失敗，要分開記錄。
        prefix_ok = first_diff is None
        text_identical = base_text.strip() == spec_text.strip()
        rec = {
            "id": i,
            "category": category,
            "prompt": prompt,
            "dtype": dtype,
            "lossless_identical": identical,
            "lossless_prefix_ok": prefix_ok,
            "text_identical": text_identical,
            "first_diff": first_diff,
            "spec_overshoot": max(0, int(spec_ids.shape[0]) - max_new_tokens),
            "acceptance_length": acc,
            "base_tokens": int(base_ids.shape[0]),
            "spec_tokens": int(spec_ids.shape[0]),
            "base_sec": round(base_time, 3),
            "spec_sec": round(spec_time, 3),
            "hf_speedup": round(base_time / spec_time, 3) if spec_time > 0 else None,
            "base_text_head": base_text[:160],
            "spec_text_head": spec_text[:160],
        }
        records.append(rec)
        print(f"  [{i+1:2d}/{len(prompts)}] {category:6s} "
              f"g={acc:.2f} token_exact={'OK' if identical else 'MISMATCH'} "
              f"text={'OK' if text_identical else 'DIFF'} first_diff={first_diff} "
              f"base={rec['base_sec']:.1f}s spec={rec['spec_sec']:.1f}s "
              f"({rec['hf_speedup']}x)", flush=True)
    return {"records": records, "summary": summarize(records)}


def summarize(records):
    """依 category 與整體彙總接受長度與無損性。"""
    summary = {}
    for cat in sorted({r["category"] for r in records}):
        rs = [r for r in records if r["category"] == cat]
        accs = [r["acceptance_length"] for r in rs if r["acceptance_length"] == r["acceptance_length"]]
        sps = [r["hf_speedup"] for r in rs if r["hf_speedup"]]
        summary[cat] = {
            "n": len(rs),
            "mean_acceptance_length": round(statistics.mean(accs), 3) if accs else None,
            "mean_hf_speedup": round(statistics.mean(sps), 3) if sps else None,
            "all_lossless": all(r["lossless_identical"] for r in rs),
            "all_lossless_prefix_ok": all(r.get("lossless_prefix_ok", False) for r in rs),
            "text_identical_count": sum(1 for r in rs if r.get("text_identical")),
        }
    all_accs = [r["acceptance_length"] for r in records if r["acceptance_length"] == r["acceptance_length"]]
    summary["ALL"] = {
        "n": len(records),
        "dtype": records[0].get("dtype") if records else None,
        "mean_acceptance_length": round(statistics.mean(all_accs), 3) if all_accs else None,
        "all_lossless": all(r["lossless_identical"] for r in records),
        "all_lossless_prefix_ok": all(r.get("lossless_prefix_ok", False) for r in records),
        "token_exact_count": sum(1 for r in records if r["lossless_identical"]),
        "text_identical_count": sum(1 for r in records if r.get("text_identical")),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(Path(__file__).with_name("prompts.json")))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=1, help="1 = 最快；2 = 接受長度較高但慢一點")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES),
                    help="目標/草稿模型精度；嚴格無損性檢查請用 float32（見上游 benchmark.py 說明）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    print(f"[info] {len(prompts)} 個 prompt, num_steps={args.num_steps}, dtype={args.dtype}", flush=True)
    result = measure_acceptance(prompts, max_new_tokens=args.max_new_tokens,
                                num_steps=args.num_steps, dtype=args.dtype)

    suffix = "" if args.dtype == "bfloat16" else f"_{args.dtype}"
    out = args.out or str(Path(__file__).with_name(f"results_acceptance_step{args.num_steps}{suffix}.json"))
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 結果已寫入 {out}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0 if result["summary"]["ALL"]["all_lossless"] else 2


if __name__ == "__main__":
    sys.exit(main())


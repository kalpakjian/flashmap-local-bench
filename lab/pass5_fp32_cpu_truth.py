#!/usr/bin/env python3
"""PASS 5 — 對「Phase 1 實際使用的 8B 模型」取得 fp32 真值（CPU 版）

Phase 1 的 all_lossless=false 是在 Qwen3-8B + bf16 上得到的。8B 的 fp32 權重需要
32 GiB，超過 RTX 4090 的 24 GiB，所以「同一模型、換 dtype」的完整 A/B 只能在 4B 上做。
但我們不需要整個 8B fp32 生成流程 —— 只需要「在共享前綴上做一次前向」的真值。
因此本腳本把 8B 的 fp32 forward 放到 CPU（32 GiB RAM，放得下），只對每個分歧點算一次。

兩階段，可分開執行（第二階段完全不需要 GPU）：

  --stage collect   在 GPU 上以 bf16 跑 baseline 與 spec_generate，找出每個 prompt 的
                    第一次分歧，記錄共享前綴 ctx 的 token id、以及 bf16 在該前綴上的
                    top-2 logits 與兩個候選字的 logit。→ results_pass5_8b_fp32_truth.json
  --stage truth     在 CPU 上以 fp32 載入同一個 target，對每個 ctx 算一次 logits，
                    取得真實 top-2 gap，並與 bf16 的評測誤差比較。

判準：若真實 gap 不大於 bf16 在同樣 logit 上的評測誤差，bf16 根本無法排序這兩個候選字，
分歧就是數值假象而非演算法錯誤。
"""
import argparse
import json
import math
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent


def bf16_ulp(v):
    """bf16 在量級 v 處的相鄰間距（8 個 mantissa bit）。"""
    v = abs(float(v))
    if v == 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(v)) - 7)


def encode(tokenizer, prompt, device="cuda"):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"].to(device)


def stage_collect(args):
    """GPU / bf16：找出共享前綴，記下 bf16 在該前綴上的 logits。"""
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.target)
    pad_id = tok.pad_token_id or tok.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).cuda().eval()
    draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True,
                                     dtype=torch.bfloat16).cuda().eval()

    rows = []
    for i, item in enumerate(prompts):
        ids = encode(tok, item["prompt"])
        with torch.inference_mode():
            base = target.generate(input_ids=ids, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, temperature=1.0,
                                   pad_token_id=pad_id)[0, ids.shape[1]:]
            res = draft.spec_generate(target, ids, max_new_tokens=args.max_new_tokens,
                                      temperature=0.0, num_steps=1)
        spec = res.sequences[0, ids.shape[1]:]
        n = min(base.shape[0], spec.shape[0])
        d = next((k for k in range(n) if int(base[k]) != int(spec[k])), None)
        if d is None:
            print(f"  [{i:2d}] no divergence", flush=True)
            continue
        ctx = torch.cat([ids[0], base[:d]])[None].cuda()
        with torch.inference_mode():
            lbf = target(ctx).logits[0, -1].float()
        t2 = torch.topk(lbf, 2)
        bt, st = int(base[d]), int(spec[d])
        rec = dict(
            id=i, category=item.get("category"), first_diff=int(d),
            base_tok=bt, spec_tok=st,
            base_tok_str=tok.decode([bt]), spec_tok_str=tok.decode([st]),
            ctx_ids=[int(x) for x in ctx[0].tolist()],
            bf16_top2_ids=[int(x) for x in t2.indices],
            bf16_top2_logits=[float(x) for x in t2.values],
            bf16_ctx_gap=float(t2.values[0] - t2.values[1]),
            bf16_ulp_at_top1=bf16_ulp(t2.values[0]),
            bf16_ctx_argmax=int(t2.indices[0]),
            bf16_logit_base_tok=float(lbf[bt]),
            bf16_logit_spec_tok=float(lbf[st]),
            bf16_gap_base_minus_spec=float(lbf[bt] - lbf[st]),
        )
        rows.append(rec)
        print(f"  [{i:2d}] {item.get('category',''):5s} diff@{d} "
              f"base={rec['base_tok_str']!r} spec={rec['spec_tok_str']!r} "
              f"bf16 gap(base-spec)={rec['bf16_gap_base_minus_spec']:+.4f} "
              f"ctx_len={len(rec['ctx_ids'])}", flush=True)
        Path(args.out).write_text(json.dumps({"stage": "collect", "rows": rows},
                                             ensure_ascii=False, indent=2), encoding="utf-8")

    Path(args.out).write_text(json.dumps({"stage": "collect", "rows": rows},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect done] {len(rows)} divergences → {args.out}")
    return 0


def stage_truth(args):
    """CPU / fp32：對每個共享前綴算真實 logits。不需要 GPU。"""
    import torch
    from transformers import AutoModelForCausalLM

    data = json.loads(Path(args.out).read_text(encoding="utf-8"))
    rows = data["rows"]
    prev_summary = data.get("summary")
    print(f"[truth] loading {args.target} in float32 on CPU ({len(rows)} prefixes) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.float32)
    model.eval()
    print("[truth] loaded", flush=True)

    for rec in rows:
        ctx = torch.tensor([rec["ctx_ids"]], dtype=torch.long)
        with torch.inference_mode():
            lg = model(input_ids=ctx).logits[0, -1].float()
        t2 = torch.topk(lg, 2)
        bt, st = rec["base_tok"], rec["spec_tok"]
        # 可以吃 pass6 的輸出（那裡沒有 bf16_ulp_at_top1，但 top2 logits 有）
        if "bf16_ulp_at_top1" not in rec:
            rec["bf16_ulp_at_top1"] = bf16_ulp(rec["bf16_top2_logits"][0])
        rec["fp32_top2_ids"] = [int(x) for x in t2.indices]
        rec["fp32_top2_logits"] = [float(x) for x in t2.values]
        rec["fp32_ctx_gap"] = float(t2.values[0] - t2.values[1])
        rec["fp32_argmax"] = int(t2.indices[0])
        rec["fp32_logit_base_tok"] = float(lg[bt])
        rec["fp32_logit_spec_tok"] = float(lg[st])
        rec["fp32_gap_base_minus_spec"] = float(lg[bt] - lg[st])
        rec["fp32_base_tok_rank"] = int((lg > lg[bt]).sum())
        rec["fp32_spec_tok_rank"] = int((lg > lg[st]).sum())
        e_base = abs(rec["bf16_logit_base_tok"] - rec["fp32_logit_base_tok"])
        e_spec = abs(rec["bf16_logit_spec_tok"] - rec["fp32_logit_spec_tok"])
        rec["bf16_eval_error_base_tok"] = e_base
        rec["bf16_eval_error_spec_tok"] = e_spec
        rec["bf16_eval_error_max"] = max(e_base, e_spec)
        rec["fp32_gap_below_bf16_eval_error"] = bool(
            abs(rec["fp32_gap_base_minus_spec"]) <= max(e_base, e_spec))
        rec["fp32_gap_below_one_ulp"] = bool(rec["fp32_ctx_gap"] < rec["bf16_ulp_at_top1"])
        rec["spec_token_is_fp32_argmax"] = bool(rec["fp32_argmax"] == st)
        rec["base_token_is_fp32_argmax"] = bool(rec["fp32_argmax"] == bt)
        print(f"  [{rec['id']:2d}] diff@{rec['first_diff']:4d} "
              f"fp32 gap={rec['fp32_ctx_gap']:.5f} ulp={rec['bf16_ulp_at_top1']:.4f} | "
              f"logit(base-spec)={rec['fp32_gap_base_minus_spec']:+.5f} "
              f"bf16_err={rec['bf16_eval_error_max']:.5f} "
              f"spec_is_fp32_argmax={rec['spec_token_is_fp32_argmax']} "
              f"unresolvable={rec['fp32_gap_below_bf16_eval_error']}", flush=True)
        Path(args.out).write_text(json.dumps({"stage": "truth", "rows": rows},
                                             ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "n_divergences": len(rows),
        "fp32_below_one_bf16_ulp": sum(1 for r in rows if r["fp32_gap_below_one_ulp"]),
        "fp32_below_bf16_eval_error": sum(1 for r in rows if r["fp32_gap_below_bf16_eval_error"]),
        "spec_token_is_fp32_argmax": sum(1 for r in rows if r["spec_token_is_fp32_argmax"]),
        "base_token_is_fp32_argmax": sum(1 for r in rows if r["base_token_is_fp32_argmax"]),
        "max_fp32_gap": round(max(r["fp32_ctx_gap"] for r in rows), 6) if rows else None,
        "max_bf16_eval_error": round(max(r["bf16_eval_error_max"] for r in rows), 6) if rows else None,
    }
    Path(args.out).write_text(json.dumps({"stage": "truth", "summary": summary,
                                         "shape_summary": prev_summary, "rows": rows},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[truth done] {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["collect", "truth"])
    ap.add_argument("--prompts", default=str(LAB / "prompts.json"))
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="j-shen/FlashMap-Qwen3-8B")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=str(LAB / "results_pass5_8b_fp32_truth.json"))
    args = ap.parse_args()
    return stage_collect(args) if args.stage == "collect" else stage_truth(args)


if __name__ == "__main__":
    sys.exit(main())

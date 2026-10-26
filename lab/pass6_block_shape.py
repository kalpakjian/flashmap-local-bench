#!/usr/bin/env python3
"""PASS 6 — 真正的「前向形狀」量測：17-position 區塊驗證 vs 1-position 逐字解碼

FlashMap 的驗證是「一次前向處理 17 個位置」（anchor + 16 個 draft token），
而 target 的逐字 greedy 解碼是「每個位置一次前向」。兩者在數學上完全等價，
但在 bf16 下，一次 GEMM 的 row 數不同 → 累加順序不同 → logits 的捨入不同。
上游 examples/benchmark.py 明文提醒這件事：

    "In bfloat16 the target's batched verification pass and its token-by-token
     decoding round differently, so the two can part where the target's top two
     logits tie to within bfloat16 precision. Run with --dtype float32 for an exact check."

本腳本在「共同前綴」上直接量這個差異：對同一個 next-token 分布，用
  k=0  一次前向吃下整個前綴（純 prefill 形狀）
  k=1  前綴最後 1 個位置單獨前向（= generate() 的逐字解碼形狀）
  k=16 前綴最後 16 個位置一起前向（= 驗證區塊的形狀）
分別算 logits，比較 argmax 是否翻轉、以及兩個候選 token 的 logit 差多少。

同時把 ctx 的 token id 一起存下來，讓 pass5 的 fp32 CPU 真值階段可以直接接著跑
（pass5 只會新增欄位，不會覆蓋這裡的資料）。
"""
import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent


def encode(tokenizer, prompt, device="cuda"):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"].to(device)


def shape_logits(torch, model, ctx, k):
    """ctx 最後位置的 next-token logits，用「k 個位置一起前向」的形狀算。

    k=0  → 整個 ctx 一次前向（純 prefill 形狀）
    k>0  → 先把 ctx[:-k] 灌進 KV cache，再把最後 k 個位置當一個區塊前向
    """
    from transformers import DynamicCache

    with torch.inference_mode():
        if k == 0:
            try:
                out = model(input_ids=ctx, use_cache=False, logits_to_keep=1)
            except TypeError:
                out = model(input_ids=ctx, use_cache=False)
            return out.logits[0, -1].float()
        cache = DynamicCache()
        if ctx.shape[1] > k:
            model(input_ids=ctx[:, :-k], past_key_values=cache, use_cache=True)
        out = model(input_ids=ctx[:, -k:], past_key_values=cache, use_cache=True)
        return out.logits[0, -1].float()


def run(args):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.target)
    pad_id = tok.pad_token_id or tok.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).cuda().eval()
    draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True,
                                     dtype=dtype).cuda().eval()
    print(f"[load] {args.target} / {args.draft} dtype={args.dtype} "
          f"mem={torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    ks = args.ks
    rows = []
    src = None
    if getattr(args, "ctx_from", None):
        src = json.loads(Path(args.ctx_from).read_text(encoding="utf-8"))["rows"]
        print(f"[ctx-from] reusing {len(src)} divergence contexts from {args.ctx_from} "
              f"(no generation) — 用來在 fp32 下重測 bf16 的分歧點", flush=True)
    for i in range(len(src) if src is not None else len(prompts)):
        if src is not None:
            r0 = src[i]
            item = {"category": r0.get("category")}
            ctx = torch.tensor([r0["ctx_ids"]], dtype=torch.long, device="cuda")
            bt, st, d = int(r0["base_tok"]), int(r0["spec_tok"]), r0["first_diff"]
        else:
            item = prompts[i]
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
                print(f"  [{i:2d}] {item.get('category',''):5s} no divergence", flush=True)
                continue
            ctx = torch.cat([ids[0], base[:d]])[None].cuda()
            bt, st = int(base[d]), int(spec[d])
        per_shape = {}
        for k in ks:
            lg = shape_logits(torch, target, ctx, k)
            t2 = torch.topk(lg, 2)
            per_shape[str(k)] = dict(
                argmax=int(t2.indices[0]),
                top2_ids=[int(x) for x in t2.indices],
                top2_logits=[float(x) for x in t2.values],
                gap=float(t2.values[0] - t2.values[1]),
                logit_base_tok=float(lg[bt]),
                logit_spec_tok=float(lg[st]),
                gap_base_minus_spec=float(lg[bt] - lg[st]),
                picks_base=bool(int(t2.indices[0]) == bt),
                picks_spec=bool(int(t2.indices[0]) == st),
            )
        argmaxes = {k: v["argmax"] for k, v in per_shape.items()}
        rec = dict(
            id=(int(src[i]["id"]) if src is not None else i),
            category=item.get("category"), first_diff=int(d),
            base_tok=bt, spec_tok=st,
            base_tok_str=tok.decode([bt]), spec_tok_str=tok.decode([st]),
            ctx_ids=[int(x) for x in ctx[0].tolist()],
            per_shape=per_shape,
            shape_argmaxes=argmaxes,
            n_distinct_argmax=len(set(argmaxes.values())),
            argmax_flips_across_shapes=len(set(argmaxes.values())) > 1,
            # pass5-compatible names (full-prefill / k=0 shape)
            bf16_top2_ids=per_shape["0"]["top2_ids"],
            bf16_top2_logits=per_shape["0"]["top2_logits"],
            bf16_ctx_gap=per_shape["0"]["gap"],
            bf16_ctx_argmax=per_shape["0"]["argmax"],
            bf16_logit_base_tok=per_shape["0"]["logit_base_tok"],
            bf16_logit_spec_tok=per_shape["0"]["logit_spec_tok"],
            bf16_gap_base_minus_spec=per_shape["0"]["gap_base_minus_spec"],
        )
        rows.append(rec)
        detail = " ".join(
            f"k{k}:{per_shape[str(k)]['gap_base_minus_spec']:+.5f}"
            f"{'/B' if per_shape[str(k)]['picks_base'] else ''}"
            f"{'/S' if per_shape[str(k)]['picks_spec'] else ''}" for k in ks)
        print(f"  [{i:2d}] {item.get('category',''):5s} diff@{d:4d} "
              f"base={rec['base_tok_str']!r} spec={rec['spec_tok_str']!r} | {detail} | "
              f"flips={rec['argmax_flips_across_shapes']}", flush=True)
        Path(args.out).write_text(
            json.dumps({"stage": "shape", "dtype": args.dtype, "target": args.target,
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "dtype": args.dtype, "target": args.target,
        "n_prompts": len(prompts), "n_divergences": len(rows),
        "argmax_flips_across_shapes": sum(1 for r in rows if r["argmax_flips_across_shapes"]),
        "k1_vs_k16_flip": sum(1 for r in rows
                              if r["shape_argmaxes"].get("1") != r["shape_argmaxes"].get("16")),
        "k1_picks_base": sum(1 for r in rows if r["per_shape"].get("1", {}).get("picks_base")),
        "k16_picks_base": sum(1 for r in rows if r["per_shape"].get("16", {}).get("picks_base")),
        "k1_picks_spec": sum(1 for r in rows if r["per_shape"].get("1", {}).get("picks_spec")),
        "k16_picks_spec": sum(1 for r in rows if r["per_shape"].get("16", {}).get("picks_spec")),
        "max_abs_gap_k0": round(max((abs(r["per_shape"]["0"]["gap_base_minus_spec"])
                                     for r in rows), default=0), 6),
    }
    out = {"stage": "shape", "dtype": args.dtype, "target": args.target,
           "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(LAB / "prompts.json"))
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="j-shen/FlashMap-Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 1, 16])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--ctx-from", default=None,
                    help="reuse the divergence contexts from another pass6 JSON instead of "
                         "generating (use it to re-measure bf16 divergence points under fp32)")
    ap.add_argument("--out", required=True)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())

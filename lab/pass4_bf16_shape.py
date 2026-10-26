#!/usr/bin/env python3
"""PASS 4 — bf16「無損性」為什麼驗不出來：基準呼叫簽章、前向形狀、跨 runtime

Phase 1 (lab/acceptance.py, bf16, Qwen3-8B) 回報 all_lossless=false；
同一組程式在 float32 下 12/12 完全一致 (PASS 2)。本腳本把「不一致」的來源
釘死在三個可測量、可重複的軸上，而不是演算法本身。

Part A  基準呼叫簽章 (2x2 factorial)
        lab 的基準呼叫與上游 examples/benchmark.py 的基準呼叫有兩處不同：
          * 上游不傳 attention_mask；lab 傳 all-ones mask
          * 上游顯式傳 temperature=1.0（蓋掉 Qwen3 generation_config 的 0.6）
        bf16 下這兩處是否改變 greedy 逐字輸出？fp32 下是否仍一致？
        （同時驗證 lab 那個多餘的 add_generation_tokens=False 是否影響 token。）

Part B  前向形狀 (batch=1 逐字解碼 vs batch=17 區塊驗證)
        同一份權重、同一個共同前綴，數學等價的兩個前向形狀在 bf16 下 argmax
        是否翻轉？把「bf16 形狀抖動」直接和「fp32 真實 top-2 gap」比大小。
        這正是上游 benchmark.py 明文提醒的機制：
          "the target's batched verification pass and its token-by-token decoding
           round differently, so the two can part where the target's top two logits
           tie to within bfloat16 precision. Run with --dtype float32 for an exact check."

Part C  跨 runtime
        上游式基準輸出 vs vLLM 0.31.0 server 的基準輸出（lab/results_vllm.json）。
        若 HF 的「無 mask、temperature=1.0」基準能對上 vLLM，而 lab 式基準對不上，
        就證明 Phase 1 的 mismatch 有一半是 lab 自己的呼叫簽章造成的。

用法：
    python pass4_bf16_shape.py --parts ABC --dtypes bfloat16 float32 \\
        --target Qwen/Qwen3-4B --draft j-shen/FlashMap-Qwen3-4B --out results_pass4_4b.json
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent

TARGET_MODEL = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-4B")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "j-shen/FlashMap-Qwen3-4B")

# 2x2 factorial over the two ways the lab baseline call differs from upstream's.
VARIANTS = [
    ("upstream", dict(mask=False, temperature=1.0)),   # examples/benchmark.py: greedy
    ("lab", dict(mask=True, temperature=None)),        # lab/acceptance.py 現況
    ("mask_only", dict(mask=True, temperature=1.0)),   # 只加 mask
    ("temp_only", dict(mask=False, temperature=None)),  # 只留 generation_config 的 0.6
]


def build_clean(tokenizer, prompt):
    """上游 benchmark.py 的 encode()：乾淨、不帶多餘 template 變數。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]


def build_legacy(tokenizer, prompt):
    """lab/acceptance.py 現況：多傳一個 add_generation_tokens=False。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_tokens=False,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]


def first_diff(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else n


def ids_eq(a, b):
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def greedy(target, ids, max_new, mask, temperature, tokenizer):
    """Exact replica of one baseline call variant."""
    kwargs = dict(max_new_tokens=max_new, do_sample=False,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if mask:
        import torch
        kwargs["attention_mask"] = torch.ones_like(ids)
    if temperature is not None:
        kwargs["temperature"] = temperature
    import torch
    with torch.inference_mode():
        out = target.generate(input_ids=ids, **kwargs)
    return out[0, ids.shape[1]:].tolist()


def probe(target, ctx, cand_a, cand_b, bs=17):
    """Same context, two forward shapes: batch=1 (token-by-token) vs batch=bs (verification).

    Mathematically identical inputs in every row, so any spread across the batch
    rows is pure floating-point shape noise.
    """
    import torch

    def last_logits(batch):
        # logits_to_keep=1 keeps the (bs, 1, V) readout instead of materialising
        # (bs, L, V), which would be several GiB at L~570 and bs=17.
        try:
            out = target(input_ids=batch, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = target(input_ids=batch, use_cache=False)
        return out.logits[:, -1].float()

    with torch.inference_mode():
        lg1 = last_logits(ctx)[0]
        lgb = last_logits(ctx.repeat(bs, 1))

    am1 = int(lg1.argmax())
    ams = lgb.argmax(dim=-1).tolist()
    a1, b1 = float(lg1[cand_a]), float(lg1[cand_b])
    col_a, col_b = lgb[:, cand_a], lgb[:, cand_b]
    return dict(
        batch1_argmax=am1,
        batch1_logit_a=a1,
        batch1_logit_b=b1,
        batch1_gap=a1 - b1,
        batch1_margin=float(lg1.topk(2).values[0] - lg1.topk(2).values[1]),
        batchN_argmax=ams,
        batchN_unique=len(set(ams)),
        batchN_rows_agreeing=sum(1 for x in ams if x == am1),
        batchN_spread_a=float(col_a.max() - col_a.min()),
        batchN_spread_b=float(col_b.max() - col_b.min()),
        batchN_flip_count=len(set(ams)) - 1,
    )


def run_dtype(torch, target, draft, tokenizer, prompts, dtype_name, max_new, parts, vllm_records):
    out = {"dtype": dtype_name, "target_dtype": str(target.dtype), "draft_dtype": str(draft.dtype)}

    # ---- template sanity: does the bogus kwarg change the prompt tokens? ----
    tmpl = []
    for i, item in enumerate(prompts):
        a, b = build_clean(tokenizer, item["prompt"]), build_legacy(tokenizer, item["prompt"])
        tmpl.append(dict(id=i, clean_len=int(a.shape[1]), legacy_len=int(b.shape[1]),
                         identical=ids_eq(a[0].tolist(), b[0].tolist())))
    out["template_check"] = dict(
        all_identical=all(t["identical"] for t in tmpl),
        lengths_match_all=all(t["clean_len"] == t["legacy_len"] for t in tmpl),
        rows=tmpl,
    )
    print(f"  [template] clean vs legacy token-identical: {out['template_check']['all_identical']}",
          flush=True)

    rows = []
    for i, item in enumerate(prompts):
        ids = build_clean(tokenizer, item["prompt"]).cuda()
        seqs = {}
        for name, opt in VARIANTS:
            seqs[name] = greedy(target, ids, max_new, opt["mask"], opt["temperature"], tokenizer)

        # FlashMap arm (the thing we are testing for losslessness)
        try:
            res = draft.spec_generate(target, ids, max_new_tokens=max_new,
                                      temperature=0.0, num_steps=1)
        except TypeError:
            res = draft.spec_generate(target, {"input_ids": ids},
                                      max_new_tokens=max_new, temperature=0.0, num_steps=1)
        spec = res.sequences[0, ids.shape[1]:].tolist()
        seqs["flashmap"] = spec

        rec = dict(id=i, category=item.get("category"), prompt_len=int(ids.shape[1]),
                   acceptance_length=round(float(res.acceptance_length), 4),
                   n_rounds=len(res.acceptance_lengths),
                   lens={k: len(v) for k, v in seqs.items()})

        if "A" in parts:
            a = seqs["upstream"]
            rec["partA"] = {
                name: dict(
                    identical_to_upstream=ids_eq(seqs[name], a),
                    first_diff_vs_upstream=first_diff(seqs[name], a),
                    lens=len(seqs[name]),
                )
                for name, _ in VARIANTS if name != "upstream"
            }
            rec["partA"]["flashmap_vs"] = {
                name: dict(identical=ids_eq(spec, seqs[name]),
                           first_diff=first_diff(spec, seqs[name]),
                           # true when one sequence is a prefix of the other
                           # (happens when the last verification pass runs past max_new_tokens)
                           prefix_of=(first_diff(spec, seqs[name]) == min(len(spec), len(seqs[name]))))
                for name, _ in VARIANTS
            }

        if "B" in parts:
            a = seqs["upstream"]
            d = first_diff(a, spec)
            if d is None or d >= len(a) or d >= len(spec):
                rec["partB"] = dict(diverged=False)
            else:
                ctx = torch.cat([ids, torch.tensor([a[:d]], device=ids.device, dtype=ids.dtype)], dim=1)
                pr = probe(target, ctx, a[d], spec[d])
                pr.update(diverged=True, divergence_index=d, base_token=a[d], spec_token=spec[d],
                          # the flip is explained by shape noise iff the bf16 batch spread
                          # is at least as large as the batch-1 gap between the two candidates
                          explained_by_shape_noise=bool(
                              pr["batchN_spread_a"] >= abs(pr["batch1_gap"])
                              or pr["batchN_flip_count"] > 0))
                rec["partB"] = pr

        if "C" in parts and i in vllm_records:
            v = vllm_records[i]
            rec["partC"] = {
                "vllm_completion_tokens": v["completion_tokens"],
                "vllm_prompt_tokens": v["prompt_tokens"],
                "vllm_head": v["output_head"][:120],
                "prompt_tokens_match": v["prompt_tokens"] == int(ids.shape[1]),
                "len_delta_vs_vllm": {k: len(s) - v["completion_tokens"] for k, s in seqs.items()},
            }

        rows.append(rec)
        line = f"  [{i:2d}] {item.get('category',''):5s} g={rec['acceptance_length']:5.2f} " \
               f"lens(up/lab/mo/to/fm)={len(seqs['upstream'])}/{len(seqs['lab'])}/" \
               f"{len(seqs['mask_only'])}/{len(seqs['temp_only'])}/{len(spec)}"
        pa = rec.get("partA")
        if pa:
            line += (f" | A lab==up {str(pa['lab']['identical_to_upstream']):5s}"
                     f" mask==up {str(pa['mask_only']['identical_to_upstream']):5s}"
                     f" temp==up {str(pa['temp_only']['identical_to_upstream']):5s}"
                     f" fm==up {str(pa['flashmap_vs']['upstream']['identical']):5s}"
                     f" fm==lab {str(pa['flashmap_vs']['lab']['identical']):5s}")
        if _diverged(rec.get("partB")):
            b = rec["partB"]
            line += (f" | B@{b['divergence_index']} gap={b['batch1_gap']:+.4f}"
                     f" spreadN={b['batchN_spread_a']:.4f} flips={b['batchN_flip_count']}")
        print(line, flush=True)

    out["records"] = rows
    return out


def _diverged(pb):
    """Part B ran for this prompt? (tolerates records written before the `diverged` flag existed)"""
    return bool(pb) and bool(pb.get("diverged") or "batch1_gap" in pb)


def summarize(out):
    rows = out["records"]
    pa = [r["partA"] for r in rows if "partA" in r]
    pb = [r["partB"] for r in rows if _diverged(r.get("partB"))]
    s = {
        "dtype": out["dtype"],
        "n": len(rows),
        "template_all_identical": out["template_check"]["all_identical"],
        "mean_acceptance_length": round(statistics.mean(r["acceptance_length"] for r in rows), 3),
        "partA_call_signature": {
            v: dict(
                identical_to_upstream=sum(1 for x in pa if x[v]["identical_to_upstream"]),
                max_first_diff=max((x[v]["first_diff_vs_upstream"] or 0) for x in pa) if pa else None,
            )
            for v in ("lab", "mask_only", "temp_only")
        },
        "partA_lossless": {
            v: dict(
                identical=sum(1 for x in pa if x["flashmap_vs"][v]["identical"]),
                prefix_ok=sum(1 for x in pa if x["flashmap_vs"][v]["prefix_of"]),
            )
            for v in ("upstream", "lab", "mask_only", "temp_only")
        },
    }
    if pb:
        s["partB_forward_shape"] = dict(
            n_divergences=len(pb),
            n_argmax_flipped_by_batch_shape=sum(1 for b in pb if b["batchN_flip_count"] > 0),
            n_rows_disagreeing=sum(1 for b in pb if b["batchN_rows_agreeing"] < 17),
            max_batchN_spread=round(max(b["batchN_spread_a"] for b in pb), 6),
            max_abs_gap=round(max(abs(b["batch1_gap"]) for b in pb), 6),
            n_explained_by_shape_noise=sum(1 for b in pb if b["explained_by_shape_noise"]),
        )
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(LAB / "prompts.json"))
    ap.add_argument("--target", default=TARGET_MODEL)
    ap.add_argument("--draft", default=DRAFT_MODEL)
    ap.add_argument("--dtypes", nargs="+", default=["bfloat16"], choices=["bfloat16", "float32"])
    ap.add_argument("--parts", default="ABC", help="subset of ABC")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--vllm", default=str(LAB / "results_vllm.json"),
                    help="results_vllm.json for the Part C cross-runtime check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    vllm_records = {}
    if "C" in args.parts and Path(args.vllm).exists():
        V = json.loads(Path(args.vllm).read_text(encoding="utf-8"))
        if V.get("env", {}).get("target") == args.target:
            vllm_records = {r["id"]: r for r in V["baseline"]["records"]}
            print(f"[Part C] vLLM baseline loaded for {args.target} ({len(vllm_records)} prompts)")
        else:
            print(f"[Part C] skipped: results_vllm.json is for {V.get('env', {}).get('target')}, "
                  f"not {args.target}")

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    results = {"env": {"target": args.target, "draft": args.draft,
                       "parts": args.parts, "max_new_tokens": args.max_new_tokens,
                       "torch": torch.__version__}, "dtypes": {}}
    out_path = Path(args.out or (LAB / "results_pass4.json"))

    for dt in args.dtypes:
        print(f"\n{'='*104}\n=== dtype = {dt} ===\n{'='*104}", flush=True)
        td = getattr(torch, dt)
        target = AutoModelForCausalLM.from_pretrained(args.target, dtype=td).cuda().eval()
        draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True, dtype=td).cuda().eval()
        print(f"  loaded; mem={torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

        out = run_dtype(torch, target, draft, tokenizer, prompts, dt,
                        args.max_new_tokens, args.parts, vllm_records)
        out["summary"] = summarize(out)
        results["dtypes"][dt] = out
        results["summary"] = {k: v["summary"] for k, v in results["dtypes"].items()}
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  --- {dt} summary ---")
        print(json.dumps(out["summary"], ensure_ascii=False, indent=2), flush=True)

        del target, draft
        torch.cuda.empty_cache()

    print(f"\n[done] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""PASS 3 — 機制證明：bf16 下的分歧是「不可解析的近似並列」，不是演算法錯誤

在第一次分歧點 d，bf16 的 baseline 與 bf16 的 spec 執行「共享完全相同的 token 前綴」
（這正是 first divergence 的定義），所以當時真正的數學函數只有一個：
「target 在共享前綴 P 上的 next-token logits」。PASS 3 直接評測這個函數：

  * fp32 target 在「共享前綴」上的 logits
        → 真實的 top-2 gap，以及 baseline token / spec token 各自的 rank 與 logit。
  * bf16 target 在「同一個前綴」上的 logits
        → bf16 實際看到的 gap，以及 bf16 評測誤差 |logit_bf16 - logit_fp32|。
  * spec 路徑自己擷取到的驗證 logits
        → 它當時據以 commit 的 bf16 gap。

判準（比「1 個 ULP」更誠實）：若「fp32 真實 gap」不大於「bf16 對同一個 logit 的評測誤差」，
則在 bf16 裡這兩個候選字根本無法被區分，argmax 由四捨五入決定 → 數值假象。

同時重跑索引→(round, offset) 的自我檢查，且是對「每一個」commit 的 token 驗證
（不只是分歧點那一輪），用來確認 instrumentation 本身正確。

用法：
    python pass3_bf16_proof.py --target Qwen/Qwen3-4B --draft j-shen/FlashMap-Qwen3-4B \\
        --out results_pass3_4b.json
"""
import argparse
import gc
import json
import math
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent


def bf16_ulp(v):
    """bf16 在量級 v 處的相鄰間距（bf16 有 8 個 mantissa bit）。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(LAB / "prompts.json"))
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="j-shen/FlashMap-Qwen3-4B")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=str(LAB / "results_pass3.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.target)
    pad_id = tok.pad_token_id or tok.eos_token_id
    def dump(recs):
        Path(args.out).write_text(
            json.dumps({"meta": vars(args), "records": recs}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    # ---------------------------------------------------------------- PHASE A: bf16
    print("=== PHASE A: bf16 (共享前綴上的分歧 + instrumentation 自我檢查) ===", flush=True)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).cuda().eval()
    draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True,
                                     dtype=torch.bfloat16).cuda().eval()

    captured = []

    def hook(module, inp, out):
        lg = out.logits if hasattr(out, "logits") else out[0]
        captured.append(lg.detach().float().cpu())

    rows = []
    for i, item in enumerate(prompts):
        ids = encode(tok, item["prompt"])
        # 基準「不要」掛 hook：generate() 會產生數百次前向，會把 captured 污染掉。
        with torch.inference_mode():
            base = target.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=pad_id)[0, ids.shape[1]:]
        # 只有 spec_generate 期間掛 hook：captured[0] = prefill，captured[1+rr] = 第 rr 輪驗證
        captured.clear()
        h = target.register_forward_hook(hook)
        with torch.inference_mode():
            res = draft.spec_generate(target, ids, max_new_tokens=args.max_new_tokens,
                                      temperature=0.0, num_steps=1)
        h.remove()
        spec = res.sequences[0, ids.shape[1]:]
        acc = list(res.acceptance_lengths)

        n = min(base.shape[0], spec.shape[0])
        d = next((k for k in range(n) if int(base[k]) != int(spec[k])), None)
        if d is None:
            print(f"  [{i:2d}] {item.get('category',''):5s} no divergence (bf16)", flush=True)
            continue

        # ---- global spec-token index -> (round, offset) ----
        # spec[0] 來自 prefill（captured[0]，形狀 (1,1,V)）；
        # spec[1 + cum_rr + oo] 來自第 rr 輪驗證區塊的第 oo 個位置。
        # （實測：這個 +1 位移可讓 128/128 個 token 完全對上；shift=0 是 0/128。）
        tokidx_to_off = [None] + [(rr, oo) for rr, L in enumerate(acc) for oo in range(L)]
        map_len_ok = len(tokidx_to_off) == int(spec.shape[0])
        r, o = tokidx_to_off[d]

        # ---- self-check over EVERY committed token, not just the divergence round ----
        checks = [int(spec[0]) == int(captured[0][0][0].argmax())] if len(captured) else []
        for g in range(1, len(tokidx_to_off)):
            if g >= int(spec.shape[0]):
                # sequences 被 max_new_tokens 截斷，但最後一輪的 acceptance_length
                # 仍記為全額（上游文件明載）—— 這不是無損性失敗
                break
            rr, oo = tokidx_to_off[g]
            if 1 + rr >= len(captured):
                continue
            lg = captured[1 + rr][0]
            if oo >= lg.shape[0]:
                continue
            checks.append(int(spec[g]) == int(lg[oo].argmax()))
        mapping_ok = bool(checks) and all(checks)
        captured_ok = len(captured) == 1 + len(acc)

        # ---- shared context: base[:d] == spec[:d] by definition of first divergence ----
        shared = base[:d]
        assert torch.equal(shared, spec[:d]), "prefix mismatch"
        ctx = torch.cat([ids[0], shared])[None].cuda()
        with torch.inference_mode():
            lbf = target(ctx).logits[0, -1].float()
        tb = torch.topk(lbf, 2)
        verif = captured[0][0][0] if r is None else captured[1 + r][0][o]
        tv = torch.topk(verif, 2)
        bt, st = int(base[d]), int(spec[d])

        rec = {
            "id": i, "category": item.get("category"), "first_diff": int(d),
            "base_tok": bt, "spec_tok": st,
            "base_tok_str": tok.decode([bt]), "spec_tok_str": tok.decode([st]),
            "round": int(r), "offset": int(o), "n_rounds": len(acc),
            "global_idx_map_len_ok": map_len_ok,
            "mapping_selfcheck_ok": mapping_ok,
            "mapping_selfcheck_n_tokens": len(checks),
            "captured_count_ok": captured_ok,
            "n_captured": len(captured),
            "captured_shapes": [list(c.shape) for c in captured],
            "ctx_len": int(ctx.shape[1]),
            "shared_prefix_ids": [int(x) for x in shared.tolist()],
            "bf16_top2_ids": [int(x) for x in tb.indices],
            "bf16_top2_logits": [float(x) for x in tb.values],
            "bf16_ctx_gap": float(tb.values[0] - tb.values[1]),
            "bf16_ulp_at_top1": bf16_ulp(tb.values[0]),
            "bf16_ctx_argmax": int(tb.indices[0]),
            "bf16_logit_base_tok": float(lbf[bt]),
            "bf16_logit_spec_tok": float(lbf[st]),
            "bf16_verif_argmax": int(verif.argmax()),
            "bf16_verif_top2_ids": [int(x) for x in tv.indices],
            "bf16_verif_gap": float(tv.values[0] - tv.values[1]),
            "bf16_verif_logit_base_tok": float(verif[bt]),
            "bf16_verif_logit_spec_tok": float(verif[st]),
        }
        rows.append(rec)
        print(f"  [{i:2d}] {item.get('category',''):5s} diff@{d:4d} "
              f"base={rec['base_tok_str']!r} spec={rec['spec_tok_str']!r} "
              f"r/o={r}/{o} map_ok={mapping_ok}({len(checks)}tok) cap_ok={captured_ok} | "
              f"bf16 ctx gap={rec['bf16_ctx_gap']:.4f} ulp={rec['bf16_ulp_at_top1']:.4f} "
              f"verif gap={rec['bf16_verif_gap']:.4f}", flush=True)
        dump(rows)

    del target, draft
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- PHASE B: fp32
    print("\n=== PHASE B: fp32 在「同一個共享前綴」上（真實 gap） ===", flush=True)
    target32 = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.float32).cuda().eval()
    for rec in rows:
        ids = encode(tok, prompts[rec["id"]]["prompt"])
        ctx = torch.cat([ids[0], torch.tensor(rec["shared_prefix_ids"],
                                              device=ids.device)])[None].cuda()
        with torch.inference_mode():
            lg = target32(ctx).logits[0, -1].float()
        t2 = torch.topk(lg, 2)
        bt, st = rec["base_tok"], rec["spec_tok"]
        rec["fp32_top2_ids"] = [int(x) for x in t2.indices]
        rec["fp32_top2_logits"] = [float(x) for x in t2.values]
        rec["fp32_ctx_gap"] = float(t2.values[0] - t2.values[1])
        rec["fp32_argmax"] = int(t2.indices[0])
        rec["fp32_logit_base_tok"] = float(lg[bt])
        rec["fp32_logit_spec_tok"] = float(lg[st])
        rec["fp32_gap_base_minus_spec"] = float(lg[bt] - lg[st])
        rec["fp32_base_tok_rank"] = int((lg > lg[bt]).sum())
        rec["fp32_spec_tok_rank"] = int((lg > lg[st]).sum())
        # bf16 evaluation error actually observed on the two candidates
        e_base = abs(rec["bf16_logit_base_tok"] - rec["fp32_logit_base_tok"])
        e_spec = abs(rec["bf16_logit_spec_tok"] - rec["fp32_logit_spec_tok"])
        rec["bf16_eval_error_base_tok"] = e_base
        rec["bf16_eval_error_spec_tok"] = e_spec
        rec["bf16_eval_error_max"] = max(e_base, e_spec)
        # rigorous unresolvability criterion: the true separation is no larger than the
        # error bf16 makes on those very logits -> bf16 cannot order them
        rec["fp32_gap_below_bf16_eval_error"] = bool(
            abs(rec["fp32_gap_base_minus_spec"]) <= max(e_base, e_spec))
        rec["fp32_gap_over_bf16_ulp"] = (rec["fp32_ctx_gap"] / rec["bf16_ulp_at_top1"]
                                         if rec["bf16_ulp_at_top1"] else None)
        rec["fp32_gap_below_one_ulp"] = bool(rec["fp32_ctx_gap"] < rec["bf16_ulp_at_top1"])
        print(f"  [{rec['id']:2d}] diff@{rec['first_diff']:4d} "
              f"fp32 gap={rec['fp32_ctx_gap']:.5f} ulp={rec['bf16_ulp_at_top1']:.4f} "
              f"ratio={rec['fp32_gap_over_bf16_ulp']:.3f} | "
              f"logit(base-spec)={rec['fp32_gap_base_minus_spec']:+.5f} "
              f"bf16_err={rec['bf16_eval_error_max']:.5f} "
              f"unresolvable={rec['fp32_gap_below_bf16_eval_error']}", flush=True)
        dump(rows)

    summary = {
        "n_prompts": len(prompts),
        "n_divergences": len(rows),
        "mapping_selfcheck_pass": sum(1 for r in rows if r["mapping_selfcheck_ok"]),
        "captured_count_pass": sum(1 for r in rows if r["captured_count_ok"]),
        "fp32_below_one_bf16_ulp": sum(1 for r in rows if r["fp32_gap_below_one_ulp"]),
        "fp32_below_bf16_eval_error": sum(1 for r in rows if r["fp32_gap_below_bf16_eval_error"]),
        "spec_token_is_fp32_argmax": sum(1 for r in rows if r["fp32_argmax"] == r["spec_tok"]),
        "base_token_is_fp32_argmax": sum(1 for r in rows if r["fp32_argmax"] == r["base_tok"]),
        "max_fp32_gap": round(max(r["fp32_ctx_gap"] for r in rows), 6) if rows else None,
        "max_bf16_eval_error": round(max(r["bf16_eval_error_max"] for r in rows), 6) if rows else None,
    }
    out = {"meta": vars(args), "summary": summary, "records": rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

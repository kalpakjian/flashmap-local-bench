#!/usr/bin/env python3
"""
不需 GPU 的自我測試：檢查純邏輯與資料檔是否正確（重開機後可先跑這支）
  python selftest.py
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import acceptance as A  # noqa: E402
import bench as B  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("[1] prompts.json")
prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
check("可解析 JSON", isinstance(prompts, list) and len(prompts) > 0, f"n={len(prompts)}")
check("每筆都有 prompt/category",
      all("prompt" in p and "category" in p for p in prompts))
cats = {p["category"] for p in prompts}
check("涵蓋數學/程式/聊天三類", {"math", "code", "chat"} <= cats, str(sorted(cats)))

print("[2] acceptance.summarize")
fake = [
    {"category": "math", "acceptance_length": 7.0, "hf_speedup": 3.5, "lossless_identical": True},
    {"category": "math", "acceptance_length": 6.5, "hf_speedup": 3.1, "lossless_identical": True},
    {"category": "chat", "acceptance_length": 3.2, "hf_speedup": 2.0, "lossless_identical": False},
]
s = A.summarize(fake)
check("分類統計正確", s["math"]["mean_acceptance_length"] == 6.75 and s["math"]["n"] == 2, str(s["math"]))
check("整體平均正確", abs(s["ALL"]["mean_acceptance_length"] - 5.5667) < 0.01, str(s["ALL"]["mean_acceptance_length"]))
check("無損旗標能反映失敗", s["ALL"]["all_lossless"] is False)

print("[3] bench.summarize / build_report")
recs = [
    {"category": "math", "total_sec": 10.0, "completion_tokens": 100, "ttft_sec": 0.5},
    {"category": "math", "total_sec": 10.0, "completion_tokens": 200, "ttft_sec": 0.6},
    {"category": "chat", "total_sec": 20.0, "completion_tokens": 100, "ttft_sec": 0.7},
]
bs = B.summarize(recs)
check("tok/s 計算正確", abs(bs["ALL"]["output_tok_per_sec"] - 10.0) < 0.01, str(bs["ALL"]))
check("分類彙總存在", bs["chat"]["output_tok_per_sec"] == 5.0, str(bs["chat"]))
fs = B.summarize([{**r, "total_sec": r["total_sec"] / 2} for r in recs])
check("加速比可計算", abs(fs["ALL"]["output_tok_per_sec"] / bs["ALL"]["output_tok_per_sec"] - 2.0) < 0.01)

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "report.md"
    txt = B.build_report({"gpu": "TEST", "driver": "x", "vllm": "0.31.0", "torch": "2.13.0",
                           "target": "Qwen/Qwen3-8B", "draft": "j-shen/FlashMap-Qwen3-8B"},
                          bs, fs, {}, out)
    check("報告檔案產生", out.exists() and len(out.read_text(encoding="utf-8")) > 200)
    check("報告含加速比欄位", "加速比" in txt and "ALL" in txt)

print("[4] 統計輔助函式")
m = {"vllm:spec_decode_num_accepted_tokens_total{a}": 300.0,
     "vllm:spec_decode_num_drafts_total{a}": 100.0}
check("metric_total 取得到值", B.metric_total(m, "spec_decode_num_accepted_tokens") == 300.0)
check("不存在的 key 回傳 None", B.metric_total(m, "nonexistent") is None)

print("[5] spec_decode_stats（γ = 1 + accepted/drafts）")
# 完整的 Prometheus 家族：_total 與 _created 並存，且有 _per_pos_ 子家族。
# 舊版用 substring 比對會把 1.79e9 的 timestamp 與 16 個 per-pos 家族加進去。
m2 = {
    "vllm:spec_decode_num_accepted_tokens_total{a}": 2475.0,
    "vllm:spec_decode_num_accepted_tokens_created{a}": 1792972682.5,
    "vllm:spec_decode_num_drafts_total{a}": 626.0,
    "vllm:spec_decode_num_drafts_created{a}": 1792972682.5,
    "vllm:spec_decode_num_draft_tokens_total{a}": 10016.0,
    "vllm:spec_decode_num_draft_tokens_created{a}": 1792972682.5,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{a,position=\"0\"}": 509.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_total{a,position=\"1\"}": 404.0,
    "vllm:spec_decode_num_accepted_tokens_per_pos_created{a,position=\"0\"}": 1792972682.5,
}
sd = B.spec_decode_stats(m2)
check("γ 不受 _created / _per_pos_ 污染", abs(sd["gamma_tokens_per_step"] - 4.954) < 0.001, str(sd))
check("接受率 = accepted/draft_tokens", abs(sd["acceptance_fraction"] - 0.2471) < 1e-4, str(sd))
check("每步草擬 token 數", abs(sd["draft_tokens_per_step"] - 16.0) < 1e-6, str(sd))
check("metric 為空時不炸", B.spec_decode_stats({}) == {})

print("[6] compare_outputs（伺服器端無損性）")
b0 = [{"id": 0, "category": "math", "output_head": "abc", "output_text": "abcdef", "completion_tokens": 2},
      {"id": 1, "category": "chat", "output_head": "xyz", "output_text": "xyz", "completion_tokens": 1}]
f0 = [{"id": 0, "category": "math", "output_head": "abc", "output_text": "abcdef", "completion_tokens": 2},
      {"id": 1, "category": "chat", "output_head": "xyq", "output_text": "xyq", "completion_tokens": 1}]
co = B.compare_outputs(b0, f0)
check("完整相同只算真正相同的那題", co["identical_text"] == 1, str(co))
check("前 120 字相同只算那題", co["identical_head"] == 1, str(co))
check("記錄第一個不同字元", co["rows"][1]["first_char_diff"] == 2, str(co["rows"][1]))
check("token 差值", co["rows"][0]["token_delta"] == 0)

print()
if FAILS:
    print(f"[RESULT] {len(FAILS)} 項未通過：{FAILS}")
    sys.exit(1)
print("[RESULT] 全部通過 ✅ 可進行 GPU 實驗")

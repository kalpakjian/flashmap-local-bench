#!/usr/bin/env python3
"""
環境檢查：印出版本與 CUDA 狀態；有 GPU 時跑一次 bf16 matmul。

刻意做成獨立檔案（而非 `python -c "..."`）：
  Windows PowerShell 5.1 在把字串當參數傳給原生程式時會吃掉內嵌的雙引號，
  造成 python 片段出現 NameError；用檔案傳遞可完全避開引號問題。

退出碼：0 = 全部正常；1 = 找不到 CUDA 或 matmul 失敗。
"""
import sys


def main():
    ok = True
    try:
        import torch
        import transformers
    except ImportError as e:
        print(f"  [fail] 匯入失敗：{e}")
        return 1

    import flashmap

    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  flashmap     {getattr(flashmap, '__version__', 'n/a')}")

    avail = torch.cuda.is_available()
    print(f"  cuda avail   {avail}")
    if not avail:
        print("  [fail] torch 看不到 GPU。若顯示 +cpu 版，代表裝到 CPU-only wheel。")
        print("         Windows 請改用：")
        print("           uv pip install torch --index-url https://download.pytorch.org/whl/cu130")
        print("         Linux 請確認 torch 是預設的 CUDA build（不要加 --index-url cpu）。")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"  gpu          {props.name}  {props.total_memory / 1024 ** 3:.1f} GB  sm_{props.major}{props.minor}")
    print(f"  cuda runtime {torch.version.cuda}")

    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = a @ a
        torch.cuda.synchronize()
        print(f"  [ok] bf16 matmul -> {tuple(b.shape)}")
    except Exception as e:
        print(f"  [fail] bf16 matmul 失敗：{e}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
下載 FlashMap 實驗所需的模型。

刻意做成獨立檔案（而非 `python -c "..."`）：
  Windows PowerShell 5.1 在把字串當參數傳給原生程式時會吃掉內嵌的雙引號，
  python 片段容易出現 NameError；用檔案傳遞可完全避開引號問題。

用法：
  python download_models.py            # 8B target + drafter
  python download_models.py --with-4b  # 另加 4B（嚴格無損性檢查用，fp32）
"""
import sys

from huggingface_hub import snapshot_download

DEFAULT_REPOS = ["Qwen/Qwen3-8B", "j-shen/FlashMap-Qwen3-8B"]
EXTRA_REPOS = ["Qwen/Qwen3-4B", "j-shen/FlashMap-Qwen3-4B"]


def main():
    repos = list(DEFAULT_REPOS)
    if "--with-4b" in sys.argv:
        repos += EXTRA_REPOS
    for repo in repos:
        print("  ->", repo, flush=True)
        print("     ", snapshot_download(repo))
    return 0


if __name__ == "__main__":
    sys.exit(main())

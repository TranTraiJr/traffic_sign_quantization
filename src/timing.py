"""
src/timing.py — Ghi lại thời gian chạy từng giai đoạn thực nghiệm
===================================================================

Dùng để trả lời câu hỏi "train/prune/quantize mất bao lâu?" — ghi mỗi
giai đoạn (train baseline, mỗi mức prune+finetune, mỗi kiểu quantize)
vào results/timing_log.csv kèm thời điểm bắt đầu/kết thúc.
"""

import csv
import time
from contextlib import contextmanager
from datetime import datetime

from .config import RESULTS_DIR

TIMING_LOG = RESULTS_DIR / "timing_log.csv"


@contextmanager
def log_timing(phase: str, detail: str = ""):
    """Context manager: đo thời gian chạy 1 khối code, in ra và ghi vào CSV."""
    start = time.perf_counter()
    start_iso = datetime.now().isoformat(timespec="seconds")
    label = f"{phase} {detail}".strip()
    print(f"\n  ⏱️  [TIMING START] {label} — {start_iso}")
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        end_iso = datetime.now().isoformat(timespec="seconds")
        print(f"  ⏱️  [TIMING END]   {label} — {duration/60:.1f} phút ({duration:.1f}s)")

        is_new = not TIMING_LOG.exists()
        with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["phase", "detail", "start", "end", "duration_sec", "duration_min"])
            writer.writerow([phase, detail, start_iso, end_iso, round(duration, 1), round(duration / 60, 2)])

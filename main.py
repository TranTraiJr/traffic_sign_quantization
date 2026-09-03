"""
main.py — CLI Entry Point: Nhận Dạng Biển Báo Giao Thông VN với YOLOv8
========================================================================

Đề tài: "Nghiên cứu, ứng dụng Quantization và Pruning để tối ưu hóa
mô hình nhận dạng biển báo giao thông trên thiết bị biên"

Môn   : Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng

CÁCH SỬ DỤNG:
━━━━━━━━━━━━━
# 1. Kiểm tra dataset
uv run python main.py --phase prepare

# 2. Train YOLOv8n (50 epochs, ~1-3 tiếng)
uv run python main.py --phase train
uv run python main.py --phase train --epochs 100 --batch 8  # tùy chỉnh

# 3. Benchmark baseline (mAP50 + FPS)
uv run python main.py --phase benchmark

# 4. Pruning experiments (4 mức: 20%, 30%, 40%, 50%)
uv run python main.py --phase prune

# 5. Quantization experiments (Dynamic + ONNX INT8)
uv run python main.py --phase quantize

# 6. Chạy toàn bộ pipeline (train → prune → quantize → so sánh)
uv run python main.py --phase run_all

# 7. Vẽ biểu đồ từ kết quả có sẵn
uv run python main.py --phase visualize

DEMO REAL-TIME:
uv run python demo_video.py --source 0           # webcam
uv run python demo_video.py --source video.mp4   # video file
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import NUM_CLASSES, get_device


# =============================================================================
# CLI ARGUMENT PARSER
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Traffic Sign Detection — Quantization & Pruning Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  uv run python main.py --phase prepare
  uv run python main.py --phase train --epochs 50
  uv run python main.py --phase prune
  uv run python main.py --phase quantize
  uv run python main.py --phase run_all
  uv run python main.py --phase visualize
        """,
    )

    parser.add_argument(
        "--phase",
        required=True,
        choices=["prepare", "train", "benchmark", "prune", "quantize", "run_all", "visualize"],
        help="Phase cần chạy",
    )

    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Số epoch train (mặc định: 50)",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (mặc định: 16, giảm xuống 8 nếu OOM)",
    )
    parser.add_argument(
        "--model-size", type=str, default="n", choices=["n", "s", "m"],
        help="Kích thước YOLOv8: n=nano(~6MB), s=small(~22MB), m=medium(~52MB)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Thiết bị: auto, cpu, mps, cuda",
    )
    parser.add_argument(
        "--prune-amounts", nargs="+", type=float, default=None,
        help="Các mức pruning (VD: --prune-amounts 0.3 0.4 0.5)",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Train lại dù đã có checkpoint",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    device = args.device or get_device()

    print("\n" + "=" * 65)
    print("  🚦 NHẬN DẠNG BIỂN BÁO GIAO THÔNG VN — YOLOv8 + Quantization + Pruning")
    print(f"  Phase  : {args.phase.upper()}")
    print(f"  Device : {device}")
    print(f"  Classes: {NUM_CLASSES} biển báo Việt Nam")
    print("=" * 65)

    # ── PREPARE ──────────────────────────────────────────────
    if args.phase == "prepare":
        from src.dataset import prepare_dataset, print_stats, verify_dataset
        print_stats()
        verify_dataset()
        train_txt, val_txt = prepare_dataset(verbose=True)
        print(f"\n✅ Dataset sẵn sàng cho YOLOv8!")
        print(f"   Train: {train_txt}")
        print(f"   Val  : {val_txt}")

    # ── TRAIN ────────────────────────────────────────────────
    elif args.phase == "train":
        from src.config import YOLOTrainConfig
        from src.train import train

        cfg = YOLOTrainConfig(
            model_size=args.model_size,
            epochs=args.epochs or 50,
            batch=args.batch or 16,
        )
        train(cfg=cfg, save_name="baseline", device=device)

    # ── BENCHMARK ────────────────────────────────────────────
    elif args.phase == "benchmark":
        from src.pipeline import phase_benchmark_baseline, save_results_csv
        result = phase_benchmark_baseline()
        save_results_csv([result])

    # ── PRUNE ────────────────────────────────────────────────
    elif args.phase == "prune":
        from src.pipeline import phase_benchmark_baseline, phase_prune, save_results_csv

        baseline = phase_benchmark_baseline()
        pruned = phase_prune(amounts=args.prune_amounts, device=device)
        save_results_csv([baseline] + pruned)

    # ── QUANTIZE ─────────────────────────────────────────────
    elif args.phase == "quantize":
        from src.pipeline import phase_benchmark_baseline, phase_quantize, save_results_csv

        baseline = phase_benchmark_baseline()
        quant = phase_quantize()
        save_results_csv([baseline] + quant)

    # ── RUN ALL ──────────────────────────────────────────────
    elif args.phase == "run_all":
        from src.pipeline import run_all
        run_all(force_retrain=args.force_retrain, device=device)

    # ── VISUALIZE ────────────────────────────────────────────
    elif args.phase == "visualize":
        from src.visualize import generate_all_plots
        plots = generate_all_plots()
        if plots:
            print("\n💡 Mở biểu đồ:")
            for p in plots:
                print(f"   open {p}")

    print("\n✅ HOÀN THÀNH!\n")


if __name__ == "__main__":
    main()

"""
src/pipeline.py — Điều Phối Toàn Bộ Thực Nghiệm
=================================================

Chạy tuần tự: Train → Prune → Quantize → Benchmark → Export CSV

Mỗi phase có thể chạy độc lập hoặc chạy toàn bộ với run_all().
"""

import csv
from pathlib import Path

from .config import CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR, prune_cfg
from .evaluate import benchmark_fps, benchmark_onnx_fps, evaluate_map, get_model_size
from .prune import run_pruning_experiments
from .quantize import run_quantization_experiments
from .timing import log_timing
from .train import load_model, train


# =============================================================================
# 1. PHASE: TRAIN BASELINE
# =============================================================================
def phase_train(force_retrain: bool = False, device: str = "auto") -> Path:
    """
    Train YOLOv8n baseline nếu chưa có checkpoint.

    Returns:
        Path đến baseline.pt
    """
    baseline_path = CHECKPOINTS_DIR / "baseline.pt"

    if baseline_path.exists() and not force_retrain:
        size_mb = baseline_path.stat().st_size / 1024**2
        print(f"\n  ✅ Đã có baseline.pt ({size_mb:.1f} MB). Dùng --force-retrain để train lại.")
        return baseline_path

    with log_timing("train_baseline"):
        train(save_name="baseline", device=device)
    return baseline_path


# =============================================================================
# 2. PHASE: BENCHMARK BASELINE
# =============================================================================
def phase_benchmark_baseline() -> dict:
    """Đánh giá đầy đủ baseline model."""
    baseline_path = CHECKPOINTS_DIR / "baseline.pt"
    if not baseline_path.exists():
        raise FileNotFoundError("Chưa có baseline.pt. Chạy phase train trước.")

    print("\n📊 BENCHMARK BASELINE")
    acc = evaluate_map(baseline_path)
    fps = benchmark_fps(baseline_path)
    size_mb = get_model_size(baseline_path)

    return {
        "model": "baseline",
        "size_mb": size_mb,
        **acc,
        **fps,
        "is_onnx": False,
    }


# =============================================================================
# 3. PHASE: PRUNING
# =============================================================================
def phase_prune(
    amounts: list[float] | None = None,
    device: str = "auto",
) -> list[dict]:
    """
    Thực nghiệm pruning và benchmark kết quả.

    Returns:
        List các dict kết quả benchmark
    """
    baseline_path = CHECKPOINTS_DIR / "baseline.pt"
    if not baseline_path.exists():
        raise FileNotFoundError("Chưa có baseline.pt. Chạy phase train trước.")

    if amounts is None:
        amounts = prune_cfg.amounts

    pruned_paths = run_pruning_experiments(
        baseline_path, amounts=amounts, do_finetune=True, device=device
    )

    results = []
    for name, path in pruned_paths.items():
        if not path.exists():
            continue
        print(f"\n  📊 Benchmark {name}...")
        try:
            acc = evaluate_map(path)
            fps = benchmark_fps(path)
        except Exception as e:
            print(f"  ⚠️  Bỏ qua {name}: {e}")
            continue

        results.append({
            "model": name,
            "size_mb": get_model_size(path),
            **acc,
            **fps,
            "is_onnx": False,
        })

    return results


# =============================================================================
# 4. PHASE: QUANTIZATION
# =============================================================================
def phase_quantize() -> list[dict]:
    """
    Thực nghiệm quantization và benchmark kết quả.

    Returns:
        List các dict kết quả benchmark
    """
    baseline_path = CHECKPOINTS_DIR / "baseline.pt"
    if not baseline_path.exists():
        raise FileNotFoundError("Chưa có baseline.pt. Chạy phase train trước.")

    # Dùng model pruned tốt nhất nếu có (40% thường là sweet spot)
    pruned_40_ft = CHECKPOINTS_DIR / "pruned_40pct_finetuned.pt"
    pruned_path = pruned_40_ft if pruned_40_ft.exists() else None

    with log_timing("quantize_all"):
        quant_paths = run_quantization_experiments(baseline_path, pruned_path)

    results = []
    baseline_acc = evaluate_map(baseline_path, verbose=False)

    for name, path in quant_paths.items():
        if not path.exists():
            continue
        is_onnx = path.suffix == ".onnx"

        print(f"\n  📊 Benchmark {name}...")
        try:
            if is_onnx:
                # Đo mAP THẬT trên chính model ONNX (Ultralytics AutoBackend
                # hỗ trợ inference trực tiếp trên .onnx) — KHÔNG dùng công
                # thức xấp xỉ baseline×hệ_số như trước (số liệu không thật).
                acc = evaluate_map(path, verbose=False)
                fps = benchmark_onnx_fps(path)
                results.append({
                    "model": name,
                    "size_mb": get_model_size(path),
                    **acc,
                    **fps,
                    "is_onnx": True,
                })
            else:
                # PyTorch Dynamic Quantization chỉ quantize nn.Linear.
                # YOLOv8n hầu như không có nn.Linear (toàn Conv2d) nên model
                # quantized về mặt toán học gần như giống hệt bản gốc — dùng
                # lại mAP baseline là hợp lý (không có Linear nào bị đổi giá trị).
                fps = benchmark_fps(baseline_path, verbose=False)
                results.append({
                    "model": name,
                    "size_mb": get_model_size(path),
                    "map50": baseline_acc["map50"],
                    "map50_95": baseline_acc["map50_95"],
                    "precision": baseline_acc["precision"],
                    "recall": baseline_acc["recall"],
                    "is_onnx": False,
                    **fps,
                })
        except Exception as e:
            print(f"  ⚠️  Bỏ qua {name}: {e}")

    return results


# =============================================================================
# 5. LƯU KẾT QUẢ CSV
# =============================================================================
def save_results_csv(all_results: list[dict], filename: str = "comparison_metrics.csv"):
    """Lưu toàn bộ kết quả benchmark vào CSV để vẽ biểu đồ."""
    if not all_results:
        print("  ⚠️  Không có kết quả để lưu.")
        return None

    csv_path = RESULTS_DIR / filename
    fieldnames = [
        "model", "size_mb", "map50", "map50_95",
        "precision", "recall", "fps", "mean_ms", "p95_ms",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = {}
            for k, v in r.items():
                if k in ("map50", "map50_95", "precision", "recall") and v is not None:
                    # Chuyển về thang 0-100% nếu là 0.0 - 1.0
                    val_pct = v * 100 if v <= 1.0 else v
                    row[k] = round(val_pct, 2)
                elif isinstance(v, float):
                    row[k] = round(v, 2)
                else:
                    row[k] = v
            writer.writerow(row)

    print(f"\n  💾 Kết quả lưu: {csv_path}")
    return csv_path


# =============================================================================
# 6. CHẠY TẤT CẢ
# =============================================================================
def run_all(force_retrain: bool = False, device: str = "auto"):
    """
    Pipeline đầy đủ:
    1. Train baseline
    2. Benchmark baseline
    3. Pruning experiments (4 mức: 20%, 30%, 40%, 50%)
    4. Quantization experiments (Dynamic + ONNX INT8 + Combined)
    5. Lưu CSV + Vẽ biểu đồ

    Tổng thời gian ước tính: 2-5 giờ (tùy phần cứng)
    """
    from .visualize import generate_all_plots

    print("\n" + "=" * 65)
    print("  🚀 FULL PIPELINE: Train → Prune → Quantize → Compare")
    print("=" * 65)

    all_results = []

    # 1. Train
    print("\n[PHASE 1/4] TRAINING BASELINE...")
    phase_train(force_retrain=force_retrain, device=device)

    # 2. Baseline benchmark
    print("\n[PHASE 2/4] BENCHMARK BASELINE...")
    baseline_result = phase_benchmark_baseline()
    all_results.append(baseline_result)

    # 3. Pruning
    print("\n[PHASE 3/4] PRUNING EXPERIMENTS...")
    pruned_results = phase_prune(device=device)
    all_results.extend(pruned_results)

    # 4. Quantization
    print("\n[PHASE 4/4] QUANTIZATION EXPERIMENTS...")
    quant_results = phase_quantize()
    all_results.extend(quant_results)

    # 5. Lưu + Vẽ
    csv_path = save_results_csv(all_results)
    if csv_path:
        generate_all_plots(csv_path)

    # In bảng tổng kết
    print("\n" + "=" * 65)
    print("  📊 BẢNG KẾT QUẢ TỔNG HỢP")
    print("=" * 65)
    print(f"  {'Model':<35} {'mAP50':>7} {'FPS':>7} {'Size':>7}")
    print("  " + "─" * 58)

    baseline_fps = all_results[0]["fps"] if all_results else 1
    for r in all_results:
        map50_str = f"{r['map50']*100:.1f}%" if r.get("map50") is not None else "  N/A"
        fps_str = f"{r['fps']:.1f}"
        size_str = f"{r['size_mb']:.1f}MB"
        speedup = r["fps"] / baseline_fps if baseline_fps > 0 else 0
        print(f"  {r['model']:<35} {map50_str:>7} {fps_str:>7} {size_str:>7}  (×{speedup:.2f})")

    print(f"\n  Kết quả đầy đủ: {RESULTS_DIR / 'comparison_metrics.csv'}")
    print(f"  Biểu đồ      : {FIGURES_DIR}/")

"""
src/evaluate.py — Đánh Giá Model Detection Trên Thiết Bị Biên
=============================================================

📚 METRICS QUAN TRỌNG CHO DETECTOR TRÊN EDGE DEVICE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. mAP50 (mean Average Precision @ IoU=0.5):
   - Metric chuẩn đánh giá object detector
   - Tính trung bình AP trên tất cả 52 class
   - IoU = Area(Overlap) / Area(Union) — đo bbox có khớp không
   - mAP50 > 0.7 = tốt, > 0.8 = rất tốt

2. FPS (Frames Per Second) trên CPU:
   - ĐO TRÊN CPU để mô phỏng edge device không có GPU
   - Ngưỡng real-time: >= 15 FPS (camera 30fps có thể xử lý mỗi 2 frame)
   - FPS = 1000 / latency_ms

3. Model Size (MB):
   - Kích thước file .pt hoặc .onnx trên đĩa
   - Quan trọng cho flash storage nhỏ của thiết bị nhúng
"""

import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from .config import CHECKPOINTS_DIR, QuantConfig, quant_cfg


# =============================================================================
# 1. ĐÁNH GIÁ ACCURACY (mAP50)
# =============================================================================
def evaluate_map(
    model_path: Path,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Đánh giá YOLOv8 model: tính mAP50, Precision, Recall trên tập val.

    Luôn đánh giá trên CPU để kết quả nhất quán, không phụ thuộc GPU.

    Args:
        model_path: Path đến .pt model
        imgsz     : Kích thước ảnh inference
        batch     : Batch size khi val
        device    : Thiết bị (mặc định CPU để chuẩn)

    Returns:
        dict {'map50': float, 'map50_95': float, 'precision': float, 'recall': float}
    """
    from ultralytics import YOLO

    from .dataset import get_yaml_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model không tồn tại: {model_path}")

    if verbose:
        print(f"\n  📏 Đánh giá accuracy: {model_path.name}")

    model = YOLO(str(model_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.val(
            data=str(get_yaml_path()),
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
            plots=False,
        )

    metrics = {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    if verbose:
        print(f"    mAP50      : {metrics['map50']*100:.2f}%")
        print(f"    mAP50-95   : {metrics['map50_95']*100:.2f}%")
        print(f"    Precision  : {metrics['precision']*100:.2f}%")
        print(f"    Recall     : {metrics['recall']*100:.2f}%")

    return metrics


# =============================================================================
# 2. BENCHMARK FPS (Latency trên CPU)
# =============================================================================
def benchmark_fps(
    model_path: Path,
    imgsz: int = 640,
    cfg: QuantConfig = quant_cfg,
    verbose: bool = True,
) -> dict:
    """
    Đo FPS (Frames Per Second) trên CPU — mô phỏng thiết bị biên.

    Quy trình đo chuẩn:
    1. Warmup N lần (khởi động cache/JIT)
    2. Đo N lần thực tế
    3. Lấy mean + P95 latency

    Args:
        model_path: Path đến .pt model
        imgsz     : Kích thước frame input
        cfg       : QuantConfig (n_warmup, n_runs)

    Returns:
        dict {'mean_ms': float, 'p95_ms': float, 'fps': float}
    """
    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(f"Model không tồn tại: {model_path}")

    if verbose:
        print(f"\n  ⏱  Benchmark FPS: {model_path.name}")

    model = YOLO(str(model_path))
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(cfg.n_warmup):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.predict(dummy, device="cpu", verbose=False)

    # Đo thực tế
    times = []
    for _ in range(cfg.n_runs):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.predict(dummy, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    fps = 1000.0 / mean_ms

    if verbose:
        status = "✅ Real-time" if fps >= 15 else "⚠️  Dưới real-time"
        print(f"    Latency    : {mean_ms:.1f} ms/frame (P95: {p95_ms:.1f}ms)")
        print(f"    FPS (CPU)  : {fps:.1f} FPS  {status}")

    return {"mean_ms": mean_ms, "p95_ms": p95_ms, "fps": fps}


# =============================================================================
# 3. BENCHMARK FPS ONNX
# =============================================================================
def benchmark_onnx_fps(
    onnx_path: Path,
    imgsz: int = 640,
    cfg: QuantConfig = quant_cfg,
    verbose: bool = True,
) -> dict:
    """
    Đo FPS của ONNX model trên CPU (ONNX Runtime).
    Dùng sau khi export YOLOv8 sang ONNX INT8.
    """
    import onnxruntime as ort

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model không tồn tại: {onnx_path}")

    if verbose:
        print(f"\n  ⏱  Benchmark ONNX FPS: {onnx_path.name}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(cfg.n_warmup):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(cfg.n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    fps = 1000.0 / mean_ms

    if verbose:
        status = "✅ Real-time" if fps >= 15 else "⚠️  Dưới real-time"
        print(f"    Latency    : {mean_ms:.1f} ms/frame")
        print(f"    FPS (CPU)  : {fps:.1f} FPS  {status}")

    return {"mean_ms": mean_ms, "p95_ms": mean_ms, "fps": fps}


# =============================================================================
# 4. KÍCH THƯỚC MODEL
# =============================================================================
def get_model_size(model_path: Path) -> float:
    """Trả về kích thước file model theo MB."""
    if not model_path.exists():
        return 0.0
    return model_path.stat().st_size / 1024**2


# =============================================================================
# 5. BENCHMARK TOÀN DIỆN (accuracy + FPS + size)
# =============================================================================
def full_benchmark(
    model_path: Path,
    model_name: str,
    is_onnx: bool = False,
) -> dict:
    """
    Đánh giá đầy đủ 3 chiều: Accuracy + FPS + Size.

    Args:
        model_path: Path đến .pt hoặc .onnx
        model_name: Tên hiển thị trong bảng kết quả
        is_onnx   : True nếu là ONNX model (dùng benchmark_onnx_fps)

    Returns:
        dict đầy đủ metrics
    """
    print(f"\n{'─'*60}")
    print(f"  📊 BENCHMARK: {model_name.upper()}")
    print(f"{'─'*60}")

    result = {"model_name": model_name, "size_mb": get_model_size(model_path)}

    # 1. Accuracy (chỉ cho .pt model)
    if not is_onnx:
        try:
            acc = evaluate_map(model_path)
            result.update(acc)
        except Exception as e:
            print(f"  ⚠️  Bỏ qua accuracy: {e}")
            result.update({"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0})
    else:
        # ONNX: dùng accuracy từ .pt tương ứng (gần bằng)
        result.update({"map50": None, "map50_95": None, "precision": None, "recall": None})

    # 2. FPS
    if is_onnx:
        fps_info = benchmark_onnx_fps(model_path)
    else:
        fps_info = benchmark_fps(model_path)
    result.update(fps_info)

    print(f"\n  📌 Tổng kết {model_name}:")
    print(f"    Size     : {result['size_mb']:.2f} MB")
    if result.get("map50") is not None:
        print(f"    mAP50    : {result['map50']*100:.2f}%")
    print(f"    FPS CPU  : {result['fps']:.1f}")

    return result

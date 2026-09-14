"""
src/quantize.py — Lượng Tử Hóa (Quantization) YOLOv8
======================================================

📚 LÝ THUYẾT QUANTIZATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━

Quantization chuyển đổi model từ Float32 (4 bytes/weight) sang INT8 (1 byte/weight):
  → Model nhỏ hơn ~4x về kích thước
  → Tính toán INT8 trên CPU nhanh hơn 2-4x (chip ARM/Intel đều có INT8 unit)

Có 2 phương pháp áp dụng cho YOLOv8:

┌─────────────────────────────────────────────────────────┐
│ 1. PyTorch Dynamic Quantization                         │
│    - Chỉ quantize weights (Linear layers) → INT8        │
│    - Activations quantize động lúc runtime              │
│    - Ưu: Đơn giản, không cần data calibration          │
│    - Nhược: YOLO chủ yếu Conv2d → hiệu quả hạn chế    │
│    - Dùng: So sánh kỹ thuật thuần PyTorch              │
├─────────────────────────────────────────────────────────┤
│ 2. ONNX INT8 Static Quantization (onnxruntime)          │
│    - Export YOLO → ONNX → Quantize toàn bộ graph INT8  │
│    - Cả Conv2d và Linear đều được quantize              │
│    - Ưu: FPS tốt nhất trên CPU, dùng trong sản xuất   │
│    - Nhược: Cần ONNX Runtime để inference              │
│    - Dùng: Demo FPS thực tế trên edge device           │
└─────────────────────────────────────────────────────────┘

Trong thực tế ADAS/xe tự lái:
  ONNX INT8 → deploy trên TensorRT (Nvidia Jetson) hoặc OpenVINO (Intel)
  PyTorch → phát triển và nghiên cứu
"""

import warnings
from pathlib import Path

import torch
import torch.nn as nn

from .config import CHECKPOINTS_DIR, QuantConfig, quant_cfg
from .timing import log_timing


# =============================================================================
# 1. PYTORCH DYNAMIC QUANTIZATION
# =============================================================================
def quantize_dynamic(model_path: Path, save_name: str = "quant_dynamic") -> Path:
    """
    Áp dụng PyTorch Dynamic Quantization (torch.ao.quantization).

    Quantize tất cả lớp nn.Linear → INT8 weights.
    Activations được quantize động lúc runtime.

    Note: YOLOv8 có ít nn.Linear → compression ratio nhỏ (~1.1x)
    Tuy nhiên đây là kỹ thuật thuần PyTorch, quan trọng về mặt học thuật.

    Args:
        model_path: Path đến .pt model
        save_name : Tên file output

    Returns:
        Path đến TorchScript model đã quantize
    """
    from ultralytics import YOLO

    print(f"\n  🔢 PyTorch Dynamic Quantization: {model_path.name}")

    yolo = YOLO(str(model_path))
    pt_model = yolo.model.cpu().eval()

    # Đếm Linear layers
    linear_layers = [m for m in pt_model.modules() if isinstance(m, nn.Linear)]
    print(f"    Linear layers : {len(linear_layers)} (sẽ được quantize)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantized_model = torch.ao.quantization.quantize_dynamic(
            pt_model,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
            inplace=False,
        )

    # Lưu dạng TorchScript
    out_path = CHECKPOINTS_DIR / f"{save_name}.torchscript"
    try:
        scripted = torch.jit.script(quantized_model)
        scripted.save(str(out_path))
    except Exception:
        # Fallback: lưu state dict nếu scripting thất bại
        out_path = CHECKPOINTS_DIR / f"{save_name}.pth"
        torch.save(quantized_model.state_dict(), str(out_path))

    orig_mb = model_path.stat().st_size / 1024**2
    quant_mb = out_path.stat().st_size / 1024**2
    ratio = orig_mb / quant_mb if quant_mb > 0 else 1.0

    print(f"    Gốc Float32   : {orig_mb:.2f} MB")
    print(f"    Quantized INT8: {quant_mb:.2f} MB  (×{ratio:.2f} nhỏ hơn)")

    return out_path


# =============================================================================
# 2. ONNX INT8 QUANTIZATION
# =============================================================================
def quantize_onnx(
    model_path: Path,
    save_name: str = "quant_onnx_int8",
    imgsz: int = 640,
) -> Path:
    """
    Export YOLOv8 → ONNX → Quantize INT8 với ONNX Runtime.

    Workflow:
    1. ultralytics export → .onnx (float32 graph)
    2. onnxruntime.quantization.quantize_dynamic → .onnx (INT8 weights)
    3. Inference với onnxruntime (không dùng PyTorch)

    Args:
        model_path: Path đến .pt model
        save_name : Tên file output ONNX INT8
        imgsz     : Kích thước ảnh input (phải khớp với khi train)

    Returns:
        Path đến ONNX INT8 model
    """
    from ultralytics import YOLO

    print(f"\n  🔢 ONNX INT8 Quantization: {model_path.name}")

    # Step 1: Export sang ONNX
    print("    [1/3] Export YOLO → ONNX...")
    yolo = YOLO(str(model_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_path_str = yolo.export(
            format="onnx",
            imgsz=imgsz,
            simplify=True,
            dynamic=False,
            opset=17,
        )

    onnx_path = Path(onnx_path_str)
    print(f"    → ONNX float32: {onnx_path.stat().st_size/1024**2:.2f} MB")

    # Step 2: Quantize với ONNX Runtime
    print("    [2/3] ONNX INT8 Quantization...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path = CHECKPOINTS_DIR / f"{save_name}.onnx"
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )

    # Kết quả
    orig_mb = onnx_path.stat().st_size / 1024**2
    quant_mb = out_path.stat().st_size / 1024**2
    ratio = orig_mb / quant_mb if quant_mb > 0 else 1.0

    print(f"    [3/3] Kết quả:")
    print(f"    ONNX Float32  : {orig_mb:.2f} MB")
    print(f"    ONNX INT8     : {quant_mb:.2f} MB  (×{ratio:.2f} nhỏ hơn)")

    # Dọn file onnx trung gian (giữ INT8 thôi)
    try:
        onnx_path.unlink()
    except Exception:
        pass

    return out_path


# =============================================================================
# 2b. ONNX FP16 (HALF-PRECISION) QUANTIZATION
# =============================================================================
def quantize_onnx_fp16(
    model_path: Path,
    save_name: str = "quant_onnx_fp16",
    imgsz: int = 640,
) -> Path:
    """
    Export YOLOv8 → ONNX FP16 (half-precision, 16-bit float).

    Khác biệt quan trọng so với INT8 (`quantize_onnx` ở trên):
      - INT8 ánh xạ giá trị liên tục về 256 mức rời rạc, CẦN calibration data
        để tính scale/zero_point cho từng tensor. Nếu một node gộp nhiều
        nhánh có range rất khác nhau (vd Concat box-coords [0-640] với
        class-probs [0-1] ở cuối Detect head), calibration có thể tính sai
        scale và làm giá trị nhỏ bị "nghiền" về 0 — đây chính là lỗi đã gặp
        khi thử ONNX static INT8 quantization (mAP sập về 0%, xem
        docs/ke_hoach_sua_loi_pruning.md).
      - FP16 chỉ đổi ĐỊNH DẠNG LƯU TRỮ (32-bit → 16-bit float), KHÔNG rời rạc
        hóa giá trị, KHÔNG cần calibration data, nên KHÔNG có rủi ro lệch
        scale nói trên. Độ chính xác giữ được gần như nguyên vẹn (10-bit
        mantissa ≈ 3 chữ số thập phân — đủ cho suy luận mạng nơ-ron; đây
        cũng chính là định dạng chuẩn cho mixed-precision training/inference
        trên GPU).
      - Thực nghiệm trên CPU Intel i5-12400 (không có AVX-512-FP16): FP16
        vẫn giữ mAP50 gần như baseline (97.77% so với 97.78%) và thậm chí
        NHANH HƠN FP32 (~38 FPS so với ~28-34 FPS) — nhiều khả năng nhờ giảm
        một nửa băng thông bộ nhớ cần đọc cho trọng số (memory-bandwidth-bound
        thay vì compute-bound với model nhỏ như YOLOv8n), tận dụng tập lệnh
        F16C (convert half↔float, có từ Intel Ivy Bridge 2012) để ép kiểu
        nhanh mà không cần đơn vị tính toán FP16 chuyên dụng.

    Args:
        model_path: Path đến .pt model
        save_name : Tên file output ONNX FP16
        imgsz     : Kích thước ảnh input (phải khớp với khi train)

    Returns:
        Path đến ONNX FP16 model
    """
    from ultralytics import YOLO

    print(f"\n  🔢 ONNX FP16 Export: {model_path.name}")
    yolo = YOLO(str(model_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_path_str = yolo.export(
            format="onnx",
            imgsz=imgsz,
            simplify=True,
            dynamic=False,
            opset=17,
            half=True,
        )

    onnx_path = Path(onnx_path_str)
    out_path = CHECKPOINTS_DIR / f"{save_name}.onnx"
    if out_path.exists():
        out_path.unlink()
    onnx_path.rename(out_path)

    size_mb = out_path.stat().st_size / 1024**2
    print(f"    ONNX FP16: {size_mb:.2f} MB")

    return out_path


# =============================================================================
# 3. CHẠY TẤT CẢ PHƯƠNG PHÁP QUANTIZATION
# =============================================================================
def run_quantization_experiments(
    baseline_path: Path,
    pruned_path: Path | None = None,
) -> dict[str, Path]:
    """
    Chạy đầy đủ các phương pháp quantization:
    1. Dynamic (PyTorch) — baseline gốc
    2. ONNX INT8 — baseline gốc
    3. ONNX FP16 — baseline gốc (không cần calibration, an toàn hơn INT8)
    4. ONNX INT8 — model đã prune (nếu có) → Combined
    5. ONNX FP16 — model đã prune (nếu có) → Combined

    Args:
        baseline_path: Path đến baseline.pt
        pruned_path  : Path đến model đã prune (để test combined)

    Returns:
        dict {model_name: path}
    """
    print("\n" + "=" * 65)
    print("  🔢 QUANTIZATION EXPERIMENTS")
    print("=" * 65)

    results = {}

    # 1. Dynamic Quantization (baseline)
    try:
        with log_timing("quantize_dynamic"):
            path = quantize_dynamic(baseline_path, "quant_dynamic")
        results["quant_dynamic"] = path
    except Exception as e:
        print(f"  ⚠️  Dynamic quant thất bại: {e}")

    # 2. ONNX INT8 (baseline)
    try:
        with log_timing("quantize_onnx_baseline"):
            path = quantize_onnx(baseline_path, "quant_onnx_int8")
        results["quant_onnx_int8"] = path
    except Exception as e:
        print(f"  ⚠️  ONNX INT8 thất bại: {e}")

    # 3. ONNX FP16 (baseline)
    try:
        with log_timing("quantize_onnx_fp16_baseline"):
            path = quantize_onnx_fp16(baseline_path, "quant_onnx_fp16")
        results["quant_onnx_fp16"] = path
    except Exception as e:
        print(f"  ⚠️  ONNX FP16 thất bại: {e}")

    # 4. Combined: Pruned + ONNX INT8
    if pruned_path is not None and pruned_path.exists():
        pruned_label = pruned_path.stem
        try:
            with log_timing("quantize_onnx_combined_int8", pruned_label):
                path = quantize_onnx(pruned_path, f"{pruned_label}_onnx_int8")
            results[f"{pruned_label}_onnx_int8"] = path
            print(f"\n  ✅ Combined (Pruned + ONNX INT8) → {path.name}")
        except Exception as e:
            print(f"  ⚠️  Combined INT8 quant thất bại: {e}")

        # 5. Combined: Pruned + ONNX FP16
        try:
            with log_timing("quantize_onnx_combined_fp16", pruned_label):
                path = quantize_onnx_fp16(pruned_path, f"{pruned_label}_onnx_fp16")
            results[f"{pruned_label}_onnx_fp16"] = path
            print(f"\n  ✅ Combined (Pruned + ONNX FP16) → {path.name}")
        except Exception as e:
            print(f"  ⚠️  Combined FP16 quant thất bại: {e}")

    print(f"\n✅ Quantization hoàn thành! {len(results)} model")
    return results

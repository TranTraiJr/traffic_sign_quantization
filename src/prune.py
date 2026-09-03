"""
src/prune.py — Cắt Tỉa (Pruning) YOLOv8
=========================================

📚 LÝ THUYẾT PRUNING:
━━━━━━━━━━━━━━━━━━━━

Pruning (Cắt tỉa) loại bỏ các trọng số không quan trọng của mạng
để giảm kích thước và tăng tốc độ.

Phương pháp áp dụng cho YOLOv8:

1. L1 Unstructured Pruning (theo magnitude):
   - Đặt các weight có giá trị |w| nhỏ nhất = 0 (sparse weights)
   - Không thay đổi kiến trúc mạng → kích thước file không đổi
   - Nhưng các framework như ONNX Runtime có thể tận dụng sparsity
   - Ví dụ: prune 40% → 40% weight = 0

2. Tại sao cần Fine-tune sau Pruning?
   - Khi xóa đột ngột nhiều weight → accuracy giảm mạnh
   - Fine-tune ít epoch với lr nhỏ → mạng "học lại" với cấu trúc thưa mới
   - Kết quả: accuracy phục hồi gần baseline nhưng model thưa hơn

Mối quan hệ Pruning-Quantization:
   Pruning → Quantization → Combined (thường tốt nhất):
   Model thưa (nhiều số 0) + INT8 → Nhỏ nhất + Nhanh nhất
"""

import copy
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .config import CHECKPOINTS_DIR, prune_cfg


# =============================================================================
# 1. L1 UNSTRUCTURED PRUNING
# =============================================================================
def apply_l1_pruning(
    model_path: Path,
    amount: float,
    save_name: str,
    force: bool = False,
) -> Path:
    """
    Áp dụng L1 Unstructured Pruning lên tất cả lớp Conv2d của YOLOv8.

    Cơ chế:
    - Tính |w| (absolute value) của mỗi weight
    - Đặt `amount`% weight có |w| nhỏ nhất = 0
    - Sử dụng PyTorch torch.nn.utils.prune (chuẩn PyTorch)

    Args:
        model_path: Path đến .pt model gốc
        amount    : Tỷ lệ prune (0.4 = 40% weight nhỏ nhất bị zeroed)
        save_name : Tên để lưu model đã prune
        force     : Ép prune lại dù đã có checkpoint

    Returns:
        Path đến model đã prune (chưa fine-tune)
    """
    from ultralytics import YOLO

    out_path = CHECKPOINTS_DIR / f"{save_name}.pt"
    if out_path.exists() and not force:
        print(f"\n  ✅ Đã có checkpoint {out_path.name}. Bỏ qua bước prune lại.")
        return out_path

    print(f"\n  ✂️  L1 Pruning {int(amount*100)}%: {model_path.name}")

    # Load model và lấy nn.Module bên trong
    yolo = YOLO(str(model_path))
    pt_model = yolo.model

    # Đếm tổng Conv2d layers
    conv_layers = [(name, mod) for name, mod in pt_model.named_modules()
                   if isinstance(mod, nn.Conv2d)]

    print(f"    Tổng Conv2d layers : {len(conv_layers)}")

    # Áp dụng pruning
    for name, module in conv_layers:
        prune.l1_unstructured(module, name="weight", amount=amount)

    # Đếm số weight bị zeroed
    total_w = sum(p.numel() for p in pt_model.parameters())
    zero_w = sum((p == 0).sum().item() for p in pt_model.parameters())
    sparsity = zero_w / total_w * 100

    print(f"    Sparsity thực tế   : {sparsity:.1f}% weights = 0")

    # Make pruning permanent (xóa mask, giữ weight)
    for name, module in conv_layers:
        try:
            prune.remove(module, "weight")
        except Exception:
            pass

    # Lưu lại với ultralytics format
    yolo.model = pt_model
    yolo.save(str(out_path))

    size_mb = out_path.stat().st_size / 1024**2
    print(f"    Lưu → {out_path.name} ({size_mb:.1f} MB)")

    return out_path


# =============================================================================
# 2. CHẠY NHIỀU MỨC PRUNING
# =============================================================================
def run_pruning_experiments(
    baseline_path: Path,
    amounts: list[float] | None = None,
    do_finetune: bool = True,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Path]:
    """
    Chạy thực nghiệm pruning với nhiều mức amount.

    Workflow mỗi mức:
    1. Apply L1 pruning → pruned.pt
    2. (Optional) Fine-tune → pruned_finetuned.pt

    Args:
        baseline_path: Path đến baseline.pt đã train
        amounts      : List các mức prune (default từ prune_cfg)
        do_finetune  : Có fine-tune sau prune không (khuyến nghị True)
        device       : Thiết bị training
        force        : Ép chạy lại dù đã có checkpoint

    Returns:
        dict {save_name: model_path} cho mỗi mức prune
    """
    from .train import finetune

    if amounts is None:
        amounts = prune_cfg.amounts

    print("\n" + "=" * 65)
    print("  ✂️  PRUNING EXPERIMENTS")
    print("=" * 65)
    print(f"  Baseline : {baseline_path.name}")
    print(f"  Amounts  : {[f'{int(a*100)}%' for a in amounts]}")
    print(f"  Fine-tune: {'Có' if do_finetune else 'Không'}")

    results = {}

    for amount in amounts:
        pct = int(amount * 100)
        print(f"\n{'─'*55}")
        print(f"  [Pruning {pct}%]")

        # 1. Prune
        pruned_name = f"pruned_{pct}pct"
        pruned_path = apply_l1_pruning(baseline_path, amount, pruned_name, force=force)
        results[pruned_name] = pruned_path

        # 2. Fine-tune
        if do_finetune:
            ft_name = f"pruned_{pct}pct_finetuned"
            ft_path = CHECKPOINTS_DIR / f"{ft_name}.pt"
            if ft_path.exists() and not force:
                print(f"  ✅ Đã có checkpoint {ft_path.name}. Bỏ qua fine-tune.")
            else:
                ft_path = finetune(
                    model_path=pruned_path,
                    epochs=prune_cfg.finetune_epochs,
                    lr=prune_cfg.finetune_lr,
                    save_name=ft_name,
                    device=device,
                )
            results[ft_name] = ft_path

    print(f"\n✅ Hoàn thành {len(amounts)} mức pruning!")
    return results

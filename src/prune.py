"""
src/prune.py — Cắt Tỉa (Pruning) YOLOv8
=========================================

📚 LÝ THUYẾT PRUNING:
━━━━━━━━━━━━━━━━━━━━

Pruning (Cắt tỉa) loại bỏ các trọng số không quan trọng của mạng
để giảm kích thước và tăng tốc độ.

Phương pháp áp dụng cho YOLOv8:

1. L1 Global Unstructured Pruning (theo magnitude):
   - Xếp hạng |w| (giá trị tuyệt đối) của TẤT CẢ weight thuộc các lớp
     đủ điều kiện, rồi đặt `amount`% weight nhỏ nhất toàn cục = 0
   - "Global" (thay vì "local" — prune riêng từng layer amount% như nhau)
     giúp các lớp backbone dư thừa tự nhiên gánh phần pruning nhiều hơn,
     bảo vệ các lớp nhỏ/nhạy cảm hơn
   - Không thay đổi kiến trúc mạng → kích thước file không đổi
   - Ví dụ: prune 40% → 40% weight toàn cục = 0

2. ⚠️ VÌ SAO PHẢI LOẠI TRỪ DETECT HEAD KHỎI PRUNING?
   YOLOv8 Detect head (`model.model[-1]`, gồm nhánh `cv2` — box/DFL
   regression, `cv3` — class logits) là các lớp Conv2d:
     - Không có BatchNorm/activation phía sau để hấp thụ nhiễu do zero weight
     - Rất nhỏ (ít channel) → cùng % pruning gây tổn hại tỷ lệ lớn hơn
       nhiều so với backbone dư thừa
     - Quyết định trực tiếp phép giải mã Distribution Focal Loss (DFL)
       cho box và điểm tin cậy lớp
   Pruning các lớp này (dù chỉ 20%) phá vỡ hoàn toàn khả năng giải mã của
   model → mAP sập về gần 0%. Đây là lỗi kinh điển khi áp dụng pruning cho
   YOLO — script/tutorial pruning chính thức của Ultralytics luôn loại trừ
   Detect head. Vì vậy hàm dưới đây CHỈ prune Conv2d thuộc backbone + neck.

3. Tại sao cần Fine-tune sau Pruning?
   - Khi xóa đột ngột nhiều weight → accuracy giảm
   - Fine-tune ít epoch với lr nhỏ → mạng "học lại" với cấu trúc thưa mới
   - Để việc "học lại" không âm thầm hoàn tác pruning (optimizer cập nhật
     tự do các vị trí đã zero), ta GIỮ NGUYÊN mask trong suốt fine-tune
     (ép lại = 0 sau mỗi batch qua callback), chỉ khóa cứng kết quả cuối
     cùng sau khi fine-tune xong.

Mối quan hệ Pruning-Quantization:
   Pruning → Quantization → Combined (thường tốt nhất):
   Model thưa (nhiều số 0) + INT8 → Nhỏ nhất + Nhanh nhất
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .config import CHECKPOINTS_DIR, prune_cfg
from .timing import log_timing


# =============================================================================
# 0. XÁC ĐỊNH CÁC LỚP ĐỦ ĐIỀU KIỆN PRUNE (loại trừ Detect head)
# =============================================================================
def _get_prunable_conv_layers(pt_model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    """
    Trả về danh sách (tên, module) các lớp Conv2d thuộc backbone + neck.

    Loại trừ toàn bộ Detect head (`pt_model.model[-1]`, gồm cv2/cv3/dfl)
    — xem giải thích lý do ở docstring đầu file.
    """
    head = pt_model.model[-1]
    head_module_ids = {id(m) for m in head.modules()}

    return [
        (name, mod)
        for name, mod in pt_model.named_modules()
        if isinstance(mod, nn.Conv2d) and id(mod) not in head_module_ids
    ]


# =============================================================================
# 1. L1 GLOBAL UNSTRUCTURED PRUNING
# =============================================================================
def apply_l1_pruning(
    model_path: Path,
    amount: float,
    save_name: str,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Áp dụng L1 Global Unstructured Pruning lên Conv2d backbone+neck của YOLOv8
    (KHÔNG đụng tới Detect head).

    Args:
        model_path: Path đến .pt model gốc
        amount    : Tỷ lệ prune (0.4 = 40% weight nhỏ nhất toàn cục bị zeroed)
        save_name : Tên để lưu model đã prune
        force     : Ép prune lại dù đã có checkpoint

    Returns:
        (pruned_path, mask_path): Path model đã prune (chưa fine-tune) và
        Path file mask (dùng để giữ sparsity xuyên suốt fine-tune).
    """
    from ultralytics import YOLO

    out_path = CHECKPOINTS_DIR / f"{save_name}.pt"
    mask_path = CHECKPOINTS_DIR / f"{save_name}_masks.pt"
    if out_path.exists() and mask_path.exists() and not force:
        print(f"\n  ✅ Đã có checkpoint {out_path.name}. Bỏ qua bước prune lại.")
        return out_path, mask_path

    print(f"\n  ✂️  L1 Global Pruning {int(amount*100)}%: {model_path.name}")

    # Load model và lấy nn.Module bên trong
    yolo = YOLO(str(model_path))
    pt_model = yolo.model

    all_convs = [(n, m) for n, m in pt_model.named_modules() if isinstance(m, nn.Conv2d)]
    prunable = _get_prunable_conv_layers(pt_model)

    print(f"    Tổng Conv2d layers      : {len(all_convs)}")
    print(f"    Conv2d bị prune (loại Detect head): {len(prunable)}")

    # Global unstructured: xếp hạng |w| trên TOÀN BỘ layer đủ điều kiện
    prune.global_unstructured(
        [(mod, "weight") for _, mod in prunable],
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Đếm số weight bị zeroed (toàn model, bao gồm cả head — head phải = 0%)
    total_w = sum(p.numel() for p in pt_model.parameters())
    zero_w = sum((p == 0).sum().item() for p in pt_model.parameters())
    sparsity = zero_w / total_w * 100
    print(f"    Sparsity thực tế (toàn model): {sparsity:.1f}% weights = 0")

    # Lưu mask TRƯỚC khi bake permanent — dùng để giữ sparsity qua fine-tune
    masks = {name: mod.weight_mask.detach().clone() for name, mod in prunable}

    # Make pruning permanent (xóa mask/hook, giữ weight đã zero)
    for _, module in prunable:
        prune.remove(module, "weight")

    # Lưu lại với ultralytics format
    yolo.model = pt_model
    yolo.save(str(out_path))
    torch.save(masks, mask_path)

    size_mb = out_path.stat().st_size / 1024**2
    print(f"    Lưu → {out_path.name} ({size_mb:.1f} MB), mask → {mask_path.name}")

    return out_path, mask_path


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
    1. Apply L1 global pruning (loại trừ Detect head) → pruned.pt + mask
    2. (Optional) Fine-tune, GIỮ mask xuyên suốt → pruned_finetuned.pt

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
    print(f"  Fine-tune: {'Có (giữ mask xuyên suốt)' if do_finetune else 'Không'}")

    results = {}

    for amount in amounts:
        pct = int(amount * 100)
        print(f"\n{'─'*55}")
        print(f"  [Pruning {pct}%]")

        # 1. Prune
        pruned_name = f"pruned_{pct}pct"
        with log_timing("prune", f"{pct}%"):
            pruned_path, mask_path = apply_l1_pruning(baseline_path, amount, pruned_name, force=force)
        results[pruned_name] = pruned_path

        # 2. Fine-tune (giữ mask xuyên suốt để sparsity không bị hoàn tác)
        if do_finetune:
            ft_name = f"pruned_{pct}pct_finetuned"
            ft_path = CHECKPOINTS_DIR / f"{ft_name}.pt"
            if ft_path.exists() and not force:
                print(f"  ✅ Đã có checkpoint {ft_path.name}. Bỏ qua fine-tune.")
            else:
                with log_timing("finetune", f"{pct}%"):
                    ft_path = finetune(
                        model_path=pruned_path,
                        epochs=prune_cfg.finetune_epochs,
                        lr=prune_cfg.finetune_lr,
                        save_name=ft_name,
                        device=device,
                        mask_path=mask_path,
                    )
            results[ft_name] = ft_path

    print(f"\n✅ Hoàn thành {len(amounts)} mức pruning!")
    return results

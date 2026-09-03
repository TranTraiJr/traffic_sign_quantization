"""
src/train.py — Huấn Luyện YOLOv8 Trên Biển Báo Giao Thông VN
=============================================================

📚 LÝ THUYẾT TRANSFER LEARNING VỚI YOLO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLOv8 sử dụng Transfer Learning từ COCO dataset (80 lớp, 118k ảnh):
  - Backbone (CSPDarknet) đã học các đặc trưng hình ảnh phổ quát
  - Ta fine-tune trên 52 lớp biển báo VN → cần ít data hơn train từ đầu

Metric quan trọng:
  - mAP50: mean Average Precision @ IoU=0.5
    → IoU (Intersection over Union) đo độ chồng lấn bbox dự đoán vs thực
    → mAP50 > 0.7 là tốt với dataset 3k ảnh
  - Precision: Trong các bbox được detect, bao nhiêu % đúng
  - Recall: Trong các biển báo thực, bao nhiêu % được tìm thấy
"""

import shutil
from pathlib import Path

from .config import (
    CHECKPOINTS_DIR,
    RUNS_DIR,
    YOLOTrainConfig,
    get_yolo_device,
    train_cfg,
)
from .dataset import get_yaml_path, prepare_dataset

YOLO_RUNS_DIR = RUNS_DIR / "detect"


# =============================================================================
# 1. TRAIN YOLOv8
# =============================================================================
def train(
    cfg: YOLOTrainConfig = train_cfg,
    save_name: str = "baseline",
    device: str = "auto",
    resume: bool = False,
) -> object:
    """
    Train YOLOv8 trên dataset biển báo VN.

    Args:
        cfg      : Training config (epochs, batch, imgsz...)
        save_name: Tên checkpoint lưu trong checkpoints/
        device   : 'auto', 'cpu', 'mps', 'cuda'
        resume   : True để tiếp tục train từ checkpoint cũ

    Returns:
        Ultralytics YOLO model đã train
    """
    from ultralytics import YOLO

    # Xác định device
    actual_device = get_yolo_device() if device == "auto" else device

    print("\n" + "=" * 65)
    print(f"  🚀 TRAIN YOLOv8{cfg.model_size.upper()} — Vietnamese Traffic Signs")
    print("=" * 65)
    print(f"\n  Epochs    : {cfg.epochs} (early stop {cfg.patience}ep)")
    print(f"  Img size  : {cfg.imgsz}×{cfg.imgsz} px")
    print(f"  Batch     : {cfg.batch}")
    print(f"  Device    : {actual_device}")
    print(f"  Save as   : checkpoints/{save_name}.pt\n")

    # Chuẩn bị dataset paths
    prepare_dataset(verbose=True)

    # Load pretrained model
    ckpt_path = CHECKPOINTS_DIR / f"{save_name}.pt"
    if resume and ckpt_path.exists():
        print(f"  📂 Resume từ: {ckpt_path}")
        model = YOLO(str(ckpt_path))
    else:
        model = YOLO(f"yolov8{cfg.model_size}.pt")

    # Train
    model.train(
        data=str(get_yaml_path()),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        patience=cfg.patience,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        # Augmentation
        mosaic=cfg.mosaic,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        # Output
        device=actual_device,
        project=str(YOLO_RUNS_DIR),
        name=save_name,
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
    )

    # Copy best.pt → checkpoints/
    best_pt = YOLO_RUNS_DIR / save_name / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, ckpt_path)
        size_mb = ckpt_path.stat().st_size / 1024**2
        print(f"\n  ✅ Best model ({size_mb:.1f} MB) → {ckpt_path}")
    else:
        print(f"\n  ⚠️  Không tìm thấy best.pt tại {best_pt}")

    return ckpt_path


# =============================================================================
# 2. LOAD MODEL TỪ CHECKPOINT
# =============================================================================
def load_model(save_name: str = "baseline") -> object:
    """
    Load YOLOv8 model từ checkpoint đã train.

    Args:
        save_name: Tên file trong checkpoints/ (không cần .pt)

    Returns:
        YOLO model
    """
    from ultralytics import YOLO

    ckpt_path = CHECKPOINTS_DIR / f"{save_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {ckpt_path}\n"
            f"Chạy train trước: uv run python main.py --phase train"
        )

    print(f"  📦 Nạp model: {save_name}.pt ({ckpt_path.stat().st_size/1024**2:.1f} MB)")
    return YOLO(str(ckpt_path))


# =============================================================================
# 3. FINE-TUNE SAU PRUNING
# =============================================================================
def finetune(
    model_path: Path,
    epochs: int = 10,
    lr: float = 1e-4,
    save_name: str = "pruned_finetuned",
    device: str = "auto",
) -> Path:
    """
    Fine-tune model sau khi pruning để phục hồi accuracy.

    Dùng learning rate nhỏ hơn nhiều so với training gốc
    để tránh "quên" các đặc trưng đã học (catastrophic forgetting).

    Args:
        model_path: Path đến .pt model đã prune
        epochs    : Số epoch fine-tune (thường 5-15, ít hơn train gốc nhiều)
        lr        : Learning rate nhỏ (default 1e-4 << 1e-2 training gốc)
        save_name : Tên checkpoint kết quả

    Returns:
        Path đến checkpoint sau fine-tune
    """
    from ultralytics import YOLO

    actual_device = get_yolo_device() if device == "auto" else device

    print(f"\n  🔧 Fine-tune sau pruning: {epochs} epochs, lr={lr}")
    model = YOLO(str(model_path))

    model.train(
        data=str(get_yaml_path()),
        epochs=epochs,
        imgsz=train_cfg.imgsz,
        batch=train_cfg.batch,
        lr0=lr,
        lrf=lr * 0.1,
        device=actual_device,
        project=str(YOLO_RUNS_DIR),
        name=save_name,
        exist_ok=True,
        verbose=False,  # Ít verbose hơn khi fine-tune
        save=True,
        plots=False,
    )

    best_pt = YOLO_RUNS_DIR / save_name / "weights" / "best.pt"
    out_path = CHECKPOINTS_DIR / f"{save_name}.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, out_path)
        print(f"  ✅ Fine-tuned model → {out_path}")

    return out_path

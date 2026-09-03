"""
src/config.py — Cấu Hình Tập Trung Cho Dự Án YOLOv8
=====================================================

Toàn bộ hyperparameters, đường dẫn, và cấu hình thiết bị
được quản lý tại đây để tránh "magic numbers" rải rác trong code.
"""

import platform
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# 1. ĐƯỜNG DẪN DỰ ÁN
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset gốc (YOLO format sẵn)
ARCHIVE_DIR = PROJECT_ROOT / "archive"
IMAGES_DIR = ARCHIVE_DIR / "images"
LABELS_DIR = ARCHIVE_DIR / "labels"
CLASSES_FILE = ARCHIVE_DIR / "classes.txt"
TRAIN_LIST = ARCHIVE_DIR / "split_dataset" / "train_files.txt"
TEST_LIST = ARCHIVE_DIR / "split_dataset" / "test_files.txt"

# Thư mục output
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

# File paths YOLO
YOLO_TRAIN_TXT = DATA_DIR / "yolo_train_paths.txt"
YOLO_VAL_TXT = DATA_DIR / "yolo_val_paths.txt"
YOLO_YAML = DATA_DIR / "traffic_signs.yaml"

# Tự động tạo thư mục cần thiết
for _dir in [CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. CLASS NAMES (52 biển báo giao thông Việt Nam)
# =============================================================================
def load_class_names() -> list[str]:
    """Đọc danh sách tên mã biển báo từ file classes.txt."""
    if not CLASSES_FILE.exists():
        return [f"class_{i}" for i in range(52)]
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


CLASSES_VIE_FILE = ARCHIVE_DIR / "classes_vie.txt"


def load_class_names_vie() -> list[str]:
    """Đọc danh sách tên tiếng Việt giải nghĩa biển báo từ classes_vie.txt."""
    if not CLASSES_VIE_FILE.exists():
        return load_class_names()
    with open(CLASSES_VIE_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


CLASS_NAMES: list[str] = load_class_names()
CLASS_NAMES_VIE: list[str] = load_class_names_vie()
NUM_CLASSES: int = len(CLASS_NAMES)


# =============================================================================
# 3. CẤU HÌNH YOLO TRAINING
# =============================================================================
@dataclass
class YOLOTrainConfig:
    """Hyperparameters cho việc train YOLOv8."""

    model_size: str = "n"       # n=nano(~6MB), s=small(~22MB), m=medium(~52MB)
    epochs: int = 50            # Số epoch tối đa
    imgsz: int = 640            # Kích thước ảnh input YOLO (640×640)
    batch: int = 16             # Batch size (giảm xuống 8 nếu OOM)
    patience: int = 15          # Early stopping: dừng nếu không cải thiện
    lr0: float = 0.01           # Learning rate ban đầu
    lrf: float = 0.01           # Learning rate cuối (lr0 * lrf)
    weight_decay: float = 0.0005
    warmup_epochs: int = 3      # Epoch warmup scheduler
    conf_threshold: float = 0.35  # Ngưỡng confidence khi inference
    iou_threshold: float = 0.45   # Ngưỡng IoU cho NMS

    # Data augmentation
    mosaic: float = 0.8         # Xác suất áp dụng mosaic augmentation
    flipud: float = 0.0         # Lật ảnh dọc (không áp dụng với biển báo)
    fliplr: float = 0.5         # Lật ảnh ngang
    degrees: float = 5.0        # Xoay ảnh tối đa ±5 độ
    translate: float = 0.1      # Dịch ảnh tối đa 10%
    scale: float = 0.3          # Scale ảnh ±30%


# =============================================================================
# 4. CẤU HÌNH PRUNING
# =============================================================================
@dataclass
class PruneConfig:
    """Cấu hình thực nghiệm Cắt tỉa (Pruning) cho YOLOv8."""

    # Các mức pruning cần thử nghiệm
    amounts: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5])

    # Số epoch fine-tune sau khi prune để phục hồi accuracy
    finetune_epochs: int = 10
    finetune_lr: float = 1e-4   # Learning rate nhỏ hơn để fine-tune


# =============================================================================
# 5. CẤU HÌNH QUANTIZATION
# =============================================================================
@dataclass
class QuantConfig:
    """Cấu hình Lượng tử hóa (Quantization) cho YOLOv8."""

    # Kích thước ảnh khi export (phải khớp với imgsz lúc train)
    imgsz: int = 640

    # Số warmup runs khi benchmark
    n_warmup: int = 5

    # Số runs để lấy trung bình latency
    n_runs: int = 30


# =============================================================================
# 6. KHỞI TẠO CONFIG INSTANCES
# =============================================================================
train_cfg = YOLOTrainConfig()
prune_cfg = PruneConfig()
quant_cfg = QuantConfig()


# =============================================================================
# 7. DEVICE DETECTION
# =============================================================================
def get_device() -> str:
    """
    Tự động chọn thiết bị tính toán mạnh nhất:
    CUDA (Nvidia) > MPS (Apple Silicon) > CPU

    Note: Benchmarking latency LUÔN dùng CPU để mô phỏng edge device.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_yolo_device() -> str:
    """
    Device string theo định dạng ultralytics:
    - 'cpu', 'mps', '0' (GPU index)
    """
    device = get_device()
    if device == "cuda":
        return "0"
    return device  # 'mps' hoặc 'cpu'

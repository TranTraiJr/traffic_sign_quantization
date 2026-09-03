"""
src/dataset.py — Chuẩn Bị Dataset YOLOv8 Cho Biển Báo Giao Thông VN
=====================================================================

📚 FORMAT DỮ LIỆU:
━━━━━━━━━━━━━━━━━
Dataset gốc đã ở định dạng YOLO chuẩn:
  archive/images/0001.jpg  ← ảnh gốc (chụp từ đường phố VN)
  archive/labels/0001.txt  ← annotation tương ứng

Mỗi dòng trong file .txt là 1 biển báo:
  <class_id> <x_center> <y_center> <width> <height>
  Tọa độ được normalize về [0, 1] theo chiều ảnh.

Ví dụ: "10 0.940625 0.451852 0.055208 0.096296"
  → Class 10 (P.130), bbox tâm tại (94%, 45%), kích thước 5.5% × 9.6% ảnh

Ultralytics YOLOv8 cần:
  1. File .yaml mô tả dataset (đã có: data/traffic_signs.yaml)
  2. File .txt chứa đường dẫn TUYỆT ĐỐI đến từng ảnh
"""

from pathlib import Path

from .config import (
    IMAGES_DIR,
    LABELS_DIR,
    NUM_CLASSES,
    TEST_LIST,
    TRAIN_LIST,
    YOLO_TRAIN_TXT,
    YOLO_VAL_TXT,
    YOLO_YAML,
    CLASS_NAMES,
)


# =============================================================================
# 1. TẠO FILE PATHS CHO ULTRALYTICS
# =============================================================================
def prepare_dataset(verbose: bool = True) -> tuple[Path, Path]:
    """
    Tạo 2 file .txt chứa đường dẫn tuyệt đối đến ảnh train/val.

    Ultralytics cần đường dẫn tuyệt đối khi dùng file list thay vì thư mục.
    Đồng thời lọc bỏ các cặp ảnh/label bị thiếu.

    Returns:
        (train_txt_path, val_txt_path)
    """
    YOLO_TRAIN_TXT.parent.mkdir(parents=True, exist_ok=True)

    for split_name, list_file, out_file in [
        ("train", TRAIN_LIST, YOLO_TRAIN_TXT),
        ("val", TEST_LIST, YOLO_VAL_TXT),
    ]:
        with open(list_file, "r") as f:
            filenames = [l.strip() for l in f if l.strip()]

        valid_paths = []
        skipped = 0

        for fname in filenames:
            img_path = IMAGES_DIR / fname
            lbl_path = LABELS_DIR / fname.replace(".jpg", ".txt")

            if img_path.exists() and lbl_path.exists():
                valid_paths.append(str(img_path.resolve()))
            else:
                skipped += 1

        with open(out_file, "w") as f:
            f.write("\n".join(valid_paths))

        if verbose:
            msg = f"   [{split_name:5s}] {len(valid_paths):4d} ảnh hợp lệ"
            if skipped:
                msg += f" (bỏ qua {skipped} thiếu file)"
            print(msg)

    return YOLO_TRAIN_TXT, YOLO_VAL_TXT


# =============================================================================
# 2. KIỂM TRA TÍNH TOÀN VẸN DATASET
# =============================================================================
def verify_dataset(n_check: int = 300) -> bool:
    """
    Kiểm tra nhanh:
    - Ảnh và label tồn tại theo cặp
    - Label có đúng format YOLO (5 cột)
    - Class ID trong [0, NUM_CLASSES-1]

    Args:
        n_check: Số ảnh kiểm tra mỗi split (để nhanh)

    Returns:
        True nếu dataset hợp lệ
    """
    print("\n🔍 Kiểm tra dataset...")
    errors = []
    total_ann = 0

    for list_file in [TRAIN_LIST, TEST_LIST]:
        split = "train" if list_file == TRAIN_LIST else "val"
        with open(list_file, "r") as f:
            fnames = [l.strip() for l in f if l.strip()][:n_check]

        for fname in fnames:
            img = IMAGES_DIR / fname
            lbl = LABELS_DIR / fname.replace(".jpg", ".txt")

            if not img.exists():
                errors.append(f"Thiếu ảnh [{split}]: {fname}")
                continue
            if not lbl.exists():
                errors.append(f"Thiếu label [{split}]: {fname}")
                continue

            with open(lbl) as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        errors.append(f"Format sai dòng {i} [{split}]: {fname}")
                        continue
                    cid = int(parts[0])
                    if not (0 <= cid < NUM_CLASSES):
                        errors.append(f"Class ID {cid} không hợp lệ [{split}]: {fname}")
                    total_ann += 1

    if errors:
        print(f"   ❌ {len(errors)} lỗi:")
        for e in errors[:5]:
            print(f"      {e}")
        return False

    print(f"   ✅ {n_check*2} ảnh kiểm tra, {total_ann} annotations — OK!")
    return True


# =============================================================================
# 3. THỐNG KÊ DATASET
# =============================================================================
def print_stats():
    """In thống kê đầy đủ: số ảnh, annotations, phân phối class."""
    from collections import Counter

    print("\n📊 THỐNG KÊ DATASET")
    print("=" * 55)

    for list_file, split in [(TRAIN_LIST, "TRAIN"), (TEST_LIST, "VAL")]:
        with open(list_file) as f:
            fnames = [l.strip() for l in f if l.strip()]

        counter = Counter()
        multi = 0
        total_ann = 0

        for fname in fnames:
            lbl = LABELS_DIR / fname.replace(".jpg", ".txt")
            if not lbl.exists():
                continue
            with open(lbl) as f:
                lines = [l for l in f if l.strip()]
            if len(lines) > 1:
                multi += 1
            for line in lines:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1
                    total_ann += 1

        print(f"\n  [{split}]")
        print(f"  • Ảnh          : {len(fnames)}")
        print(f"  • Annotations  : {total_ann}")
        print(f"  • Avg biển/ảnh : {total_ann/max(len(fnames),1):.2f}")
        print(f"  • Nhiều biển   : {multi} ảnh ({multi/max(len(fnames),1)*100:.1f}%)")
        print(f"  • Classes có mặt: {len(counter)}/{NUM_CLASSES}")

    print("=" * 55)


# =============================================================================
# 4. LẤY ĐƯỜNG DẪN YAML
# =============================================================================
def get_yaml_path() -> Path:
    """Trả về đường dẫn tuyệt đối đến traffic_signs.yaml."""
    return YOLO_YAML.resolve()


if __name__ == "__main__":
    print_stats()
    verify_dataset()
    train_txt, val_txt = prepare_dataset()
    print(f"\n✅ Dataset sẵn sàng:")
    print(f"   YAML : {get_yaml_path()}")
    print(f"   Train: {train_txt}")
    print(f"   Val  : {val_txt}")

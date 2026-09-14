"""
demo_video.py — Real-time Video Demo với FPS Counter
====================================================

Demo thiết bị biên (edge device) thực sự:
  - Nhận video file hoặc webcam
  - YOLOv8 detect và nhận dạng biển báo giao thông VN
  - Hiển thị bounding box + tên biển + confidence + FPS real-time
  - Có thể chọn giữa model baseline hoặc đã quantize để so sánh FPS

CÁCH CHẠY:
    # Dùng video file
    uv run python demo_video.py --source video.mp4

    # Dùng webcam (index 0)
    uv run python demo_video.py --source 0

    # So sánh baseline vs ONNX INT8 side-by-side
    uv run python demo_video.py --source video.mp4 --compare

    # Lưu output video
    uv run python demo_video.py --source video.mp4 --save-video

YÊU CẦU:
    - Đã train YOLOv8: uv run python main.py --task yolo --phase train
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    # Console/redirect trên Windows mặc định dùng codepage ANSI (vd cp1252),
    # không encode được emoji/tiếng Việt -> ép UTF-8 để tránh UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CHECKPOINTS_DIR, CLASS_NAMES, CLASS_NAMES_VIE, PROJECT_ROOT


# =============================================================================
# 1. PALETTE MÀU SẮC CHO 52 CLASS & FONT TIẾNG VIỆT
# =============================================================================
def get_color_palette(n: int = 52) -> list:
    """Tạo bảng màu đẹp cho N class."""
    np.random.seed(42)
    palette = []
    for i in range(n):
        # HSV → BGR (màu sặc sỡ, dễ phân biệt)
        hue = int(i * 180 / n)
        color_hsv = np.array([[[hue, 220, 240]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append(tuple(int(c) for c in color_bgr))
    return palette


COLORS = get_color_palette(52)


def get_system_font(size: int = 15):
    """Tìm font hệ thống có hỗ trợ tiếng Việt Unicode."""
    font_candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in font_candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


LABEL_FONT = get_system_font(15)


# =============================================================================
# 2. VẼ KẾT QUẢ DETECT LÊN FRAME (KÈM GIẢI NGHĨA TIẾNG VIỆT)
# =============================================================================
def draw_detections(
    frame,
    results,
    class_names: list,
    class_names_vie: list = None,
    conf_threshold: float = 0.35,
) -> tuple:
    """
    Vẽ bounding box, mã biển báo, tên tiếng Việt và confidence lên frame.
    Returns: (annotated_frame, n_detections)
    """
    n_det = 0

    if not (results and results[0].boxes is not None and len(results[0].boxes) > 0):
        return frame, 0

    boxes = results[0].boxes
    valid_boxes = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            valid_boxes.append(box)

    if not valid_boxes:
        return frame, 0

    labels_to_draw = []
    for box in valid_boxes:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        code = class_names[class_id] if class_id < len(class_names) else f"cls{class_id}"
        vie = class_names_vie[class_id] if class_names_vie and class_id < len(class_names_vie) else ""
        color = COLORS[class_id % len(COLORS)]

        # Tọa độ bbox
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: Mã hiệu + Tên tiếng Việt + % confidence
        if vie and vie != code:
            label = f"{code}: {vie} ({conf:.0%})"
        else:
            label = f"{code} ({conf:.0%})"

        labels_to_draw.append((label, x1, y1, color))
        n_det += 1

    # Dùng PIL vẽ chữ tiếng Việt có dấu đẹp, sắc nét
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for label, x1, y1, color in labels_to_draw:
            bbox = draw.textbbox((x1, y1), label, font=LABEL_FONT)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            label_y = max(y1 - th - 8, 4)

            # Nền nhãn
            draw.rectangle(
                [(x1, label_y), (x1 + tw + 8, label_y + th + 6)],
                fill=(color[2], color[1], color[0]),  # BGR -> RGB
            )
            # Chữ nhãn
            text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
            draw.text((x1 + 4, label_y + 2), label, font=LABEL_FONT, fill=text_color)

        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback vẽ bằng OpenCV nếu PIL gặp lỗi
        for label, x1, y1, color in labels_to_draw:
            cv2.putText(frame, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, n_det


# =============================================================================
# 3. VẼ HUD (FPS + THÔNG TIN MODEL)
# =============================================================================
def draw_hud(frame, fps: float, model_name: str, n_det: int, frame_count: int):
    """Vẽ Heads-Up Display góc trên trái."""
    h, w = frame.shape[:2]

    # Nền HUD semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS — màu xanh nếu >= 15, vàng nếu 8-15, đỏ nếu < 8
    fps_color = (0, 255, 0) if fps >= 15 else (0, 200, 255) if fps >= 8 else (0, 0, 255)
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fps_color, 2, cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Model: {model_name}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Detected: {n_det} sign(s)",
        (10, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Frame: #{frame_count}",
        (10, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    return frame


# =============================================================================
# 4. LOAD MODEL
# =============================================================================
def load_yolo_model(model_path: Path):
    """Load YOLOv8 model từ .pt file."""
    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path}\n"
            f"Hãy train trước: uv run python main.py --task yolo --phase train"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = YOLO(str(model_path), task="detect")

    return model


def load_onnx_model(onnx_path: Path):
    """Load ONNX INT8 model."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess


# =============================================================================
# 5. INFERENCE VỚI YOLO .pt
# =============================================================================
def predict_yolo(model, frame):
    """Chạy inference với YOLO .pt model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.predict(frame, device="cpu", verbose=False, conf=0.35, iou=0.45)
    return results


# =============================================================================
# 6. MAIN LOOP
# =============================================================================
def run_video_demo(
    source,
    model_path: Path,
    conf_threshold: float = 0.35,
    save_video: bool = False,
    max_frames: int = 0,
):
    """
    Main video inference loop.

    Args:
        source    : Path video hoặc int (webcam index)
        model_path: Path đến .pt model
        conf_threshold: Ngưỡng confidence tối thiểu
        save_video: True để lưu output
        max_frames: Giới hạn số frame (0 = không giới hạn)
    """
    # Load model
    print(f"\n📦 Nạp model: {model_path.name}")
    model = load_yolo_model(model_path)
    model_name = model_path.stem

    # Nạp class names (đã import từ src.config ở đầu file)
    class_names = CLASS_NAMES

    # Mở video source
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video source: {source}")

    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Video: {w}×{h} @ {fps_cap:.0f} FPS")

    # VideoWriter nếu cần lưu
    writer = None
    if save_video:
        out_path = PROJECT_ROOT / "results" / f"demo_{model_name}.mp4"
        out_path.parent.mkdir(exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_cap, (w, h))
        print(f"💾 Lưu video tại: {out_path}")

    print("\n🎬 Đang chạy... Nhấn Q để thoát\n")

    frame_count = 0
    fps_history = []
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames > 0 and frame_count > max_frames:
            break

        # Inference
        t0 = time.perf_counter()
        results = predict_yolo(model, frame)
        t_infer = (time.perf_counter() - t0) * 1000

        # FPS tính theo thời gian thực giữa các frame
        t_now = time.perf_counter()
        frame_fps = 1.0 / (t_now - t_prev)
        t_prev = t_now
        fps_history.append(frame_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        smooth_fps = float(np.mean(fps_history))

        # Vẽ kết quả (kèm tên tiếng Việt)
        frame, n_det = draw_detections(
            frame, results, class_names, CLASS_NAMES_VIE, conf_threshold
        )
        frame = draw_hud(frame, smooth_fps, model_name, n_det, frame_count)

        # Hiển thị
        cv2.imshow(f"Traffic Sign Detection — {model_name}", frame)

        if writer:
            writer.write(frame)

        # In log mỗi 30 frame
        if frame_count % 30 == 0:
            print(
                f"   Frame {frame_count:4d} | FPS: {smooth_fps:5.1f} | Infer: {t_infer:5.1f}ms | Det: {n_det}"
            )

        # Thoát khi nhấn Q hoặc ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    avg_fps = float(np.mean(fps_history)) if fps_history else 0
    print(f"\n✅ Hoàn thành! {frame_count} frames, FPS trung bình: {avg_fps:.1f}")
    return avg_fps


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Demo real-time nhận dạng biển báo giao thông với YOLOv8"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Nguồn video: đường dẫn file .mp4 hoặc số webcam (mặc định: 0)",
    )
    parser.add_argument(
        "--model",
        default=str(CHECKPOINTS_DIR / "baseline.pt"),
        help="Đường dẫn .pt hoặc .onnx model (mặc định: checkpoints/baseline.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Ngưỡng confidence (0.0-1.0, mặc định: 0.35)",
    )
    parser.add_argument("--save-video", action="store_true", help="Lưu output video ra file")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Giới hạn số frame (0 = không giới hạn)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  🚦 DEMO REAL-TIME: Nhận Dạng Biển Báo Giao Thông VN")
    print("  📌 Môn: Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng")
    print("=" * 65)

    # Parse source
    try:
        source = int(args.source)  # Webcam index
    except ValueError:
        source = Path(args.source)  # Video file
        if not source.exists():
            print(f"❌ Không tìm thấy video: {source}")
            sys.exit(1)

    model_path = Path(args.model)

    run_video_demo(
        source=source,
        model_path=model_path,
        conf_threshold=args.conf,
        save_video=args.save_video,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

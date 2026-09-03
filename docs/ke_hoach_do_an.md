# Kế Hoạch Thực Hiện Đồ Án — Nhóm [số nhóm]

**Đề tài:** Nghiên cứu, ứng dụng kỹ thuật Lượng tử hóa (Quantization) và Cắt tỉa (Pruning) để tối ưu hóa mô hình nhận dạng biển báo giao thông Việt Nam trên thiết bị biên  
**Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng  

---

| Tuần | Nội dung | Ngày báo cáo |
|---|---|---|
| **1** (29/7 – 8/8) | **Khởi động & Nghiên cứu:** Chốt đề tài nhận dạng biển báo giao thông Việt Nam; khảo sát tài liệu về Object Detection (YOLOv8), Quantization (Dynamic INT8, ONNX INT8), Pruning (L1 Unstructured); lựa chọn dataset biển báo VN thực tế (3,191 ảnh, 52 lớp, định dạng YOLO sẵn); phân công vai trò; cài đặt môi trường phát triển (Python 3.12, PyTorch, Ultralytics, ONNX Runtime). | **08/8/2026** |
| **2** (9/8 – 15/8) | **Dữ liệu & Xây dựng pipeline:** Phân tích và kiểm tra tính toàn vẹn dataset (8,334 annotations, 2.59 biển/ảnh trung bình); xây dựng toàn bộ pipeline code (config, dataset, train, evaluate, prune, quantize, pipeline, visualize); cấu hình YAML dataset cho Ultralytics; chạy thử `prepare` và kiểm tra luồng dữ liệu end-to-end. **Huấn luyện mô hình YOLOv8n baseline** (50 epochs, imgsz=640, Transfer Learning từ COCO). | **15/8/2026** |
| **3** (16/8 – 22/8) | **Cắt tỉa mô hình (Pruning):** Áp dụng L1 Unstructured Pruning (`torch.nn.utils.prune`) lên các lớp Conv2d của YOLOv8n tại 4 mức: 20%, 30%, 40%, 50%; fine-tune 10 epochs sau mỗi mức để phục hồi accuracy; đánh giá mAP50 và FPS CPU sau từng mức pruning; xác định mức tối ưu (dự kiến 40% — cân bằng tốt nhất accuracy/tốc độ). | **22/8/2026** |
| **4** (23/8 – 29/8) | **Lượng tử hóa (Quantization) & Benchmark:** Áp dụng Dynamic INT8 Quantization (`torch.ao.quantization`) và ONNX INT8 Quantization (`onnxruntime`); chạy Combined (Pruned 40% + ONNX INT8); benchmark đầy đủ 3 chiều: mAP50, FPS trên CPU (mô phỏng edge device), kích thước model; sinh file CSV kết quả và 4 biểu đồ so sánh; hoàn thiện `demo_video.py` — demo nhận dạng real-time trên video có bounding box và FPS counter. | **29/8/2026** |
| **5** (30/8 – 05/9) | **Kiểm thử & Báo cáo:** Thiết kế kịch bản kiểm thử (video đường phố Việt Nam); đo hiệu năng cuối cùng (độ trễ ms/frame, FPS, kích thước MB, mAP50%); tổng hợp số liệu vào bảng so sánh; viết báo cáo đồ án đầy đủ; thiết kế slide thuyết trình; chuẩn bị kịch bản demo video thực tế trước giáo viên. | **5/9/2026** |
| **6** (5/9 – 16/9) | **Bảo vệ, hoàn thiện và nộp báo cáo.** | **15/9/2026** |

---

## Phân Công Vai Trò

| Thành viên | Phụ trách |
|---|---|
| [Thành viên 1] | Dataset, config, pipeline tổng thể |
| [Thành viên 2] | Pruning, fine-tune, evaluate |
| [Thành viên 3] | Quantization (PyTorch + ONNX), benchmark |
| [Thành viên 4] | Demo video, visualize, báo cáo |

---

## Công Nghệ Sử Dụng

| Hạng mục | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.12 |
| Deep Learning | PyTorch 2.x |
| Detection Framework | Ultralytics YOLOv8 |
| Quantization | `torch.ao.quantization`, ONNX Runtime |
| Pruning | `torch.nn.utils.prune` |
| Video/Camera | OpenCV |
| Visualization | Matplotlib, Pandas |
| Package Manager | uv |
| Môi trường | macOS (phát triển), CPU benchmark (mô phỏng edge) |

---

## Kết Quả Kỳ Vọng

| Model | mAP50 | FPS CPU | Size |
|---|---|---|---|
| YOLOv8n Baseline (float32) | ~75–85% | ~8–12 FPS | ~6 MB |
| Pruned 40% + Fine-tune | ~72–82% | ~10–14 FPS | ~6 MB |
| ONNX INT8 (baseline) | ~73–83% | ~20–30 FPS | ~2–3 MB |
| **Pruned 40% + ONNX INT8** | ~70–80% | **~25–35 FPS** | **~1.5–2 MB** |

> **Ngưỡng real-time:** ≥ 15 FPS → đáp ứng yêu cầu thiết bị biên không GPU

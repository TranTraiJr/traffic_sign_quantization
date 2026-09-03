# BÁO CÁO TIẾN ĐỘ TUẦN 1
**Thời gian:** 29/07/2026 – 08/08/2026  
**Giai đoạn:** Khởi động & Nghiên cứu  
**Nhóm thực hiện:** [Tên nhóm]  
**Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng  

---

## 1. Tổng quan công việc tuần 1

Tuần đầu tiên tập trung vào ba mục tiêu lớn: **(1)** xác định và chốt đề tài, **(2)** nghiên cứu tài liệu tổng quan về các kỹ thuật liên quan, và **(3)** chuẩn bị hạ tầng (dataset, môi trường phát triển, phân công công việc). Đây là tuần nền móng, quyết định hướng đi kỹ thuật cho toàn bộ đồ án.

---

## 2. Công việc đã thực hiện

### 2.1. Xác định và chốt đề tài

Nhóm đã đề xuất và thống nhất đề tài:

> **"Nghiên cứu và xây dựng hệ thống nhận diện biển báo giao thông Việt Nam, ứng dụng kỹ thuật Lượng tử hóa (Quantization) và Cắt tỉa (Pruning) để tối ưu hóa mô hình trên thiết bị biên."**

**Lý do lựa chọn đề tài:**

- Biển báo giao thông là thành phần thiết yếu trong hệ thống ADAS (Advanced Driver Assistance Systems) và xe tự lái — bài toán có tính ứng dụng thực tế cao.
- Thách thức cốt lõi: các mô hình deep learning chính xác cao thường có kích thước lớn (hàng trăm MB), không thể triển khai trực tiếp trên thiết bị biên (Raspberry Pi, camera thông minh…) có tài nguyên hạn chế.
- Quantization và Pruning là hai kỹ thuật tối ưu mô hình phổ biến nhất trong nghiên cứu AI biên — có cơ sở lý thuyết vững chắc và kết quả đã được kiểm chứng trong nhiều công trình quốc tế.
- Sử dụng dữ liệu biển báo **Việt Nam** (thay vì GTSRB — Đức) mang lại tính ứng dụng thực tiễn và giá trị nghiên cứu bản địa cao hơn.

---

### 2.2. Khảo sát tài liệu và bài báo liên quan

Nhóm đã đọc và phân tích 4 công trình liên quan:

#### Bài báo nước ngoài 1
**Hasan et al. (2021) — "Real-Time Traffic Sign Recognition based AI Edge Computing"**

- **Phương pháp:** Triển khai Tiny-YOLOv3 trên chip Edge AI K210-KPU (RISC-V 64-bit, module Sipeed MAIX). Dùng K-Means để tối ưu anchor boxes cho dataset biển báo.
- **Kết quả:** 9 FPS (112ms/ảnh) trên luồng video thực tế.
- **Ý nghĩa:** Xác nhận tính khả thi của họ YOLO trên edge hardware. Tuy nhiên 9 FPS còn thấp, cho thấy cần thêm Pruning + Quantization để đạt ngưỡng real-time ≥15-20 FPS — đúng hướng đề tài.

#### Bài báo nước ngoài 2
**IEEE (2025) — "Dynamic Quantization and Pruning for Efficient CNN-Based Road Sign Recognition on FPGA"**

- **Phương pháp:** Kết hợp Dynamic Quantization (điều chỉnh bit-width riêng từng lớp) + Pruning (loại bỏ kết nối ít quan trọng) trên FPGA Tang Primer 25K.
- **Kết quả:** Giảm mức tiêu thụ điện năng và tài nguyên phần cứng đáng kể, độ chính xác giảm không đáng kể.
- **Ý nghĩa:** Củng cố luận điểm cốt lõi — kết hợp Pruning + Quantization đồng thời cho kết quả tốt hơn từng kỹ thuật riêng lẻ. Gợi ý hướng phát triển Mixed-Precision Quantization.

#### Nghiên cứu trong nước 1
**ResearchGate (2025) — "Phát hiện biển báo giao thông Việt Nam: So sánh YOLOv8 và Faster R-CNN"**

- **Phương pháp:** So sánh YOLOv8 và Faster R-CNN trên dataset biển báo VN (1,170 ảnh gốc, 29 lớp, tăng cường lên 10,170 ảnh).
- **Kết quả:** YOLOv8 đạt **92.68% mAP, 95.83% Precision, 45 FPS** — vượt trội Faster R-CNN.
- **Ý nghĩa:** Căn cứ tham chiếu quan trọng — YOLOv8 phù hợp nhất cho biển báo VN. Con số 45 FPS (GPU) làm cơ sở để đặt mục tiêu ≥20 FPS cho YOLOv8-nano đã tối ưu chạy trên CPU.

#### Nghiên cứu trong nước 2
**Khóa luận tốt nghiệp — "Xây dựng hệ thống phát hiện sớm tín hiệu biển báo giao thông cho lái xe dựa trên kỹ thuật học sâu"**

- **Phương pháp:** Xây dựng dataset biển báo VN chuẩn hóa; áp dụng CNN kết hợp ResNet cho điều kiện khó (ánh sáng yếu, che khuất).
- **Ý nghĩa:** Nhấn mạnh thách thức đặc thù Việt Nam mà GTSRB không phản ánh đủ — biển báo VN đa dạng về hình dạng, màu sắc, và điều kiện chụp.

**Tổng hợp khoảng trống nghiên cứu được xác định:**
> Chưa có công trình nào kết hợp đồng thời **YOLOv8-nano + L1 Pruning + Quantization (Dynamic INT8 + ONNX INT8) + dataset biển báo Việt Nam thực tế**. Đây chính là đóng góp của đề tài.

---

### 2.3. Lựa chọn kiến trúc mô hình

Sau khi phân tích tài liệu, nhóm chốt kiến trúc:

**Mô hình: YOLOv8-nano (YOLOv8n)**

| Tiêu chí | Lý do chọn |
|---|---|
| Single-stage detector | Phát hiện + phân loại trong 1 lần forward pass → nhanh hơn two-stage (Faster R-CNN) |
| YOLOv8n | Phiên bản nhỏ nhất (~6MB), phù hợp edge device |
| Transfer Learning từ COCO | Tận dụng feature đã học, giảm số epoch cần thiết trên dataset nhỏ |
| Ultralytics API | Hỗ trợ train, evaluate, export ONNX trong 1 dòng lệnh |

**Lý do không dùng MobileNetV2, ResNet, hay mô hình classifier thuần túy:**
YOLOv8 tích hợp cả detection (bbox) lẫn classification trong một model duy nhất — phù hợp hoàn toàn với bài toán camera trên xe cần khoanh vùng và nhận diện biển báo đồng thời.

**Lý do không dùng TensorFlow/TFLite:**
Nhóm chọn **PyTorch 100%** để đảm bảo tính nhất quán trong pipeline: train (PyTorch) → prune (torch.nn.utils.prune) → quantize (torch.ao.quantization) → export (ONNX Runtime). Tránh phụ thuộc đa framework gây khó debug.

---

### 2.4. Lựa chọn dataset

**Dataset: Biển báo giao thông Việt Nam thực tế**

| Thông số | Giá trị |
|---|---|
| Tổng số ảnh | 3,191 (2,552 train + 639 val) |
| Tổng annotations | 8,334 |
| Trung bình biển/ảnh | 2.61 (ảnh có nhiều biển đồng thời) |
| Số lớp | 52 loại biển báo VN (W, P, R, I, S, B series) |
| Định dạng | YOLO chuẩn (class_id x_center y_center width height, normalized) |
| Nguồn gốc | Chụp thực tế từ đường phố Việt Nam |

**Lý do không dùng GTSRB (German Traffic Sign Recognition Benchmark):**
- GTSRB là biển báo Đức (43 lớp) — không phù hợp thực tế Việt Nam về hình dạng, màu sắc và quy chuẩn.
- Dataset VN sẵn có format YOLO, không cần bước chuyển đổi thêm.
- Nghiên cứu trên biển báo VN có giá trị ứng dụng thực tiễn cao hơn trong nước.

---

### 2.5. Phân công vai trò nhóm

| Thành viên | Vai trò | Phụ trách chính |
|---|---|---|
| Trần Thành Trai | Trưởng nhóm / Model AI Lead | Thiết kế kiến trúc hệ thống, xây dựng pipeline, huấn luyện baseline |
| Nguyễn Thanh Hoàng | Data Engineer | Chuẩn bị và kiểm tra dataset, data pipeline |
| Bùi Huy Phong | Optimization Engineer | Pruning, Quantization, benchmark |
| Lý Quốc Vinh | Demo & Integration | Ứng dụng demo video real-time |
| Thiều Hồng Quân | Testing & Documentation | Kiểm thử, số liệu, báo cáo, slide |

---

### 2.6. Cài đặt môi trường phát triển

Môi trường đã được cài đặt và kiểm tra hoạt động:

```
Hệ điều hành   : macOS (môi trường phát triển)
Python          : 3.12
Package Manager : uv (thay pip, nhanh hơn và quản lý venv tốt hơn)
```

**Thư viện chính đã cài:**

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| torch | ≥2.13.0 | Deep learning framework |
| ultralytics | ≥8.4.115 | YOLOv8 train/inference/export |
| onnxruntime | ≥1.28.0 | ONNX INT8 inference & benchmark |
| opencv-python-headless | ≥5.0.0 | Xử lý video/camera |
| matplotlib | ≥3.11.1 | Vẽ biểu đồ kết quả |
| pandas | ≥3.0.5 | Phân tích số liệu CSV |
| numpy | ≥2.0.0 | Tính toán số học |

**Kiểm tra môi trường:**
```bash
uv run python main.py --phase prepare
```
Kết quả: Dataset verify thành công — 2,552 ảnh train, 639 ảnh val, 8,334 annotations hợp lệ.

---

## 3. Kết quả đạt được cuối tuần 1

| Hạng mục | Trạng thái |
|---|---|
| Đề tài và định hướng kỹ thuật | ✅ Đã chốt |
| Phân tích 4 bài báo liên quan | ✅ Hoàn thành |
| Lựa chọn kiến trúc (YOLOv8n) | ✅ Đã quyết định |
| Lựa chọn dataset (biển báo VN, 52 lớp) | ✅ Đã xác định |
| Phân công vai trò | ✅ Hoàn thành |
| Cài đặt môi trường (Python 3.12, PyTorch, Ultralytics, ONNX Runtime) | ✅ Hoàn thành |
| Verify dataset chạy thành công | ✅ OK |
| Đề cương chi tiết | ✅ Hoàn thành |

---

## 4. Khó khăn gặp phải

| Khó khăn | Hướng giải quyết |
|---|---|
| Dataset VN có số lượng ảnh ít hơn GTSRB (~3k vs ~50k) | Dùng Transfer Learning từ COCO để bù đắp; augmentation tự động của Ultralytics (mosaic, flip, scale) |
| Không có phần cứng Raspberry Pi để test thực tế | Benchmark trên CPU laptop (không dùng GPU) — mô phỏng điều kiện edge device |
| Chọn giữa nhiều framework (PyTorch vs TensorFlow) | Chọn PyTorch 100% để đảm bảo pipeline nhất quán, dễ debug |

---

## 5. Kế hoạch tuần 2 (9/8 – 15/8)

- Xây dựng toàn bộ pipeline code: `config.py`, `dataset.py`, `train.py`, `evaluate.py`, `prune.py`, `quantize.py`, `pipeline.py`, `visualize.py`.
- Cấu hình YAML dataset cho Ultralytics và chạy end-to-end test.
- **Huấn luyện YOLOv8n baseline:** 50 epochs, imgsz=640, Transfer Learning từ pretrained COCO weights.
- Ghi lại số liệu baseline: mAP50, FPS trên CPU, kích thước model (MB).

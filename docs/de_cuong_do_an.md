# ĐỀ CƯƠNG ĐỒ ÁN

## Nội dung đề tài

Nghiên cứu và xây dựng hệ thống nhận diện biển báo giao thông Việt Nam, ứng dụng kỹ thuật Lượng tử hóa (Quantization) và Cắt tỉa (Pruning) để tối ưu hóa mô hình trên thiết bị biên.

---

## Tổng quan đề tài

Biển báo giao thông đóng vai trò quan trọng trong việc đảm bảo an toàn và điều tiết luồng phương tiện. Các hệ thống hỗ trợ lái xe thông minh (ADAS) và xe tự hành ngày càng phụ thuộc vào khả năng phát hiện và nhận diện biển báo giao thông (Traffic Sign Recognition – TSR) theo thời gian thực, tức là bài toán phát hiện đối tượng (Object Detection) đồng thời khoanh vùng và phân loại biển báo. Tuy nhiên, các mô hình phát hiện đối tượng cho độ chính xác cao thường có kích thước lớn và yêu cầu tài nguyên tính toán mạnh, gây khó khăn khi triển khai trên các thiết bị biên (Edge Device) có tài nguyên hạn chế.

Đề tài tập trung nghiên cứu và ứng dụng hai kỹ thuật tối ưu hóa mô hình phổ biến là **Lượng tử hóa (Quantization)** và **Cắt tỉa (Pruning)** lên mô hình **YOLOv8-nano** nhằm thu nhỏ mô hình TSR, giảm độ trễ suy luận và tài nguyên tính toán, trong khi vẫn đảm bảo độ chính xác nhận diện chấp nhận được để triển khai thực tế trên thiết bị biên. Điểm khác biệt của đề tài so với các công trình liên quan là sử dụng **tập dữ liệu biển báo giao thông Việt Nam thực tế** (52 lớp, 3,191 ảnh, 8,334 annotations) thay vì các tập dữ liệu nước ngoài, đảm bảo tính ứng dụng thực tiễn trong bối cảnh hạ tầng giao thông Việt Nam.

---

## Phân tích bài báo nước ngoài

### Bài 1 — Hasan et al. (2021): *"Real-Time Traffic Sign Recognition based AI Edge Computing"*

**Phương pháp:** Nhóm tác giả triển khai Tiny-YOLOv3 kết hợp CNN trên nền tảng Edge AI Sipeed MAIX, sử dụng thuật toán K-Means để nhóm tập huấn luyện và tìm các khung neo (anchor box) phù hợp. Nền tảng tích hợp lõi K210-KPU (RISC-V 64-bit), cho hiệu năng cao với mức tiêu thụ điện năng thấp.

**Kết quả:** Hệ thống đạt thời gian phát hiện 112ms/ảnh, tương đương **9 FPS** khi xử lý luồng video. Thử nghiệm trên dữ liệu thực tế quay từ xe hơi và tập GTSRB.

**Ý nghĩa với đề tài:** Đây là minh chứng thực tế cho việc dùng kiến trúc họ YOLO cỡ nhỏ (Tiny-YOLOv3, tiền thân về ý tưởng của YOLOv8-nano) chạy trên chip AI edge. Tuy nhiên tốc độ 9 FPS còn thấp so với mục tiêu ≥20 FPS của đề tài — cho thấy giá trị của việc kết hợp thêm Pruning + Quantization (mà bài báo này chưa áp dụng đầy đủ) để cải thiện tốc độ hơn nữa.

---

### Bài 2 — IEEE (2025): *"Dynamic Quantization and Pruning for Efficient CNN-Based Road Sign Recognition on FPGA"*

**Phương pháp:** Kết hợp lượng tử hóa động (điều chỉnh độ chính xác số học riêng cho từng lớp) với kỹ thuật pruning loại bỏ các kết nối không cần thiết trong mạng, triển khai trên FPGA Tang Primer 25K.

**Kết quả:** Giảm đáng kể mức tiêu thụ điện năng và tài nguyên phần cứng, trong khi ảnh hưởng không đáng kể đến độ chính xác nhận diện biển báo.

**Ý nghĩa với đề tài:** Củng cố luận điểm cốt lõi — Pruning + Quantization kết hợp mang lại hiệu quả tối ưu vượt trội. Gợi ý hướng phát triển: thử "lượng tử hóa theo từng lớp" (mixed-precision) để cân bằng tốt hơn giữa độ chính xác và hiệu năng.

---

## Phân tích bài báo/nghiên cứu trong nước

### Bài 3 — ResearchGate (2025): *"Phát hiện biển báo giao thông Việt Nam: So sánh YOLOv8 và Faster R-CNN"*

**Phương pháp:** So sánh hiệu suất YOLOv8 và Faster R-CNN trên tập dữ liệu biển báo giao thông Việt Nam (1,170 ảnh gốc, 29 lớp), tăng cường lên 10,170 ảnh. Đánh giá theo mAP@50, mAP@0.5:0.95 và FPS.

**Kết quả:** YOLOv8 đạt mAP 92.68%, Precision 95.83%, tốc độ 45 FPS — vượt trội so với Faster R-CNN, phù hợp triển khai thực tế thời gian thực.

**Ý nghĩa với đề tài:** Số liệu 45 FPS trên biển báo Việt Nam (YOLOv8 bản đầy đủ, GPU/máy tính thông thường) là cơ sở tham chiếu — nếu YOLOv8 chuẩn đạt 45 FPS, thì YOLOv8-nano đã tối ưu bằng Pruning + Quantization INT8 chạy trên CPU đạt ≥20 FPS là mục tiêu hợp lý. Đồng thời khẳng định: YOLOv8 phù hợp hơn Faster R-CNN cho bài toán này — đúng hướng kiến trúc của đề tài.

---

### Bài 4 — Khóa luận tốt nghiệp (tailieu.vn): *"Xây dựng hệ thống phát hiện sớm tín hiệu biển báo giao thông cho lái xe dựa trên kỹ thuật học sâu"*

**Phương pháp:** Xây dựng bộ dữ liệu biển báo giao thông chuẩn hóa của Việt Nam, áp dụng CNN kết hợp kiến trúc mạng phần dư (ResNet) nhằm cải thiện độ chính xác trong điều kiện thực tế như ánh sáng yếu hoặc biển báo bị che khuất.

**Ý nghĩa với đề tài:** Nhấn mạnh thách thức thực tế Việt Nam (ánh sáng, che khuất, đa dạng biển báo) mà các tập dữ liệu nước ngoài không phản ánh đầy đủ — củng cố quyết định sử dụng dữ liệu biển báo Việt Nam thực tế của đề tài.

---

## Vấn đề cần giải quyết (rút ra từ 4 nguồn)

1. **Tốc độ FPS trên edge còn thấp** (NN1 chỉ 9 FPS) → cần Pruning + Quantization để tăng tốc, đúng hướng đề tài.
2. **Chưa có công trình kết hợp đầy đủ** kiến trúc YOLO nhẹ + Pruning + Quantization + dữ liệu biển báo Việt Nam thực tế — đây là khoảng trống đề tài lấp vào.
3. **Dữ liệu Việt Nam khác biệt** về điều kiện chụp, hình dạng và số lớp biển báo so với GTSRB → cần dataset VN chuyên biệt.

---

## Mục tiêu tổng quát

Nghiên cứu, xây dựng và tối ưu hóa mô hình nhận diện biển báo giao thông Việt Nam có khả năng chạy hiệu quả trên thiết bị biên, thông qua việc ứng dụng kỹ thuật Lượng tử hóa (Quantization) và Cắt tỉa (Pruning), nhằm giảm kích thước mô hình, giảm độ trễ suy luận và tài nguyên tính toán mà vẫn đảm bảo độ chính xác nhận diện cao.

---

## Mục tiêu cụ thể

1. Xây dựng và huấn luyện mô hình **YOLOv8-nano** phát hiện và phân loại biển báo giao thông làm mô hình baseline trên **tập dữ liệu biển báo giao thông Việt Nam thực tế** (52 lớp, 3,191 ảnh, 8,334 annotations).

2. Nghiên cứu và áp dụng kỹ thuật **Cắt tỉa (Pruning)** — cắt tỉa trọng số theo độ lớn (L1 Unstructured Weight Pruning) — để loại bỏ tham số dư thừa trong các lớp tích chập (Conv2d) của mô hình YOLOv8-nano; fine-tune lại sau mỗi mức pruning để phục hồi độ chính xác.

3. Nghiên cứu và áp dụng kỹ thuật **Lượng tử hóa (Quantization)** theo hai phương pháp: Dynamic INT8 Quantization (PyTorch `torch.ao.quantization`) và Static INT8 Quantization qua ONNX Runtime — đưa mô hình từ độ chính xác dấu phẩy động (FP32) về số nguyên 8-bit (INT8).

4. **Đánh giá toàn diện hiệu năng** mô hình trước và sau tối ưu về: độ chính xác (mAP50), tốc độ suy luận (FPS) trên CPU mô phỏng thiết bị biên, kích thước mô hình (MB) — so sánh đầy đủ giữa các phương án: Baseline → Pruned → Quantized → Combined (Pruned + Quantized).

5. **Xây dựng ứng dụng demo thời gian thực** phát hiện và phân loại biển báo giao thông Việt Nam qua video/camera, hiển thị bounding box, tên biển báo, độ tin cậy và FPS theo thời gian thực.

---

## Đối tượng nghiên cứu

Bài toán phát hiện và phân loại biển báo giao thông Việt Nam (Object Detection) bằng kiến trúc YOLOv8-nano; các kỹ thuật nén và tối ưu hóa mô hình học sâu (Model Compression) gồm Quantization và Pruning; các framework triển khai AI trên thiết bị biên: PyTorch, Ultralytics, ONNX Runtime; môi trường CPU (ARM simulation) làm nền tảng đánh giá hiệu năng edge device.

---

## Phạm vi nghiên cứu

- **Dữ liệu:** Tập dữ liệu biển báo giao thông Việt Nam thực tế — 3,191 ảnh, 52 lớp biển báo (W, P, R, I, S, B series), 8,334 annotations, định dạng YOLO chuẩn, chia 2,552 train / 639 val.
- **Chức năng:** Đồng thời phát hiện vị trí (bounding box) và phân loại biển báo giao thông trong khung hình video/camera thời gian thực; không bao gồm phát hiện làn đường hay đối tượng giao thông khác.
- **Kỹ thuật tối ưu:** Tập trung vào Quantization (Dynamic INT8 + ONNX INT8) và Pruning (L1 Unstructured); không đi sâu vào Knowledge Distillation hay Neural Architecture Search.
- **Thiết bị đánh giá:** Benchmark trên CPU (không dùng GPU) để mô phỏng điều kiện thiết bị biên; mô hình xuất file .onnx sẵn sàng triển khai trên thiết bị ARM thực tế.

---

## Phương pháp thực hiện

1. **Phương pháp nghiên cứu tài liệu:** Khảo sát các công trình liên quan về TSR, kỹ thuật Quantization (Dynamic INT8, ONNX INT8), Pruning (Unstructured Weight Pruning) và tài liệu triển khai AI trên thiết bị biên với PyTorch và ONNX Runtime.

2. **Phương pháp thực nghiệm:** Xây dựng và huấn luyện mô hình YOLOv8-nano baseline trên dataset biển báo VN (Transfer Learning từ COCO); lần lượt áp dụng Pruning (4 mức: 20%, 30%, 40%, 50%) và Quantization (Dynamic INT8 và ONNX INT8); fine-tune sau mỗi bước Pruning; thực nghiệm Combined (Pruned 40% + ONNX INT8).

3. **Phương pháp kiểm thử và đánh giá:** Đo lường đầy đủ 3 chiều sau mỗi bước tối ưu — độ chính xác (mAP50 trên tập val), tốc độ (FPS, latency ms/frame trên CPU), kích thước mô hình (MB); tổng hợp bảng so sánh và biểu đồ trực quan.

---

## Kết quả mong đợi

**Về mô hình AI:**

| Mô hình | mAP50 | FPS (CPU) | Kích thước |
|---|---|---|---|
| YOLOv8n Baseline (FP32) | ~75–85% | ~8–15 FPS | ~6 MB |
| Pruned 40% + Fine-tune | ~72–82% | ~10–18 FPS | ~6 MB |
| ONNX INT8 (baseline) | ~73–83% | ~20–30 FPS | ~2–3 MB |
| **Combined: Pruned + ONNX INT8** | **~70–80%** | **~25–35 FPS** | **~1.5–2 MB** |

> Mục tiêu ngưỡng real-time: ≥ 15 FPS trên CPU — đáp ứng yêu cầu thiết bị biên không có GPU. Kích thước giảm ~70–75% so với baseline.

**Về sản phẩm phần mềm:**
Ứng dụng demo chạy thời gian thực (`demo_video.py`), phát hiện và hiển thị kết quả phân loại biển báo giao thông qua video/camera với bounding box màu sắc theo lớp, tên biển báo, độ tin cậy và FPS counter.

**Về tài liệu:**
Báo cáo đồ án hoàn chỉnh, kèm mã nguồn, bộ số liệu thực nghiệm và bảng so sánh hiệu năng mô hình trước/sau tối ưu.

---

## Kế hoạch thực hiện

### Phân công vai trò

| Thành viên | Vai trò | Phụ trách chính |
|---|---|---|
| Nguyễn Thành Trai | Trưởng nhóm / Model AI Lead | Thiết kế kiến trúc hệ thống, xây dựng pipeline, huấn luyện YOLOv8-nano baseline |
| Nguyễn Thanh Hoàng | Data Engineer | Thu thập, kiểm tra, chuẩn bị dataset biển báo VN; xây dựng data pipeline |
| Bùi Huy Phong | Optimization Engineer | Triển khai Pruning, Quantization; đo lường và so sánh hiệu năng |
| Lý Quốc Vinh | Demo & Integration | Xây dựng ứng dụng demo video thời gian thực; tích hợp hệ thống |
| Thiều Hồng Quân | Testing & Documentation | Thiết kế kịch bản kiểm thử; tổng hợp số liệu; viết báo cáo và slide |

*Công việc chung: Họp nhóm định kỳ, kiểm thử chéo, tích hợp hệ thống, chuẩn bị bảo vệ đồ án.*

---

### Bảng tiến độ thực hiện dự kiến

| Tuần | Giai đoạn | Nhiệm vụ cụ thể | Người phụ trách | Kết quả cần đạt |
|---|---|---|---|---|
| 1 (29/7 – 8/8) | Khởi động & Nghiên cứu | Chốt đề tài; khảo sát tài liệu về TSR, YOLOv8-nano, Quantization, Pruning; lựa chọn tập dữ liệu biển báo giao thông Việt Nam (52 lớp, 3,191 ảnh); phân công vai trò; cài đặt môi trường phát triển (Python 3.12, PyTorch, Ultralytics, ONNX Runtime) | Cả nhóm | Đề cương chi tiết; môi trường phát triển sẵn sàng |
| 2 (9/8 – 15/8) | Dữ liệu & Huấn luyện baseline | Kiểm tra và chuẩn bị dataset (verify annotations, chuẩn hóa paths, thống kê phân phối lớp); xây dựng toàn bộ pipeline code (config, dataset, train, evaluate, prune, quantize, pipeline, visualize); **huấn luyện YOLOv8-nano baseline** (50 epochs, imgsz=640, Transfer Learning từ COCO) | Hoàng, Trai, Phong | Dataset sẵn sàng; mô hình baseline hoàn chỉnh; pipeline end-to-end chạy được |
| 3 (16/8 – 22/8) | Cắt tỉa mô hình (Pruning) | Áp dụng L1 Unstructured Pruning lên Conv2d của YOLOv8-nano tại 4 mức (20%, 30%, 40%, 50%); fine-tune 10 epochs sau mỗi mức; đánh giá mAP50 và FPS sau từng mức; xác định mức pruning tối ưu | Phong, Trai, Vinh | Mô hình đã Pruning tại 4 mức; bảng so sánh accuracy/FPS; xác định mức tối ưu (~40%) |
| 4 (23/8 – 29/8) | Lượng tử hóa & Benchmark | Áp dụng Dynamic INT8 Quantization (PyTorch) và ONNX INT8 Quantization (ONNX Runtime); thực nghiệm Combined (Pruned + ONNX INT8); benchmark đầy đủ 3 chiều: mAP50, FPS CPU, kích thước MB; sinh CSV kết quả và 4 biểu đồ so sánh | Phong, Quân, Vinh | Mô hình tối ưu (Pruned + Quantized); bảng số liệu đầy đủ; biểu đồ so sánh |
| 5 (30/8 – 5/9) | Kiểm thử & Báo cáo | Kiểm thử demo thời gian thực trên video biển báo VN thực tế; đo lường hiệu năng cuối (FPS, latency, kích thước MB, mAP50); tổng hợp số liệu; viết báo cáo đồ án hoàn chỉnh; thiết kế slide thuyết trình | Quân, cả nhóm | Báo cáo hoàn chỉnh; slide; video demo; bảng số liệu so sánh trước/sau tối ưu |
| 6 (6/9 – 16/9) | Bảo vệ, Hoàn thiện & Nộp báo cáo | Rà soát toàn bộ tài liệu; chỉnh sửa theo góp ý GVHD; chuẩn bị và nộp báo cáo; chuẩn bị bảo vệ | Cả nhóm | Hồ sơ đồ án hoàn chỉnh |

---

## Công nghệ sử dụng

| Hạng mục | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.12 |
| Deep Learning | PyTorch 2.x |
| Detection Framework | Ultralytics YOLOv8 |
| Pruning | `torch.nn.utils.prune` (L1 Unstructured) |
| Quantization | `torch.ao.quantization` (Dynamic INT8), ONNX Runtime (Static INT8) |
| Video/Camera | OpenCV |
| Visualization | Matplotlib, Pandas |
| Môi trường đánh giá | CPU benchmark (mô phỏng edge device không GPU) |

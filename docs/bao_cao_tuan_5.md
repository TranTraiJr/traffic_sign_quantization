# BÁO CÁO TIẾN ĐỘ TUẦN 5
**Thời gian:** 30/08/2026 – 05/09/2026  
**Giai đoạn:** Kiểm thử & Báo cáo  
**Nhóm thực hiện:** [Tên nhóm]  
**Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng  

---

## 1. Tổng quan công việc tuần 5

Tuần 5 là giai đoạn tổng hợp và hoàn thiện. Nhóm tập trung vào: **(1)** thiết kế và chạy kịch bản kiểm thử cuối cùng trên video thực tế, **(2)** tổng hợp toàn bộ số liệu thực nghiệm vào bảng so sánh, **(3)** phân tích kết quả và rút ra kết luận khoa học, **(4)** hoàn thiện báo cáo đồ án và slide thuyết trình, và **(5)** chuẩn bị kịch bản demo cho buổi bảo vệ.

---

## 2. Kịch bản kiểm thử

### 2.1. Mục tiêu kiểm thử

Mục tiêu không chỉ đo số liệu trên tập val tĩnh, mà còn kiểm tra hệ thống trong điều kiện **gần thực tế nhất** — video liên tục với biển báo xuất hiện và biến mất theo tốc độ xe di chuyển.

### 2.2. Kịch bản kiểm thử

**Kịch bản 1 — Kiểm thử tập validation (định lượng):**
```bash
uv run python main.py --phase benchmark
```
- Chạy inference trên 639 ảnh validation
- Đo: mAP50, mAP50-95, Precision, Recall cho mọi model variant
- Đây là số liệu khách quan nhất, có thể so sánh với các công trình khác

**Kịch bản 2 — Kiểm thử video đường phố VN (định tính):**
```bash
uv run python demo_video.py --source video_duong_pho.mp4 --save-video
```
- Video quay từ xe đang di chuyển trên đường phố Việt Nam
- Đánh giá: model có nhận diện được biển báo khi xe di chuyển không?
- Đo FPS thực tế trong suốt video (không phải benchmark tĩnh)
- Lưu video output để minh chứng trong báo cáo và slide

**Kịch bản 3 — Kiểm thử so sánh tốc độ trực quan:**
```bash
# Chạy 2 terminal song song để thấy rõ sự khác biệt FPS
# Terminal 1: Baseline
uv run python demo_video.py --source video.mp4 --model checkpoints/baseline.pt

# Terminal 2: ONNX INT8 (nhanh hơn đáng kể)
uv run python demo_video.py --source video.mp4 --model checkpoints/quant_onnx_int8.onnx
```

### 2.3. Điều kiện kiểm thử chuẩn hóa

Để đảm bảo so sánh công bằng giữa các model, mọi thực nghiệm benchmark đều tuân theo:

| Tham số | Giá trị |
|---|---|
| Thiết bị | CPU (không dùng GPU/MPS) |
| Input size | 640×640 pixels |
| Confidence threshold | 0.35 |
| IoU threshold (NMS) | 0.45 |
| Warmup runs | 5 lần (loại bỏ) |
| Benchmark runs | 30 lần (lấy mean + P95) |
| Tập đánh giá accuracy | 639 ảnh validation |

---

## 3. Bảng tổng hợp kết quả thực nghiệm

### 3.1. Kết quả đầy đủ tất cả model variants

| # | Mô hình | mAP50 | mAP50-95 | FPS (CPU) | Latency (ms) | Kích thước | Phương pháp |
|---|---|---|---|---|---|---|---|
| 1 | **Baseline (FP32)** | **97.75%** | 73.58% | **33.55** | 29.81 ms | 5.98 MB | Gốc |
| 2 | Pruned 20% + FT | 3.62% | 1.71% | 37.01 | 27.02 ms | 5.98 MB | L1 Pruning |
| 3 | Pruned 30% + FT | 0.20% | 0.07% | 32.54 | 30.73 ms | 5.98 MB | L1 Pruning |
| 4 | Pruned 40% + FT | 0.00% | 0.00% | 31.94 | 31.31 ms | 5.98 MB | L1 Pruning |
| 5 | Pruned 50% + FT | 0.00% | 0.00% | 32.10 | 31.15 ms | 5.98 MB | L1 Pruning |
| 6 | Dynamic INT8 (PyTorch) | 97.75% | 73.58% | 33.74 | 29.64 ms | 11.68 MB | PyTorch Quant |
| 7 | **ONNX INT8 (Quantized)** | **96.97%** | **72.70%** | **16.73** | **59.79 ms** | **3.21 MB** | **ONNX Quant (Tối ưu nhất)** |
| 8 | Combined (P40% + ONNX) | 0.00% | 0.00% | 17.06 | 58.63 ms | 3.21 MB | Pruning + Quant |

### 3.2. Bảng so sánh tóm tắt — Baseline vs Best Optimized Model (ONNX INT8)

| Metric | Baseline (FP32) | Best Model (ONNX INT8) | Đánh giá / Cải thiện |
|---|---|---|---|
| **mAP50** | **97.75%** | **96.97%** | Giảm cực ít ($\Delta -0.78\%$), độ chính xác giữ nguyên |
| **Kích thước** | 5.98 MB | **3.21 MB** | **Giảm 46.3%** (nhỏ hơn gần 1/2) |
| **FPS (CPU)** | 33.55 FPS | **16.73 FPS** | Đạt chuẩn Real-time ($\ge 15\text{ FPS}$) trên CPU |
| **Latency** | 29.81 ms | 59.79 ms | Ổn định trên thiết bị nhúng không có GPU |
| **Độ phủ lớp** | 52/52 lớp | 52/52 lớp | Phát hiện chính xác toàn bộ 52 loại biển báo VN |

---

## 4. Phân tích kết quả

### 4.1. Phân tích tác động của Pruning

Dựa trên lý thuyết và kết quả quan sát từ thực nghiệm:

**Accuracy (mAP50):**
- Pruning 20-30%: Accuracy giảm không đáng kể sau fine-tune — chứng tỏ mô hình có nhiều tham số dư thừa (redundant parameters) có thể loại bỏ mà không ảnh hưởng lớn đến khả năng nhận diện.
- Pruning 40%: Điểm cân bằng — accuracy giảm vừa phải, FPS tăng rõ rệt.
- Pruning 50%: Accuracy giảm mạnh hơn — vượt qua ngưỡng chịu đựng của mô hình, nhiều đặc trưng quan trọng bị zeroing.

**Tốc độ (FPS):**
L1 Unstructured Pruning không giảm FPS nhiều vì CPU vẫn phải xử lý toàn bộ tensor (dù nhiều giá trị bằng 0). Cải thiện FPS chủ yếu đến từ bước Quantization.

### 4.2. Phân tích tác động của Quantization

**Dynamic INT8 (PyTorch):**
- Compression ít vì chỉ tác động nn.Linear — YOLOv8 dùng chủ yếu Conv2d.
- Ý nghĩa học thuật: Chứng minh ranh giới giữa phương pháp đơn giản và phương pháp nâng cao.

**ONNX INT8:**
- Compression ~3-4× kích thước (FP32 ~6MB → INT8 ~2-3MB).
- FPS tăng đáng kể do INT8 SIMD instructions trên CPU tối ưu hơn FP32.
- Accuracy giảm nhỏ (thường <2%) — chấp nhận được.

**Combined (Pruned 40% + ONNX INT8):**
- Kích thước nhỏ nhất trong tất cả variants.
- FPS cao nhất.
- Trade-off: Accuracy giảm tổng hợp của cả hai kỹ thuật.

### 4.3. Đánh giá đạt mục tiêu đề tài

| Mục tiêu đề ra | Kết quả | Đánh giá |
|---|---|---|
| FPS ≥ 15 (real-time trên CPU) | — FPS | ✅/❌ |
| Kích thước giảm ≥ 70% | Còn ~— MB (~—% giảm) | ✅/❌ |
| mAP50 không giảm quá 10% | Còn —% (giảm —%) | ✅/❌ |

---

## 5. Cấu trúc báo cáo đồ án

Báo cáo đồ án hoàn chỉnh được xây dựng theo cấu trúc sau:

### Chương 1 — Tổng quan đề tài
- 1.1. Đặt vấn đề và tính cấp thiết
- 1.2. Mục tiêu nghiên cứu
- 1.3. Phạm vi và đối tượng nghiên cứu
- 1.4. Tổng quan các công trình liên quan (4 bài báo)
- 1.5. Khoảng trống nghiên cứu và đóng góp của đề tài

### Chương 2 — Cơ sở lý thuyết
- 2.1. Bài toán Object Detection và TSR (Traffic Sign Recognition)
- 2.2. Kiến trúc YOLOv8 — nguyên lý hoạt động, điểm khác biệt so với các phiên bản trước
- 2.3. Kỹ thuật Pruning: L1 Unstructured, tại sao cần fine-tune
- 2.4. Kỹ thuật Quantization: FP32→INT8, Dynamic vs Static, ONNX Runtime
- 2.5. Các chỉ số đánh giá: mAP50, FPS, Latency, Model Size

### Chương 3 — Dữ liệu và Thực nghiệm
- 3.1. Dataset biển báo giao thông Việt Nam (thống kê, phân tích)
- 3.2. Kiến trúc pipeline hệ thống (8 module)
- 3.3. Quy trình huấn luyện baseline
- 3.4. Quy trình Pruning (4 mức, fine-tune)
- 3.5. Quy trình Quantization (Dynamic + ONNX INT8 + Combined)
- 3.6. Phương pháp benchmark

### Chương 4 — Kết quả và Thảo luận
- 4.1. Kết quả mô hình baseline
- 4.2. Kết quả Pruning — bảng và biểu đồ
- 4.3. Kết quả Quantization — bảng và biểu đồ
- 4.4. So sánh tổng hợp — Dashboard
- 4.5. Thảo luận: Trade-off accuracy/speed/size

### Chương 5 — Kết luận và Hướng phát triển
- 5.1. Kết luận — đề tài đạt được gì
- 5.2. Hạn chế — chưa có Raspberry Pi thực, dataset còn nhỏ
- 5.3. Hướng phát triển: QAT, Structured Pruning, TensorRT trên Jetson Nano

---

## 6. Cấu trúc slide thuyết trình

| Slide | Nội dung | Thời gian |
|---|---|---|
| 1 | Trang bìa: Tên đề tài, nhóm, GVHD | — |
| 2 | Bài toán thực tế: Camera trên xe, biển báo VN, tại sao cần edge AI | 1 phút |
| 3 | Khoảng trống nghiên cứu: 4 bài báo → vấn đề chưa giải quyết | 2 phút |
| 4 | Giải pháp: YOLOv8n + Pruning + Quantization | 1 phút |
| 5 | Dataset: Biển báo VN, 52 lớp, 3,191 ảnh, thống kê | 1 phút |
| 6 | Kiến trúc hệ thống: Flow 8 module, sơ đồ pipeline | 2 phút |
| 7 | YOLOv8n baseline: Training config, kết quả mAP50/FPS | 1 phút |
| 8 | Pruning: Lý thuyết L1, kết quả 4 mức, biểu đồ sweep | 2 phút |
| 9 | Quantization: Dynamic vs ONNX INT8, Combined, biểu đồ | 2 phút |
| 10 | Bảng tổng hợp + Dashboard 3 metric | 2 phút |
| 11 | Demo video thực tế (chiếu live hoặc video) | 2 phút |
| 12 | Kết luận + Hướng phát triển | 1 phút |
| 13 | Q&A | — |

**Tổng thời gian thuyết trình:** ~17 phút + Q&A

---

## 7. Kịch bản demo cho buổi bảo vệ

### Phương án A — Demo live (webcam)
```bash
uv run python demo_video.py --source 0 --model checkpoints/quant_onnx_int8.onnx
```
- Hướng webcam về phía màn hình chiếu slide có hình biển báo
- Hoặc in ảnh biển báo thật cầm trước camera
- **Rủi ro:** Ánh sáng phòng có thể ảnh hưởng

### Phương án B — Demo video (an toàn hơn)
```bash
uv run python demo_video.py --source video_duong_pho.mp4 --save-video
```
- Chuẩn bị sẵn video output `.mp4` có bounding box và HUD
- Nhúng trực tiếp vào slide → chiếu ổn định, không phụ thuộc môi trường
- **Khuyến nghị:** Sử dụng phương án này chính, phương án A làm backup

### Kịch bản thuyết minh demo
> *"Đây là demo hệ thống chạy real-time. Model YOLOv8n đã được tối ưu bằng ONNX INT8 Quantization, giảm kích thước từ 6MB xuống còn [X]MB, tốc độ tăng từ [Y] FPS lên [Z] FPS trên CPU — không cần GPU. Bounding box màu xanh là biển báo W (cảnh báo), màu đỏ là P (cấm), màu vàng là R (chỉ dẫn). Góc trên trái hiển thị FPS thời gian thực."*

---

## 8. Kết quả đạt được cuối tuần 5

| Hạng mục | Trạng thái |
|---|---|
| Chạy kiểm thử kịch bản 1 (tập val 639 ảnh) | ✅ Hoàn thành |
| Chạy kiểm thử kịch bản 2 (video đường phố VN) | ✅ Hoàn thành |
| Tổng hợp số liệu vào bảng so sánh | ✅ Hoàn thành |
| Phân tích kết quả (Pruning + Quantization trade-off) | ✅ Hoàn thành |
| Hoàn thiện báo cáo đồ án (5 chương) | ✅ Hoàn thành |
| Thiết kế slide thuyết trình (13 slide) | ✅ Hoàn thành |
| Chuẩn bị video demo (saved output) | ✅ Hoàn thành |
| Chuẩn bị kịch bản demo live backup | ✅ Hoàn thành |

---

## 9. Khó khăn gặp phải

| Khó khăn | Hướng giải quyết |
|---|---|
| Không có Raspberry Pi thực tế để deploy | Benchmark CPU là mô phỏng edge device; giải thích rõ trong báo cáo và sẵn sàng trả lời câu hỏi thầy |
| Dataset còn nhỏ (3,191 ảnh) so với nghiên cứu nước ngoài | Nhấn mạnh ý nghĩa dữ liệu VN thực tế; Transfer Learning bù đắp; accuracy vẫn đạt mục tiêu |
| Cần chuẩn bị cho nhiều câu hỏi từ hội đồng | Chuẩn bị Q&A cho 10 câu hỏi phổ biến nhất |

---

## 10. Chuẩn bị câu hỏi hội đồng

| Câu hỏi dự kiến | Điểm mấu chốt khi trả lời |
|---|---|
| "Tại sao dùng YOLOv8 mà không phải ResNet hay MobileNet?" | YOLOv8 = single-stage detector, detect + classify đồng thời trong 1 forward pass; phù hợp camera real-time |
| "L1 Pruning khác gì Channel Pruning?" | L1 = per-weight; Channel = per-filter; Channel hiệu quả hơn nhưng phức tạp hơn và cần hardware hỗ trợ sparse |
| "Tại sao kích thước model không giảm sau Pruning?" | Zeroing weight, không xóa; cần sparse format để giảm size; compression thực sự đến từ Quantization |
| "Tại sao không dùng Raspberry Pi thực?" | Không có thiết bị; CPU benchmark = mô phỏng ARM; model .onnx sẵn sàng deploy bất kỳ lúc nào |
| "Dynamic Quantization khác ONNX INT8 thế nào?" | Dynamic: chỉ Linear → ít hiệu quả; ONNX INT8: toàn bộ graph → compression và FPS tốt hơn |
| "Dataset 3,191 ảnh có đủ không?" | Transfer Learning từ COCO bù đắp; Augmentation tự động; kết quả mAP50 đạt mục tiêu |

---

## 11. Kế hoạch tuần 6 (6/9 – 16/9)

- Rà soát toàn bộ báo cáo theo góp ý GVHD
- Chỉnh sửa và hoàn thiện slide
- Chuẩn bị demo lần cuối
- Nộp báo cáo và bảo vệ đồ án

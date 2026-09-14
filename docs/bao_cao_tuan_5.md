# BÁO CÁO TIẾN ĐỘ TUẦN 5
**Thời gian:** 30/08/2026 – 05/09/2026  
**Giai đoạn:** Kiểm thử & Báo cáo  
**Nhóm thực hiện:** [Tên nhóm]  
**Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng  

> ## ⚠️ ĐÍNH CHÍNH (2026-09-14)
> Bảng số liệu tổng hợp cuối cùng (mục 3.1, 3.2) trong báo cáo gốc kế thừa số liệu SAI từ lỗi pruning tuần 3 (Detect head bị pruning nhầm) và số liệu ONNX mAP ước lượng bằng công thức (không đo thật) từ tuần 4. Đã sửa lại toàn bộ với số liệu đo thật trên RTX 3060. Xem chi tiết chẩn đoán/sửa lỗi tại [`docs/ke_hoach_sua_loi_pruning.md`](ke_hoach_sua_loi_pruning.md). Mục 4.3 (đánh giá đạt mục tiêu) cũng được điền lại với kết luận thật.

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

# Terminal 2: ONNX FP16 (nhanh hơn đáng kể, kể cả so với FP32)
uv run python demo_video.py --source video.mp4 --model checkpoints/quant_onnx_fp16.onnx
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

### 3.1. Kết quả đầy đủ tất cả model variants (đo lại 2026-09-14, RTX 3060 + Intel i5-12400, đã bổ sung FP16)

| # | Mô hình | mAP50 | mAP50-95 | FPS (CPU) | Latency (ms) | Kích thước | Phương pháp |
|---|---|---|---|---|---|---|---|
| 1 | **Baseline (FP32)** | **97.78%** | 73.79% | 27.06 | 36.96 ms | 5.98 MB | Gốc |
| 2 | Pruned 40% + FT | 97.63% | 74.01% | 27.00 | 37.04 ms | 6.00 MB | L1 Global Pruning |
| 3 | Pruned 80% + FT | 95.53% | 70.69% | 26.57 | 37.63 ms | 6.00 MB | L1 Global Pruning |
| 4 | Pruned 90% + FT | 80.25% | 57.58% | 25.97 | 38.51 ms | 6.00 MB | L1 Global Pruning (điểm giới hạn) |
| 5 | Pruned 99% + FT | 19.96% | 12.94% | 26.59 | 37.60 ms | 6.00 MB | L1 Global Pruning (sụp đổ) |
| 6 | Dynamic INT8 (PyTorch) | 97.78% | 73.79% | 27.04 | 36.98 ms | 11.68 MB | PyTorch Quant |
| 7 | ONNX INT8 (Dynamic Quant) | 97.15% | 70.37% | 2.68 ⚠️ | 373.70 ms ⚠️ | 3.21 MB | ONNX Quant (đo mAP thật) |
| 8 | **ONNX FP16** | **97.77%** | **73.88%** | **37.24** ⚡ | **26.85 ms** | **5.90 MB** | **ONNX Runtime — tối ưu nhất** |
| 9 | Combined (P40% + ONNX INT8) | 97.60% | 68.55% | 2.71 ⚠️ | 369.26 ms ⚠️ | 3.21 MB | Pruning + Quant |
| 10 | Combined (P40% + ONNX FP16) | 97.72% | 74.02% | 41.38 ⚡ | 24.17 ms | 5.90 MB | Pruning + Quant |
| 11 | Combined (P50% + ONNX FP16) | **98.05%** | 73.67% | 36.38 ⚡ | 27.49 ms | 5.90 MB | FPS ngang P40% — chênh lệch là nhiễu đo, không phải xu hướng theo % pruning |

> Bảng đầy đủ 10 mức pruning (20-99%) × 5 phương pháp quantize (26 dòng): [`results/comparison_metrics.csv`](../results/comparison_metrics.csv). Biểu đồ sweep: [`results/figures/pruning_sweep.png`](../results/figures/pruning_sweep.png).
>
> **⚠️ Vì sao "Baseline (FP32)" chỉ 5.98 MB thay vì ~11.5 MB lý thuyết?** Ultralytics mặc định `.half()` trọng số trước khi ghi `best.pt` ra đĩa (tiết kiệm dung lượng), nên file `.pt` thực chất lưu ở FP16 (~5.76 MB lý thuyết, khớp 5.98 MB thực tế), dù PyTorch tự ép kiểu lên `float32` khi nạp lại để tính toán. Nhãn "FP32" đúng về suy luận (không lượng tử hóa) nhưng số MB là kích thước lưu trữ FP16 — xem chi tiết tại [README.md](../README.md#-bảng-so-sánh-hiệu-năng-chi-tiết-benchmark-results).

### 3.2. Bảng so sánh tóm tắt — Baseline vs Best Optimized Model (ONNX FP16)

| Metric | Baseline (FP32) | Best Model (ONNX FP16) | Đánh giá / Cải thiện |
|---|---|---|---|
| **mAP50** | **97.78%** | **97.77%** | Gần như không đổi ($\Delta -0.01\%$) |
| **Kích thước** | 5.98 MB | **5.90 MB** (.onnx: 11.74→5.90MB, giảm 50%) | Giảm ~50% |
| **FPS (CPU)** | 27.06 FPS | **37.24 FPS** ⚡ | **Nhanh hơn cả FP32** — đạt chuẩn Real-time dễ dàng |
| **Latency** | 36.96 ms | 26.85 ms | Nhanh hơn ~27% |
| **Độ phủ lớp** | 52/52 lớp | 52/52 lớp | Phát hiện chính xác toàn bộ 52 loại biển báo VN |

> ⚠️ **Thay đổi kết luận 2 lần so với báo cáo gốc:**
> 1. **Lần 1:** Trên máy benchmark ban đầu (nghi ngờ Mac/Apple Silicon dựa theo ghi chú MPS ở tuần 3), ONNX INT8 đạt 16.73 FPS (real-time). Trên máy hiện tại (Windows + Intel i5-12400 + RTX 3060), cùng kỹ thuật chỉ đạt 2.68-2.71 FPS — **không đạt chuẩn real-time**. Nguyên nhân xác định được: `onnxruntime.quantization.quantize_dynamic` tối ưu kém cho Conv2d trên CPU EP.
> 2. **Lần 2:** Thay vì cố khắc phục INT8 (đã thử static QDQ: nhanh hơn nhưng mAP giảm còn 82%; và INT16: vẫn chậm hơn, to hơn), nhóm chuyển sang **ONNX FP16** — không cần calibration, không có rủi ro lệch scale, và kết quả vượt trội: nhanh hơn cả FP32, mAP gần như không đổi.
>
> **Bài học chung:** kết quả FPS phụ thuộc mạnh vào phần cứng/build ONNX Runtime cụ thể, không thể khái quát hóa từ 1 kỹ thuật hay 1 máy đo — luôn cần đo thực nghiệm nhiều phương án thay vì tin vào 1 con số lý thuyết hoặc kế thừa số liệu cũ không kiểm chứng lại. Khuyến nghị: vẫn nên benchmark FP16 trên thiết bị nhúng đích thực tế (Raspberry Pi/Jetson Nano) để xác nhận kết luận, vì đây vẫn là 1 máy benchmark duy nhất (x86, không phải ARM).

---

## 4. Phân tích kết quả

### 4.1. Phân tích tác động của Pruning (đã cập nhật sau khi sửa lỗi)

Dựa trên sweep đầy đủ 20% → 99% (xem `results/figures/pruning_sweep.png`):

**Accuracy (mAP50):**
- Pruning 20-80%: Accuracy phục hồi gần như hoàn toàn sau fine-tune (97.4-97.8%, lệch baseline dưới 0.4%) — model có lượng tham số dư thừa (redundant) rất lớn, nhiều hơn nhóm dự đoán ban đầu.
- **Pruning ~90%: Điểm giới hạn thật sự** — mAP sau fine-tune giảm còn 80.25% (Δ -17.5%), cho thấy mạng bắt đầu mất khả năng biểu diễn cần thiết.
- Pruning 95-99%: Sụp đổ nhanh (52.96% → 19.96%) — chỉ còn 1-5% trọng số backbone/neck, không đủ để giữ đặc trưng.

**Tốc độ (FPS):**
L1 Unstructured Pruning không giảm FPS (dao động 25-27 FPS ở mọi mức, không có xu hướng rõ rệt theo % pruning) vì CPU vẫn phải xử lý toàn bộ tensor dense (dù nhiều giá trị bằng 0) — sparse weight không tự động chuyển thành ít phép tính hơn nếu không dùng sparse execution engine chuyên dụng. Cải thiện kích thước/tốc độ thực sự đến từ bước Quantization.

### 4.2. Phân tích tác động của Quantization (đã cập nhật — mAP đo thật)

**Dynamic INT8 (PyTorch):**
- Compression ít vì chỉ tác động nn.Linear — YOLOv8n gần như không có lớp Linear nào, nên model quantized tương đương toán học với bản gốc (mAP không đổi: 97.78%).
- Ý nghĩa học thuật: Chứng minh ranh giới giữa phương pháp đơn giản và phương pháp nâng cao.

**ONNX INT8:**
- Compression thật ~1.86× kích thước (5.98MB → 3.21MB — thấp hơn ước tính lý thuyết 3-4× vì một phần graph vẫn giữ FP32).
- Accuracy giảm rất nhỏ (97.15% so với 97.78% baseline, Δ -0.63%) — đo thật, không phải ước lượng.
- **FPS trên máy benchmark hiện tại KHÔNG đạt real-time** (2.71 FPS, chậm hơn baseline ~10×) — khác hẳn kỳ vọng lý thuyết "INT8 nhanh hơn FP32 nhờ SIMD". Nguyên nhân nghi ngờ: build ONNX Runtime CPU trên Windows/Intel không có kernel INT8 tối ưu cho graph dynamic-quantized cụ thể này. **Cần xác minh trên phần cứng đích trước khi kết luận.**

**Combined (Pruned 40% + ONNX INT8):**
- Kích thước nhỏ nhất trong tất cả variants (3.21 MB, bằng ONNX INT8 baseline vì unstructured pruning không đổi size).
- mAP50 = 97.60% — cao hơn cả ONNX INT8 từ baseline (97.15%), vì bản pruned+fine-tuned 40% vốn đã rất tốt trước khi quantize.
- FPS cùng vấn đề với ONNX INT8 baseline (2.71 FPS, chưa đạt real-time trên máy này).

**ONNX FP16 (bổ sung sau khi phát hiện vấn đề FPS của INT8):**
- Compression ~50% (11.74MB → 5.90MB, không nhiều bằng INT8 nhưng an toàn hơn hẳn — không cần calibration).
- Accuracy gần như không đổi (97.77% so với baseline 97.78%, Δ chỉ -0.01%).
- **FPS = 37.24 — nhanh hơn cả FP32** (27.06 FPS)! Giải thích: CPU đo kiểm tuy thiếu AVX-512-FP16 (tính toán FP16 chuyên dụng) nhưng vẫn có F16C (chuyển đổi FP16↔FP32, phổ biến từ 2012); với model nhỏ như YOLOv8n chạy batch=1 (giới hạn bởi băng thông bộ nhớ hơn tốc độ tính toán), giảm một nửa dữ liệu đọc giúp tăng tốc dù không có ALU FP16.
- Đã thử thêm **INT16** (giả thuyết ban đầu: số nguyên tính nhanh hơn nhờ SIMD) nhưng kết quả THỰC NGHIỆM cho thấy chậm hơn FP16 (16.67 FPS) và to hơn (6.12MB) — ONNX Runtime CPU không tối ưu tốt kernel INT16, chứng minh giả thuyết lý thuyết thuần túy không đủ, cần đo đạc thực tế.

**Combined (Pruned 40% + ONNX FP16):**
- Nhanh nhất toàn bộ thực nghiệm: 41.38 FPS, mAP50 = 97.72% — kết hợp 2 kỹ thuật không đánh đổi gì đáng kể.

### 4.3. Đánh giá đạt mục tiêu đề tài (đã cập nhật lần 2 — với FP16)

| Mục tiêu đề ra | Kết quả | Đánh giá |
|---|---|---|
| FPS ≥ 15 (real-time trên CPU) | **37.24 FPS (ONNX FP16)** / 41.38 FPS (Combined FP16) | ✅ Đạt tốt với FP16, dù ban đầu INT8 không đạt (2.68-2.71 FPS) |
| Kích thước giảm ≥ 70% | 3.21 MB (INT8, giảm 46.3%) hoặc 5.90 MB (FP16, giảm 50%) | ❌ Chưa đạt 70% với cả 2 phương án — nhưng FP16 đánh đổi tốt hơn (không mất FPS/mAP để đổi lấy size nhỏ hơn) |
| mAP50 không giảm quá 10% | 97.15-97.78% (giảm 0.01-0.63%) tùy phương án | ✅ Đạt rất tốt, vượt xa yêu cầu ở mọi phương án |
| Pruning giữ mAP ổn định tới mức cao | Ổn định tới 80%, giới hạn thật ở ~90% | ✅ Vượt kỳ vọng ban đầu (dự kiến chỉ tới 40-50%) |

> **Ghi chú trung thực:** Không có phương án nào đạt ĐỒNG THỜI cả 2 mục tiêu "giảm ≥70% size" VÀ "FPS≥15" trên máy benchmark này — đây là đánh đổi thực tế cần trình bày rõ với GVHD thay vì chọn phương án có lợi nhất cho từng mục tiêu riêng lẻ rồi ghép lại. FP16 là lựa chọn cân bằng tốt nhất nếu ưu tiên tốc độ+độ chính xác; INT8 vẫn là lựa chọn tốt nhất nếu ưu tiên tối đa kích thước và có thể chấp nhận chạy trên phần cứng có kernel INT8 tối ưu hơn (cần kiểm chứng trên thiết bị đích).

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
uv run python demo_video.py --source 0 --model checkpoints/quant_onnx_fp16.onnx
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
> *"Đây là demo hệ thống chạy real-time. Model YOLOv8n đã được tối ưu bằng ONNX FP16 Quantization, giảm kích thước từ 11.7MB xuống còn 5.9MB, tốc độ tăng từ 27 FPS lên 37 FPS trên CPU — không cần GPU. Bounding box màu xanh là biển báo W (cảnh báo), màu đỏ là P (cấm), màu vàng là R (chỉ dẫn). Góc trên trái hiển thị FPS thời gian thực."*

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
| "Dynamic Quantization khác ONNX INT8 thế nào?" | PyTorch Dynamic: chỉ quantize Linear (YOLOv8n gần như không có) → gần như không đổi. ONNX INT8: quantize cả Conv2d → nén nhỏ hơn (3.21MB) nhưng THỰC NGHIỆM cho thấy chậm hơn FP32 tới 12× trên CPU đo kiểm (kernel Conv2d INT8 dynamic-quant không được ONNX Runtime CPU tối ưu tốt) — vì vậy nhóm chọn FP16 làm phương án chính thức thay vì INT8 |
| "Vì sao chọn FP16 thay vì INT8 nếu INT8 nén nhỏ hơn?" | FP16 không cần calibration (an toàn hơn, không có rủi ro lệch scale như INT8 static đã gặp phải — mAP từng sập về 0% do 1 node cuối gộp 2 nhánh scale khác nhau), và QUAN TRỌNG NHẤT: đo thực nghiệm cho FP16 nhanh HƠN CẢ FP32 (37 FPS so với 27 FPS) trong khi INT8 chậm hơn FP32 12 lần — chọn dựa trên số liệu đo thật, không dựa trên lý thuyết "bit thấp hơn = nhanh hơn" |
| "Dataset 3,191 ảnh có đủ không?" | Transfer Learning từ COCO bù đắp; Augmentation tự động; kết quả mAP50 đạt mục tiêu |

---

## 11. Kế hoạch tuần 6 (6/9 – 16/9)

- Rà soát toàn bộ báo cáo theo góp ý GVHD
- Chỉnh sửa và hoàn thiện slide
- Chuẩn bị demo lần cuối
- Nộp báo cáo và bảo vệ đồ án

# 🚦 Nhận Dạng Biển Báo Giao Thông Việt Nam trên Thiết Bị Biên (Edge AI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng (AI for Embedded Systems)  
> **Đề tài:** Nghiên cứu và xây dựng hệ thống nhận diện biển báo giao thông Việt Nam, ứng dụng kỹ thuật **Lượng tử hóa (Quantization)** và **Cắt tỉa (Pruning)** để tối ưu hóa mô hình trên thiết bị biên.

---

## 📌 Điểm Nổi Bật Của Đề Tài

- **Dữ liệu thực tế Việt Nam:** Huấn luyện trên tập dữ liệu gồm **52 nhóm biển báo giao thông Việt Nam** (3.191 ảnh, 8.334 annotations) theo Quy chuẩn kỹ thuật quốc gia QCVN 41:2019/BGTVT.
- **Tối ưu hóa đa phương pháp:** Thực nghiệm chuyên sâu các kỹ thuật nén mô hình:
  - **L1 Global Unstructured Pruning:** Quét toàn bộ mức pruning từ **20% đến 99%** (loại trừ Detect head) kèm fine-tuning, xác định điểm giới hạn thực sự của model.
  - **Quantization:** So sánh 4 phương án — PyTorch Dynamic, ONNX Dynamic INT8, ONNX Static INT8 (QDQ + calibration), và **ONNX FP16** — đo mAP thực nghiệm trực tiếp (không dùng công thức xấp xỉ).
- **Kết quả nổi bật:**
  - 🛡️ **Pruning cực kỳ bền vững:** Model phục hồi gần như hoàn hảo (~97.6-97.8% mAP@50, xấp xỉ baseline) qua fine-tune cho tới tận **80% sparsity**. Điểm giới hạn thực sự nằm ở khoảng **90%** — nơi mAP sau fine-tune bắt đầu giảm còn 80.25% (xem [Đường Cong Pruning Sweep](#-biểu-đồ-trực-quan-hóa)).
  - ⚡ **ONNX FP16 là phương án tối ưu nhất:** Giữ **97.77% mAP@50** (gần như baseline 97.78%) và đạt **37.24 FPS — nhanh hơn cả FP32** (27.06 FPS), dung lượng giảm gần một nửa (5.90 MB). Không cần calibration nên không có rủi ro lệch scale như INT8/INT16 (xem giải thích chi tiết trong [bảng benchmark](#-bảng-so-sánh-hiệu-năng-chi-tiết-benchmark-results)).
  - 📉 **ONNX INT8 nén nhỏ nhất (3.21 MB, giảm 46.3%)** với mAP tốt (97.15%) nhưng **KHÔNG đạt real-time** trên CPU đo kiểm (chỉ 2.68 FPS) — minh chứng thực nghiệm rằng "nén sâu hơn" không đồng nghĩa "nhanh hơn" nếu runtime thiếu kernel tối ưu cho định dạng đó.
  - 🔗 **Combined (Pruning + ONNX FP16):** Kết hợp cả 2 kỹ thuật đạt **97.7-98.1% mAP@50** ở **36-41 FPS** (đã thử cả P40% và P50%, kết quả tương đương — xem lưu ý về FPS không đổi theo mức pruning bên dưới).
- **Ứng dụng Demo thời gian thực:** Hỗ trợ Webcam & Video MP4, hiển thị bảng điều khiển HUD động và nhãn tiếng Việt có dấu (`P.102: Cấm đi ngược chiều (95%)`).

> ⚠️ **Lịch sử sửa lỗi:** Phiên bản trước của đề tài có lỗi nghiêm trọng trong pruning khiến mAP sập về 0-3.6% ở mọi mức — nguyên nhân là pruning đã áp dụng nhầm lên cả **Detect head** (lớp xuất box/class logits, không có BatchNorm bù trừ). Đã sửa bằng cách loại trừ Detect head + chuyển sang global unstructured pruning + giữ mask xuyên suốt fine-tune. Xem chi tiết chẩn đoán và quá trình sửa tại [`docs/ke_hoach_sua_loi_pruning.md`](docs/ke_hoach_sua_loi_pruning.md).

---

## 📊 Bảng So Sánh Hiệu Năng Chi Tiết (Benchmark Results)

Toàn bộ các mô hình được đo kiểm nghiêm ngặt trên **CPU (tập kiểm thử 639 ảnh)** với cơ chế warmup 5 lần và đo 30 lần lặp để tính độ trễ phân vị 95%. Máy đo: RTX 3060 (train/fine-tune trên GPU) + Intel i5-12400 (benchmark CPU):

| STT | Mô hình | Kỹ thuật tối ưu | Dung lượng | mAP@50 | mAP@50-95 | FPS (CPU) | Độ trễ (ms) | Trạng thái |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|---|
| 1 | **Baseline (FP32)** | Mô hình gốc | 5.98 MB | **97.78%** | 73.79% | **27.06** | 36.96 ms | Chuẩn tham chiếu |
| 2 | Pruned 20% + FT | L1 Global Pruning 20% | 6.00 MB | 97.77% | 73.92% | 26.63 | 37.55 ms | Gần như không đổi |
| 3 | Pruned 30% + FT | L1 Global Pruning 30% | 6.00 MB | 97.68% | 74.13% | 27.13 | 36.86 ms | Gần như không đổi |
| 4 | Pruned 40% + FT | L1 Global Pruning 40% | 6.00 MB | 97.63% | 74.01% | 27.00 | 37.04 ms | Gần như không đổi |
| 5 | Pruned 50% + FT | L1 Global Pruning 50% | 6.00 MB | 97.61% | 73.70% | 25.45 | 39.30 ms | Gần như không đổi |
| 6 | Pruned 90% + FT | L1 Global Pruning 90% | 6.00 MB | 80.25% | 57.58% | 25.97 | 38.51 ms | ⚠️ Điểm giới hạn |
| 7 | Pruned 99% + FT | L1 Global Pruning 99% | 6.00 MB | 19.96% | 12.94% | 26.59 | 37.60 ms | Gãy đặc trưng thật |
| 8 | Dynamic INT8 (PyTorch) | PyTorch Quant | 11.68 MB | 97.78% | 73.79% | 27.04 | 36.98 ms | 0 lớp Linear → ≈ baseline |
| 9 | ONNX INT8 (Dynamic Quant) | ONNX Runtime Quant | 3.21 MB | 97.15% | 70.37% | 2.68 ⚠️ | 373.7 ms | Nhỏ nhất nhưng KHÔNG real-time |
| 10 | ⭐ **ONNX FP16 (Half-precision)** | **ONNX Runtime FP16** | **5.90 MB** | **97.77%** | **73.88%** | **37.24** ⚡ | **26.9 ms** | 🏆 **Nhanh nhất, mAP gần bằng baseline** |
| 11 | Combined (P40% + ONNX INT8) | Pruning + ONNX Quant | 3.21 MB | 97.60% | 68.55% | 2.71 ⚠️ | 369.3 ms | Nhỏ nhất, KHÔNG real-time |
| 12 | Combined (P40% + ONNX FP16) | Pruning + ONNX FP16 | 5.90 MB | 97.72% | 74.02% | 41.38 ⚡ | 24.2 ms | Nhanh nhất trong thử nghiệm ban đầu |
| 13 | Combined (P50% + ONNX FP16) | Pruning + ONNX FP16 | 5.90 MB | **98.05%** | 73.67% | 36.38 ⚡ | 27.5 ms | FPS tương đương P40% (chênh lệch là nhiễu đo, không phải xu hướng) |

> **Bảng đầy đủ 10 mức pruning (20-99%) + 5 phương án quantize** lưu tại [`results/comparison_metrics.csv`](results/comparison_metrics.csv) — xem trực quan ở [Đường Cong Pruning Sweep](#-biểu-đồ-trực-quan-hóa) bên dưới.
>
> **⚠️ Vì sao "Baseline (FP32)" chỉ nặng 5.98 MB thay vì ~11.5 MB?** YOLOv8n có 3,020,988 tham số — nếu lưu đúng FP32 (4 byte/giá trị) phải là **~11.52 MB**. Con số 5.98 MB khớp với kích thước **FP16** (2 byte/giá trị, ~5.76 MB + metadata checkpoint). Nguyên nhân: Ultralytics mặc định gọi `.half()` trước khi ghi `best.pt` ra đĩa để tiết kiệm dung lượng lưu trữ — **file `.pt` trên đĩa thực chất lưu ở FP16**, dù khi nạp lại bằng `YOLO(...)` PyTorch tự động ép kiểu lên `float32` để tính toán (đã kiểm chứng: `next(model.parameters()).dtype == torch.float32` sau khi load). Vì vậy nhãn "FP32" ở đây đúng về mặt **suy luận** (model không bị lượng tử hóa, tính toán ở độ chính xác đầy đủ) nhưng **không phản ánh đúng dung lượng FP32 thật** — minh chứng: export cùng model sang ONNX với `half=False` cho ra file **11.74 MB**, đúng ~2× baseline.pt, khớp lý thuyết FP32 thật.
>
> **Lưu ý về việc chọn mức pruning cho Combined:** FPS của P40%+FP16 (41.38) và P50%+FP16 (36.38) khác nhau ~12% — đây là **nhiễu đo giữa các lần chạy** (CPU throttling, scheduler, cache warm-up...), KHÔNG phải vì pruning nhiều hơn chạy nhanh hơn. Toàn bộ 10 mức pruning (20-99%) đều cho FPS dao động ngẫu nhiên quanh 25-27 FPS khi chưa quantize — pruning không cắt giảm phép tính thực tế (xem [Đường Cong Pruning Sweep](#-biểu-đồ-trực-quan-hóa)), chỉ có bước quantize (FP16/INT8) mới thực sự tăng tốc. Mức pruning ảnh hưởng đến mAP, không ảnh hưởng đến FPS.
>
> ### 🔍 Vì sao chọn FP16 thay vì "nén sâu hơn" bằng INT8/INT16?
> - **INT8 (dynamic quantization)** nén nhỏ nhất (3.21 MB) và giữ mAP tốt (97.15%), nhưng **chậm hơn FP32 tới 12 lần** trên CPU đo kiểm (Intel i5-12400) — vì `onnxruntime.quantization.quantize_dynamic` được tối ưu chủ yếu cho MatMul/Linear (Transformer), hỗ trợ Conv2d (chiếm phần lớn YOLO) kém trên CPU Execution Provider, phải chạy qua kernel fallback chậm.
> - **INT8 (static/QDQ, có calibration)** đạt tốc độ tốt (~20 FPS) nhưng mAP chỉ còn ~82% trong thử nghiệm ban đầu — do node cuối Detect head gộp box-coordinates (range 0-640) và class-probabilities (range 0-1) vào cùng 1 tensor, khiến calibration tính sai scale và "nghiền" giá trị nhỏ về 0 (loại lỗi tương tự đã gặp với pruning: một layer/tensor có cấu trúc/scale đặc biệt bị xử lý đồng nhất như phần còn lại).
> - **INT16** có 65536 mức lượng tử (mịn hơn INT8 rất nhiều) nên giữ mAP tốt (97.64%), nhưng vẫn **chậm hơn FP16** (~17 FPS) và **to hơn FP16** (6.12 MB > 5.90 MB dù cùng 16-bit) — ONNX Runtime CPU không có kernel INT16 được tối ưu tốt (phải tự nâng opset 17→21 mới chạy), khác hẳn FP16 vốn được hỗ trợ native.
> - **FP16** chỉ đổi định dạng lưu trữ (32-bit → 16-bit float), **không rời rạc hóa giá trị nên không cần calibration** → không có rủi ro lệch scale như INT8/INT16. Kết quả: mAP gần như nguyên vẹn (97.77% so với baseline 97.78%) và **nhanh hơn cả FP32** — bất ngờ vì CPU đo kiểm (Intel 12th gen) không có tập lệnh tính toán FP16 chuyên dụng (AVX-512-FP16), nhưng vẫn có tập lệnh **F16C** (chuyển đổi FP16↔FP32, có từ 2012) để ép kiểu gần như miễn phí; với model nhỏ như YOLOv8n (giới hạn bởi băng thông bộ nhớ hơn là tốc độ tính toán ở batch=1), giảm một nửa lượng dữ liệu cần đọc giúp tăng tốc dù không có ALU FP16 chuyên dụng.
> - **Bài học tổng quát:** tốc độ suy luận thực tế của một định dạng lượng tử hóa phụ thuộc vào **mức độ tối ưu của runtime cho định dạng đó trên phần cứng cụ thể**, không chỉ vào lý thuyết "ít bit hơn = nhanh hơn". Luôn cần đo thực nghiệm thay vì suy luận thuần lý thuyết.
>
> ⚠️ **Lưu ý về FPS ONNX INT8:** Trên máy benchmark hiện tại (Windows + Intel i5-12400), ONNX Runtime CPU cho INT8 dynamic-quantized graph chạy chậm hơn nhiều so với FP32 (2.7 FPS vs ~27 FPS baseline) — dưới ngưỡng real-time 15 FPS. Đây là đặc điểm phụ thuộc phần cứng/build ONNX Runtime (không tối ưu tốt kernel INT8 cho graph này trên CPU Intel), không phải lỗi pruning/quantization logic. Cần benchmark lại trên thiết bị đích thực tế (Raspberry Pi, Jetson, hoặc dùng static QDQ quantization) trước khi kết luận về tính khả thi real-time.

---

## 📈 Biểu Đồ Trực Quan Hóa

Bốn biểu đồ khoa học đã được tự động kết xuất tại thư mục [`results/figures/`](results/figures/):

### 1. Dashboard Tổng Hợp (Summary Dashboard)
![Summary Dashboard](results/figures/summary_dashboard.png)

### 2. Đánh Đổi Độ Chính Xác vs Dung Lượng (Trade-off mAP vs Size)
![Accuracy vs Size](results/figures/accuracy_vs_size.png)

### 3. So Sánh Tốc Độ FPS trên CPU (Threshold 15 FPS Real-time)
![FPS Comparison](results/figures/fps_comparison.png)

### 4. Đường Cong Pruning Sweep — Tìm Điểm Giới Hạn (20% → 99%)
![Pruning Sweep](results/figures/pruning_sweep.png)

Model giữ mAP gần như không đổi (~97.5-97.8%) từ 20% đến 80% sparsity nhờ fine-tune, sau đó suy giảm rõ rệt từ **90%** trở đi — đây là điểm giới hạn thực sự của kiến trúc YOLOv8n trên bộ dữ liệu này (10 epoch fine-tune, lr=1e-4). FPS gần như không đổi theo mức pruning vì L1 unstructured pruning chỉ zero hóa trọng số (sparse weights), không giảm số phép tính FLOPs thực tế trên CPU dense — điều này giải thích tại sao dung lượng file (~6MB) cũng không đổi qua mọi mức pruning.

---

## ✅ Đã Đạt Được Mục Tiêu Đề Tài Chưa? (Độ chính xác không đổi + FPS tối đa)

**Mục tiêu:** Tìm tổ hợp pruning + quantization giữ độ chính xác gần như không đổi, đồng thời tăng FPS nhiều nhất có thể.

**Kết quả:** ✅ Đạt được — nhưng với 1 điểm cần hiểu đúng: **toàn bộ mức tăng FPS đến từ quantization (FP16), không đến từ pruning.**

| Phương án | mAP@50 | Δ so với baseline | FPS | Δ FPS |
|---|---:|---:|---:|---:|
| Baseline (FP32, chưa tối ưu gì) | 97.78% | — | 27.06 | — |
| ONNX FP16 (chỉ quantize, KHÔNG pruning) | 97.77% | -0.01% | 37.24 | **+37.6%** |
| Combined P40% + FP16 | 97.72% | -0.06% | 41.38 | **+52.9%** |
| Combined P50% + FP16 | 98.05% | +0.27% (nhiễu đo) | 36.38 | +34.4% |

**Phân tích trung thực:**
- ✅ **Độ chính xác:** Đạt xuất sắc — mọi phương án FP16 (có hoặc không pruning) đều giữ mAP trong khoảng ±0.3% so với baseline, thực tế nằm trong sai số đo giữa các lần chạy chứ không phải suy giảm thật.
- ✅ **FPS:** Đạt — tăng ~37% so với baseline (xem bằng chứng thống kê bên dưới), vượt xa ngưỡng real-time 15 FPS.
- ⚠️ **NHƯNG: pruning không đóng góp thêm FPS nào ngoài mức FP16 đã có sẵn** — xem 2 bằng chứng độc lập ngay bên dưới.

### 🔬 Bằng chứng thực nghiệm: pruning không cải thiện FPS

**Bằng chứng 1 — Cấu trúc (GFLOPs không đổi):** Đo trực tiếp số phép tính (GFLOPs) và số tham số của model ở 7 mức pruning khác nhau bằng `ultralytics` model profiler:

| Model | Tổng tham số | Tham số khác 0 | Sparsity thực tế | **GFLOPs** |
|---|---:|---:|---:|---:|
| Baseline (0%) | 3,020,988 | 3,020,979 | 0.0% | **8.25** |
| Pruned 20% | 3,020,988 | 2,570,851 | 14.9% | **8.25** |
| Pruned 40% | 3,020,988 | 2,120,714 | 29.8% | **8.25** |
| Pruned 60% | 3,020,988 | 1,670,581 | 44.7% | **8.25** |
| Pruned 80% | 3,020,988 | 1,220,448 | 59.6% | **8.25** |
| Pruned 90% | 3,020,988 | 995,382 | 67.1% | **8.25** |
| Pruned 99% | 3,020,988 | 792,821 | 73.8% | **8.25** |

**GFLOPs giống hệt nhau (8.25) ở MỌI mức pruning từ 0% đến 99%.** Đây là bằng chứng toán học trực tiếp: L1 unstructured pruning chỉ đặt giá trị trọng số về 0 chứ không xóa kênh/filter/lớp nào khỏi kiến trúc mạng — số phép nhân-cộng (FLOPs) mà CPU phải thực hiện hoàn toàn không đổi bất kể bao nhiêu % trọng số bằng 0. Về mặt toán học, **không có cơ chế nào để pruning kiểu này làm tăng FPS** trên phần cứng thực thi dense (không có sparse execution engine chuyên dụng).

**Bằng chứng 2 — Thống kê (đo lặp lại độc lập):** Để loại trừ khả năng "40% nhanh hơn 0% chỉ là do đo 1 lần", đã đo FPS của 4 model (không pruning, 40%, 50%, 80% — đều kết hợp FP16) **5 lần độc lập** (mỗi lần khởi động lại process Python riêng biệt, không dùng chung cache):

| Model | FPS 5 lần đo | Mean | Std (độ lệch chuẩn) | CV (%) |
|---|---|---:|---:|---:|
| Không pruning + FP16 | 36.78, 37.23, 36.82, 37.10, 36.74 | 36.93 | 0.19 | 0.5% |
| Pruned 40% + FP16 | 37.22, 37.16, 36.51, 38.08, 37.14 | 37.22 | 0.50 | 1.3% |
| Pruned 50% + FP16 | 37.05, 37.43, 36.55, 37.01, 36.25 | 36.86 | 0.41 | 1.1% |
| Pruned 80% + FP16 | 36.66, 37.14, 37.60, 36.82, 37.23 | 37.09 | 0.33 | 0.9% |

**Kết luận thống kê:** Cả 4 model (0% → 80% pruning) có **mean FPS nằm trong khoảng 36.86–37.22 — chênh lệch tối đa chỉ 0.36 FPS (~1%)**, trong khi độ lệch chuẩn của MỖI model khi tự đo lặp lại đã là 0.19–0.50 FPS. Nói cách khác: **độ nhiễu giữa các lần đo của CÙNG MỘT model đã lớn ngang hoặc hơn chênh lệch giữa các model có mức pruning khác nhau.** Đây là dấu hiệu kinh điển cho thấy **không có khác biệt thật sự** — sai lệch quan sát được hoàn toàn nằm trong nhiễu đo (do OS scheduler, cache CPU, tần số xung nhịp dao động...), không phải do pruning gây ra.

**➜ Kết luận cuối cùng (có bằng chứng kép):** Giả thuyết "pruning giúp tăng FPS" bị **bác bỏ** cho phương pháp L1 unstructured pruning trên CPU dense execution. Toàn bộ mức tăng FPS trong đề tài (27→37 FPS, +37%) đến từ **quantization FP16**, không phải từ pruning. Pruning vẫn có giá trị riêng (chứng minh model dư thừa 80-90% tham số, mở đường cho structured pruning trong tương lai) nhưng **không phải là kỹ thuật tăng tốc trong cấu hình hiện tại**.

> Dữ liệu thô: [`results/repeated_fps_evidence.json`](results/repeated_fps_evidence.json). Script tái tạo: xem `docs/ke_hoach_sua_loi_pruning.md`.

**Kết luận cho việc triển khai thực tế:** Nếu mục tiêu THUẦN TÚY là "chính xác không đổi + FPS tối đa", **chỉ cần ONNX FP16 là đủ — không bắt buộc phải pruning trước**. Việc pruning trước khi quantize (Combined) không làm hại gì (vẫn giữ mAP tốt tới tận 80-90%) nhưng cũng không mang lại lợi ích tốc độ đo được trong cấu hình hiện tại (unstructured pruning + CPU dense execution). Pruning vẫn có giá trị học thuật quan trọng — chứng minh model dư thừa tới 80-90% tham số backbone/neck — và sẽ thực sự chuyển hóa thành FPS cao hơn nếu áp dụng **structured/channel pruning** (xóa hẳn kênh, giảm kích thước tensor thật) hoặc chạy trên runtime hỗ trợ sparse execution — đây là hướng phát triển tiếp theo hợp lý cho đề tài.

---

## 🎓 Bài Học Rút Ra Từ Quá Trình Lượng Tử Hóa

Đề tài thử nghiệm **5 phương án quantization/precision khác nhau** (PyTorch Dynamic, ONNX Dynamic INT8, ONNX Static INT8, ONNX INT16, ONNX FP16) và rút ra những bài học quan trọng hơn cả kết quả cuối cùng:

### 1. "Ít bit hơn" không đồng nghĩa "nhanh hơn"
Trực giác thông thường: INT8 (1 byte) < FP16 (2 byte) < FP32 (4 byte) nên INT8 phải nhanh nhất. **Thực nghiệm ngược lại hoàn toàn** trên CPU đo kiểm (Intel i5-12400):

| Định dạng | Bytes/giá trị | FPS thực đo | So với FP32 |
|---|---:|---:|---|
| FP32 (gốc) | 4 | 27.06 | — |
| **FP16** | **2** | **37.24** | **Nhanh hơn 1.4×** |
| INT16 | 2 | 16.67 | Chậm hơn 1.6× |
| INT8 (dynamic) | 1 | 2.68 | **Chậm hơn 10×** |

**Nguyên nhân gốc:** Tốc độ thực tế phụ thuộc vào việc **runtime (ONNX Runtime) có kernel được lập trình tối ưu cho định dạng đó, trên loại phép toán đó (Conv2d), trên kiến trúc CPU cụ thể đó hay không** — không phải vào lý thuyết số bit. `onnxruntime.quantization.quantize_dynamic` được tối ưu chủ yếu cho MatMul (Transformer/NLP), hỗ trợ Conv2d (chiếm phần lớn YOLO) rất kém trên CPU Execution Provider. INT16 thậm chí phải tự nâng opset (17→21) mới chạy được — dấu hiệu rõ ràng đây là đường ít được dùng/tối ưu.

### 2. FP16 nhanh hơn FP32 dù CPU không có phần cứng tính FP16 chuyên dụng
CPU đo kiểm (Intel 12th gen, dòng tiêu dùng) **không có AVX-512-FP16** (tập lệnh tính toán trực tiếp trên FP16). Nhưng nó **có F16C** (tập lệnh chuyển đổi FP16↔FP32 nhanh, phổ biến từ 2012) — ONNX Runtime dùng F16C để convert gần như miễn phí rồi tính bằng FP32 ALU. Vì YOLOv8n là model nhỏ chạy batch=1 (bị giới hạn bởi **băng thông bộ nhớ** hơn là tốc độ tính toán thuần túy), giảm một nửa dữ liệu trọng số cần đọc từ RAM giúp tăng tốc thật — dù không có đơn vị tính FP16 chuyên dụng.

### 3. Calibration (INT8/INT16) là con dao 2 lưỡi
INT8/INT16 cần ước lượng scale/zero-point từ dữ liệu mẫu. Nếu 1 node trong graph gộp nhiều nhánh có range giá trị khác nhau — đúng tình huống ở Detect head của YOLO (node `Concat` cuối gộp box-coordinates [0-640] với class-probabilities [0-1]) — calibration có thể tính sai hoàn toàn, "nghiền" giá trị nhỏ về 0 (mAP từng sập về 0.00% trong thử nghiệm ban đầu, phải sửa bằng cách loại trừ node đó khỏi quantization). **FP16 không có bước ước lượng này nên miễn nhiễm với lớp lỗi này.**

### 4. Luôn đo thực nghiệm, không suy luận thuần lý thuyết
Cả 2 dự đoán ban đầu trong quá trình làm đề tài đều SAI khi kiểm chứng thực tế: (a) dự đoán FP16 sẽ không nhanh hơn FP32 vì thiếu AVX-512-FP16 — **sai**, và (b) dự đoán INT16 sẽ nhanh hơn FP16 vì SIMD xử lý số nguyên hiệu quả hơn — **cũng sai**. Đây là minh chứng thực nghiệm cho việc trong tối ưu hóa hệ thống nhúng, **luôn phải đo đạc trên phần cứng/runtime cụ thể**, không được tin tưởng tuyệt đối vào lý thuyết phần cứng chung chung.

### 5. Dự đoán cho Raspberry Pi 4 — dựa trên phân tích kiến trúc (chưa kiểm chứng thực tế)

Pi 4 dùng chip Broadcom BCM2711 — 4 nhân **ARM Cortex-A72** (ARMv8.0-A, 2015), khác hẳn CPU Intel đo kiểm hiện tại. So sánh đặc điểm liên quan đến quantization:

| Đặc điểm | Intel i5-12400 (đã đo) | Cortex-A72 / Pi 4 |
|---|---|---|
| Tính toán FP16 trực tiếp | Không (thiếu AVX-512-FP16) | Không (ARMv8.0, thiếu FEAT_FP16 — cần ARMv8.2+) |
| Convert FP16↔FP32 nhanh | Có (F16C, từ 2012) | Có (FCVT/FCVTL — có sẵn từ NEON cơ bản ARMv8.0) |
| Dot-product INT8 chuyên dụng | Có (AVX-VNNI, dù ORT có thể chưa tận dụng hết) | **Không** (SDOT/UDOT chỉ có từ Cortex-A75/ARMv8.2 trở lên) |
| Băng thông RAM | DDR4/5, ~25-40+ GB/s (dual channel desktop) | LPDDR4, ~4-6 GB/s hiệu dụng — **hẹp hơn nhiều** |
| Ưu tiên tối ưu của ONNX Runtime | x86 desktop — ít ưu tiên INT8 Conv | **ARM/mobile — ưu tiên CAO cho INT8** (mục tiêu triển khai chính của edge ML: TFLite, NCNN, ORT Mobile) |

**Dự đoán:**
- **FP16 vẫn sẽ nhanh hơn FP32** trên Pi 4, khả năng còn rõ hơn trên Intel — vì Pi 4 bị giới hạn băng thông RAM (LPDDR4 hẹp) NẶNG hơn desktop nhiều, nên lợi ích "giảm một nửa dữ liệu cần đọc" của FP16 sẽ càng có tác động lớn. Đây vẫn là lựa chọn AN TOÀN, rủi ro thấp nhất.
- **INT8 (đặc biệt static/QDQ, đã sửa lỗi loại trừ Concat/Sigmoid) có khả năng cao sẽ đảo ngược kết quả trên Intel** — tức là THỰC SỰ nhanh trên Pi 4, khác hẳn thất bại trên desktop. Lý do: (a) ONNX Runtime dành ưu tiên tối ưu kernel INT8 cho ARM/mobile nhiều hơn hẳn x86 desktop (đây mới là thị trường mục tiêu thật của edge AI quantization), và (b) Pi 4 bị giới hạn băng thông RAM nặng nên INT8 (giảm 4× dữ liệu, so với FP16 chỉ giảm 2×) mang lại lợi ích lớn hơn tương ứng.
- **INT16 nhiều khả năng vẫn sẽ kém** — vì đây là định dạng ít phổ biến, không có lý do để các bản build ONNX Runtime cho ARM ưu tiên tối ưu nó hơn x86.
- **Khuyến nghị hành động:** Nếu có Raspberry Pi 4 thật, nên test lại CẢ static INT8 (dùng model `checkpoints/experimental_static_quant/quant_onnx_int8_static_v2.onnx` đã sửa lỗi loại trừ node, dù mAP mới đạt ~82% cần cải thiện thêm calibration) VÀ FP16 (`checkpoints/quant_onnx_fp16.onnx`) — rất có thể trên Pi 4 thứ tự xếp hạng sẽ khác hẳn trên desktop, và đây sẽ là một phát hiện thực nghiệm giá trị cho báo cáo (đúng tinh thần bài học #4: đo đạc thay vì suy luận).

---

## 📖 HƯỚNG DẪN TỪ A ĐẾN Z DÀNH CHO THÀNH VIÊN TRONG NHÓM
*(Dành cho bất kỳ ai mới bắt đầu, chỉ cần làm theo từng bước là chạy được 100%)*

### 🔹 Bước 1: Tải mã nguồn về máy tính

Mở **Terminal** (trên macOS/Linux) hoặc **Git Bash / Command Prompt** (trên Windows), gõ lệnh:
```bash
git clone https://github.com/TranTraiJr/traffic_sign_quantization.git
cd traffic_sign_quantization
```

---

### 🔹 Bước 2: Cài đặt công cụ quản lý thư viện `uv`

Dự án sử dụng công cụ quản lý gói siêu nhanh **`uv`** (không cần cài thủ công từng thư viện):

- **Trên macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Trên Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Sau khi cài xong, tải toàn bộ thư viện dự án tự động chỉ bằng 1 lệnh:**
  ```bash
  uv sync
  ```
  *(Lệnh này tự động tạo môi trường ảo `.venv` và cài đầy đủ PyTorch, Ultralytics YOLO, OpenCV, ONNX Runtime, Pillow...)*

> **Nếu không muốn dùng `uv` mà quen dùng `pip` truyền thống:**
> ```bash
> python -m venv .venv
> source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
> pip install torch torchvision ultralytics onnx onnxruntime opencv-python pillow matplotlib pandas
> ```

---

### 🔹 Bước 3: Tải Dataset Biển Báo Giao Thông (Google Drive)

Do bộ ảnh nặng hơn 760 MB nên không đẩy lên GitHub. Bạn hãy tải về theo đường link Drive dưới đây:

👉 **[BẤM VÀO ĐÂY ĐỂ TẢI FOLDER ARCHIVE TRÊN GOOGLE DRIVE](https://drive.google.com/file/d/1kHesyhAxhxm4CEi9Y9Q11SNuOwuzQDH1/view?usp=sharing)**

**Cách đặt file đúng vị trí sau khi tải về:**
1. Tải file `archive` từ link Drive về máy tính.
2. Giải nén (nếu là file zip) và đặt thư mục **`archive`** nằm ngay tại thư mục gốc của dự án `traffic_sign_quantization/`.
3. Kiểm tra cấu trúc thư mục đảm bảo như sau là chuẩn:
   ```
   traffic_sign_quantization/
   ├── archive/
   │   ├── images/          <-- Chứa các file ảnh .jpg
   │   ├── labels/          <-- Chứa các file nhãn .txt
   │   ├── classes.txt
   │   ├── classes_vie.txt
   │   └── split_dataset/
   ```

---

### 🔹 Bước 4: Chạy Demo xem kết quả ngay lập tức

Bạn **KHÔNG cần phải huấn luyện lại**, toàn bộ mô hình đã được nén sẵn trong thư mục `checkpoints/`. Bạn chỉ cần chọn một trong các cách chạy sau:

#### Cách 4.1: Chạy Demo trên Video mẫu có sẵn (Khuyên dùng khi thuyết trình)
```bash
uv run python demo_video.py --source data/sample_demo.mp4 --model checkpoints/quant_onnx_fp16.onnx
```
*(Cửa sổ sẽ hiện lên, chạy qua 45 loại biển báo khác nhau, có khung viền màu, dịch tên tiếng Việt và đo FPS trực tiếp).*

#### Cách 4.2: Chạy Demo trực tiếp bằng Webcam máy tính
```bash
uv run python demo_video.py --source 0 --model checkpoints/quant_onnx_fp16.onnx
```
*(Bạn có thể mở hình biển báo giao thông trên điện thoại rồi giơ trước camera để xem mô hình nhận dạng tức thì).*

#### Cách 4.3: So sánh với mô hình gốc Baseline (FP32)
```bash
uv run python demo_video.py --source data/sample_demo.mp4 --model checkpoints/baseline.pt
```

> 💡 **Phím tắt điều khiển:** Khi cửa sổ demo đang mở, nhấn phím **`Q`** hoặc **`ESC`** trên bàn phím để tắt.

---

### 🔹 Bước 5: Xem báo cáo tiến độ và bảng số liệu

Tất cả tài liệu làm đồ án đã được viết đầy đủ trong thư mục [`docs/`](docs/):
- Đề cương chi tiết: [`docs/de_cuong_do_an.md`](docs/de_cuong_do_an.md)
- Báo cáo Tuần 1: [`docs/bao_cao_tuan_1.md`](docs/bao_cao_tuan_1.md) (Khảo sát & Nghiên cứu)
- Báo cáo Tuần 2: [`docs/bao_cao_tuan_2.md`](docs/bao_cao_tuan_2.md) (Dữ liệu & Huấn luyện Baseline)
- Báo cáo Tuần 3: [`docs/bao_cao_tuan_3.md`](docs/bao_cao_tuan_3.md) (Cắt tỉa Pruning 4 mức)
- Báo cáo Tuần 4: [`docs/bao_cao_tuan_4.md`](docs/bao_cao_tuan_4.md) (Lượng tử hóa INT8 & Benchmark)
- Báo cáo Tuần 5: [`docs/bao_cao_tuan_5.md`](docs/bao_cao_tuan_5.md) (Kiểm thử thực tế & Tổng kết)

---

## 📁 Cấu Trúc Dự Án Chi Tiết

```
traffic-sign-quantization/
├── checkpoints/                        # Các trọng số mô hình đã huấn luyện sẵn
│   ├── baseline.pt                     # Model gốc FP32 (97.78% mAP50)
│   ├── quant_onnx_fp16.onnx            # ⭐ Model ONNX FP16 tối ưu nhất (5.90 MB, 37.2 FPS)
│   ├── quant_onnx_int8.onnx            # Model ONNX INT8 nhỏ nhất (3.21 MB, không real-time)
│   ├── pruned_XXpct_finetuned.pt       # Các model sau khi cắt tỉa và fine-tune
│   ├── pruned_40pct_finetuned_onnx_fp16.onnx
│   └── pruned_40pct_finetuned_onnx_int8.onnx
├── docs/                               # Toàn bộ hồ sơ báo cáo đồ án
│   ├── de_cuong_do_an.md               # Đề cương chi tiết
│   ├── ke_hoach_do_an.md               # Kế hoạch tiến độ 6 tuần
│   ├── bao_cao_tuan_1.md đến tuan_5.md # 5 báo cáo tiến độ theo tuần
├── results/
│   ├── comparison_metrics.csv          # Bảng số liệu đối sánh toàn diện
│   ├── demo_quant_onnx_int8.mp4        # Video kết quả chạy mô hình tối ưu
│   ├── demo_baseline.mp4               # Video kết quả chạy mô hình gốc
│   └── figures/                        # 4 biểu đồ khoa học chất lượng cao
├── src/                                # Mã nguồn module hóa
│   ├── config.py                       # Quản lý cấu hình, siêu tham số, 52 classes
│   ├── dataset.py                      # Chuẩn bị dữ liệu và sinh traffic_signs.yaml
│   ├── train.py                        # Huấn luyện baseline và fine-tuning
│   ├── prune.py                        # Cắt tỉa L1 Unstructured Pruning
│   ├── quantize.py                     # Dynamic INT8 & ONNX Runtime INT8
│   ├── evaluate.py                     # Đo đạc mAP50, Latency, FPS trên CPU
│   ├── pipeline.py                     # Điều phối tự động hóa các pha thực nghiệm
│   └── visualize.py                    # Kết xuất 4 biểu đồ khoa học
├── archive/
│   ├── classes.txt                     # 52 mã hiệu biển báo
│   ├── classes_vie.txt                 # 52 tên tiếng Việt chuẩn
│   └── split_dataset/                  # Danh sách phân chia train / val
├── data/
│   ├── traffic_signs.yaml              # Cấu hình Ultralytics YOLO
│   └── sample_demo.mp4                 # Video mẫu đa dạng 45 nhóm biển báo
├── demo_video.py                       # Ứng dụng demo thời gian thực (GUI + HUD)
├── main.py                             # Giao diện CLI điều khiển pipeline
├── pyproject.toml                      # Cấu hình môi trường uv / pip
└── README.md
```

---

## 👥 Nhóm Tác Giả & Phân Công Nhiệm Vụ

| Họ và Tên | Vai trò | Trách nhiệm chính |
|---|---|---|
| **Trần Thành Trai** | Trưởng nhóm / AI Lead | Thiết kế kiến trúc, xây dựng pipeline, huấn luyện Baseline |
| **Nguyễn Thanh Hoàng** | Data Engineer | Chuẩn bị dữ liệu 52 lớp biển báo VN, kiểm thử dataset |
| **Bùi Huy Phong** | Optimization Engineer | Nghiên cứu & triển khai Pruning, Quantization, Benchmark |
| **Lý Quốc Vinh** | Demo & Integration | Xây dựng ứng dụng Demo thời gian thực, giao diện HUD tiếng Việt |
| **Thiều Hồng Quân** | Testing & Documentation | Thiết kế kịch bản kiểm thử, tổng hợp số liệu, hoàn thiện báo cáo |

---

## 📜 Giấy Phép (License)

Dự án được phân phối dưới giấy phép mã nguồn mở **MIT License**. Xem file [`LICENSE`](LICENSE) để biết thêm chi tiết.

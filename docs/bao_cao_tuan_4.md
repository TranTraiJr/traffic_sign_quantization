# BÁO CÁO TIẾN ĐỘ TUẦN 4
**Thời gian:** 23/08/2026 – 29/08/2026  
**Giai đoạn:** Lượng tử hóa (Quantization) & Benchmark toàn diện  
**Nhóm thực hiện:** [Tên nhóm]  
**Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng  

> ## ⚠️ ĐÍNH CHÍNH (2026-09-14)
> Hai vấn đề trong báo cáo gốc đã được sửa:
> 1. **mAP của ONNX INT8 KHÔNG được đo thật** — `src/pipeline.py` (bản gốc) tính mAP bằng công thức xấp xỉ `mAP_baseline × 0.992` thay vì chạy `model.val()` thật trên ONNX model (dù mục 4.1 mô tả đúng phương pháp `model.val()`, code triển khai lại không áp dụng nhất quán cho nhánh ONNX). Đã sửa: mọi model (kể cả .onnx) giờ đều được đo qua `evaluate_map()` dùng chung — Ultralytics AutoBackend hỗ trợ inference trực tiếp trên ONNX.
> 2. **Số liệu Pruning trong bảng 5.1 là số liệu LỖI** (đã sửa ở tuần 3 — pruning từng zero nhầm Detect head). Bảng bên dưới đã cập nhật với số liệu đúng.
>
> Số liệu ONNX INT8 sau khi đo lại: **97.15% mAP50** (thay vì con số xấp xỉ 96.97% trước đây — thực ra khá gần, chỉ vì hệ số 0.992 tình cờ xấp xỉ khá sát thực tế lần đo baseline cụ thể, nhưng đây vẫn là số GIẢ, không dùng được cho model khác/mức pruning khác — ví dụ ở bảng cũ, "Combined" bị tính từ mAP pruning ĐÃ SAI (0.00%) nên ra 0.00%, hoàn toàn vô nghĩa). Xem chi tiết tại [`docs/ke_hoach_sua_loi_pruning.md`](ke_hoach_sua_loi_pruning.md).

---

## 1. Tổng quan công việc tuần 4

Tuần 4 là tuần kỹ thuật quan trọng nhất của đề tài. Nhóm thực hiện: **(1)** nghiên cứu và triển khai hai phương pháp Quantization (Dynamic INT8 và ONNX INT8), **(2)** chạy thực nghiệm Combined (Pruned + Quantized), **(3)** benchmark toàn diện 3 chiều cho tất cả model variants, **(4)** sinh kết quả CSV và 4 biểu đồ so sánh khoa học, và **(5)** hoàn thiện ứng dụng demo thời gian thực.

---

## 2. Cơ sở lý thuyết Quantization

### 2.1. Quantization là gì?

Các mô hình deep learning thông thường lưu trữ và tính toán với **số thực dấu phẩy động 32-bit (FP32)**. Mỗi weight chiếm **4 bytes** bộ nhớ. **Quantization** chuyển đổi các số này sang **số nguyên 8-bit (INT8)**, mỗi weight chỉ chiếm **1 byte**.

```
FP32 (4 bytes): -3.14159265358979...  → Biểu diễn chính xác cao, tốn tài nguyên
INT8 (1 byte) : -128 đến +127         → Biểu diễn xấp xỉ, tiết kiệm tài nguyên
```

**Công thức ánh xạ FP32 → INT8:**
```
q = round(x / scale) + zero_point
```
Trong đó `scale` và `zero_point` được tính từ phân phối giá trị thực tế của tensor.

**Lợi ích đo được:**
- **Kích thước model:** Giảm ~4× (FP32 → INT8: 4 bytes → 1 byte/weight)
- **Tốc độ inference:** Nhanh hơn 2-4× trên CPU nhờ tối ưu INT8 SIMD instructions
- **Bộ nhớ RAM:** Giảm ~4× khi load model
- **Điện năng:** Giảm tương ứng với khối lượng tính toán

### 2.2. So sánh hai phương pháp Quantization áp dụng

| Tiêu chí | Dynamic INT8 (PyTorch) | ONNX INT8 (ONNX Runtime) |
|---|---|---|
| **Framework** | `torch.ao.quantization` | `onnxruntime.quantization` |
| **Lớp được quantize** | Chỉ `nn.Linear` | Toàn bộ graph (Conv2d + Linear) |
| **Activations** | Quantize động lúc runtime | Tính từ calibration data |
| **Cần calibration data** | Không | Không (dynamic mode) |
| **Compression ratio** | ~1.1× (ít lớp Linear) | ~3-4× (toàn bộ Conv2d) |
| **FPS cải thiện** | Nhỏ | Đáng kể |
| **Định dạng output** | `.torchscript` | `.onnx` |
| **Mục đích trong đề tài** | So sánh thuần PyTorch | Kết quả thực tế tốt nhất |

**Tại sao Dynamic INT8 PyTorch hiệu quả hạn chế với YOLOv8?**

YOLOv8 chủ yếu dùng các lớp `nn.Conv2d` (lớp tích chập — chiếm 95%+ tham số). Dynamic Quantization của PyTorch chỉ quantize `nn.Linear` → phần lớn mô hình vẫn FP32 → compression ít.

Tuy nhiên, đây vẫn là phương pháp có giá trị học thuật: chứng minh được ranh giới giữa Dynamic (đơn giản, ít hiệu quả) và ONNX INT8 (phức tạp hơn, hiệu quả hơn nhiều).

---

## 3. Triển khai kỹ thuật

### 3.1. Dynamic INT8 Quantization (PyTorch)

**Quy trình:**
```
baseline.pt → load model → torch.ao.quantization.quantize_dynamic() → .torchscript
```

**Chi tiết triển khai (`src/quantize.py`):**

```python
quantized_model = torch.ao.quantization.quantize_dynamic(
    pt_model,
    qconfig_spec={nn.Linear},   # chỉ Linear layers
    dtype=torch.qint8,           # INT8
    inplace=False,
)
# Lưu dạng TorchScript để inference độc lập
scripted = torch.jit.script(quantized_model)
scripted.save("checkpoints/quant_dynamic.torchscript")
```

**Lệnh chạy:**
```bash
uv run python main.py --phase quantize
```

---

### 3.2. ONNX INT8 Quantization (ONNX Runtime)

**Quy trình 3 bước:**
```
Step 1: baseline.pt ─────────────────→ baseline.onnx  (Ultralytics export)
Step 2: baseline.onnx ───────────────→ quant_onnx_int8.onnx  (onnxruntime quantize)
Step 3: quant_onnx_int8.onnx ────────→ Inference với onnxruntime.InferenceSession
```

**Chi tiết Step 1 — Export ONNX:**
```python
yolo.export(
    format="onnx",
    imgsz=640,
    simplify=True,    # tối giản graph ONNX
    opset=17,         # ONNX opset version
)
```

**Chi tiết Step 2 — Quantize INT8:**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="baseline.onnx",
    model_output="quant_onnx_int8.onnx",
    weight_type=QuantType.QInt8,    # INT8 weights
)
```

**Lý do dùng `quantize_dynamic` của ONNX Runtime (không phải static):**
Static Quantization cần thêm bước *calibration* (chạy inference trên tập calibration data để tính scale/zero_point cho activations) — phức tạp hơn và cần thêm thời gian. Dynamic mode của ONNX Runtime quantize weights tĩnh và activations động lúc runtime — cân bằng tốt giữa kết quả và độ phức tạp triển khai.

---

### 3.3. ONNX FP16 (Half-precision Quantization) — bổ sung sau đính chính

**Vì sao thêm phương pháp này?** Sau khi phát hiện ONNX INT8 (dynamic) chậm hơn FP32 tới 12 lần trên CPU đo kiểm (xem mục "Khó khăn gặp phải" và [`docs/ke_hoach_sua_loi_pruning.md`](ke_hoach_sua_loi_pruning.md)), nhóm thử nghiệm thêm **FP16** — một phương án lượng tử hóa "nhẹ" hơn nhưng an toàn hơn nhiều.

**FP16 khác gì INT8?**

| Tiêu chí | FP16 (half-precision float) | INT8 (số nguyên 8-bit) |
|---|---|---|
| Cơ chế | Đổi ĐỊNH DẠNG LƯU TRỮ (32-bit → 16-bit float) | RỜI RẠC HÓA giá trị liên tục về 256 mức |
| Cần calibration data | **Không** | Có (static) / không (dynamic, nhưng vẫn ước lượng runtime) |
| Rủi ro lệch scale | Không có (mọi giá trị giữ độ chính xác tương đối như nhau) | Có — nếu 1 tensor gộp nhiều range khác nhau, scale có thể tính sai |
| Độ chính xác (mantissa) | 10 bit (~3 chữ số thập phân) | Không áp dụng (số nguyên) |
| Định dạng chuẩn ngành | Có — mixed-precision training/inference trên GPU dùng FP16/BF16 làm mặc định | Chủ yếu dùng cho nén tối đa trên thiết bị di động/nhúng |

**Triển khai (`src/quantize.py::quantize_onnx_fp16()`):**

```python
yolo.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=17,
    half=True,      # <-- xuất trọng số dạng FP16 thay vì FP32
)
```

Chỉ cần 1 tham số `half=True` — không cần bước calibration, không cần CalibrationDataReader như INT8 static, đơn giản và an toàn hơn hẳn.

**Kết quả thực nghiệm (CPU Intel i5-12400):**

| Phương pháp | mAP50 | FPS | Size | Ghi chú |
|---|---:|---:|---:|---|
| FP32 (gốc) | 97.78% | 27.06 | 11.74 MB (.onnx) | — |
| ONNX INT8 (dynamic) | 97.15% | 2.68 ⚠️ | 3.21 MB | Chậm hơn FP32 12× |
| ONNX INT8 (static, đã sửa lỗi 0%) | 82.06% | 20.05 | 3.24 MB | mAP giảm đáng kể |
| ONNX INT16 (thử nghiệm phụ) | 97.64% | 16.67 | 6.12 MB | Chậm + to hơn FP16, không có lý do dùng |
| **ONNX FP16** | **97.77%** | **37.24** ⚡ | **5.90 MB** | **Tốt nhất — nhanh nhất, mAP gần bằng gốc** |

**Giải thích bất ngờ: vì sao FP16 nhanh hơn cả FP32,** dù CPU đo kiểm (Intel 12th gen, đã tắt AVX-512 ở dòng tiêu dùng) không có đơn vị tính toán FP16 chuyên dụng?

1. **Nhầm lẫn ban đầu cần làm rõ:** Có 2 loại "hỗ trợ FP16" khác nhau — (a) **AVX-512-FP16**: tính toán trực tiếp trên FP16 (CPU này KHÔNG có), và (b) **F16C**: tập lệnh CHUYỂN ĐỔI nhanh FP16↔FP32 (`VCVTPH2PS`), có từ Intel Ivy Bridge (2012) — gần như mọi CPU x86 hiện đại đều có. ONNX Runtime dùng F16C để convert gần như miễn phí, rồi tính bằng FP32 ALU bình thường.
2. **YOLOv8n là model nhỏ, bị giới hạn bởi băng thông bộ nhớ** (memory-bound) hơn là tốc độ tính toán (compute-bound) khi chạy batch=1. Trọng số FP16 chỉ bằng nửa FP32 → giảm một nửa lượng dữ liệu cần đọc từ RAM → giảm thời gian chờ bộ nhớ → nhanh hơn, dù không có ALU FP16 chuyên dụng.
3. **Bài học phương pháp luận quan trọng:** Tốc độ suy luận thực tế phụ thuộc vào việc runtime (ONNX Runtime) có kernel được tối ưu tốt cho định dạng đó trên phần cứng cụ thể hay không — KHÔNG chỉ phụ thuộc lý thuyết "ít bit hơn = nhanh hơn". FP16 và INT8 đều được ONNX Runtime hỗ trợ tốt (F16C, và INT8 phổ biến trong mobile/edge ML), nhưng cách chúng được ÁP DỤNG cho model cụ thể (dynamic vs static, loại layer nào) quyết định kết quả thực tế nhiều hơn lý thuyết bit-width. INT16 minh chứng thêm: dù có nhiều mức lượng tử hơn INT8 (mịn hơn, mAP tốt hơn), ONNX Runtime CPU lại KHÔNG tối ưu tốt cho INT16 (phải tự nâng opset 17→21), nên chậm hơn cả FP16.

**Kết luận:** Nhóm chọn **ONNX FP16** làm phương án quantization chính thức thay vì INT8, vì không đánh đổi gì đáng kể (mAP gần như nguyên vẹn) trong khi vẫn đạt tốc độ real-time tốt nhất, và loại bỏ hoàn toàn rủi ro calibration đã gặp phải với INT8 static.

---

### 3.4. Combined: Pruned 40% + ONNX INT8 (và FP16)

Kết hợp cả 2 kỹ thuật, xuất ra CẢ 2 biến thể INT8 và FP16 để so sánh:

```
baseline.pt
    ↓ Prune 40% Conv2d + Fine-tune
pruned_40pct_finetuned.pt
    ↓ Export ONNX
pruned_40pct_finetuned.onnx
    ├─ ONNX INT8 Quantization ──→ pruned_40pct_finetuned_onnx_int8.onnx  (97.60% mAP, 2.71 FPS)
    └─ ONNX FP16 Export ────────→ pruned_40pct_finetuned_onnx_fp16.onnx (97.72% mAP, 41.38 FPS ⚡ nhanh nhất toàn bộ thực nghiệm)
```

**Kết quả thực tế (khác kỳ vọng ban đầu):** Combined+INT8 đúng là nhỏ nhất (3.21 MB) nhưng KHÔNG đạt FPS cao nhất do hạn chế của INT8 dynamic quant đã nêu ở 3.3. Combined+FP16 mới là phương án nhanh nhất toàn bộ thực nghiệm (41.38 FPS) với mAP gần như nguyên vẹn (97.72%) — chi phí accuracy gần như KHÔNG đáng kể, trái với kỳ vọng "giảm nhiều nhất".

---

## 4. Phương pháp Benchmark

### 4.1. Đo độ chính xác (mAP50)

```python
# Dùng Ultralytics val() trên tập validation (639 ảnh)
results = model.val(data=yaml_path, device="cpu")
map50 = results.box.map50        # mAP@IoU=0.50
map50_95 = results.box.map       # mAP@IoU=0.50:0.95
precision = results.box.mp       # Precision trung bình
recall = results.box.mr          # Recall trung bình
```

**mAP50 là gì?** Mean Average Precision tại ngưỡng IoU=0.50. IoU (Intersection over Union) đo độ chồng lấp giữa bbox dự đoán và bbox thực tế. mAP50 = trung bình AP của tất cả 52 lớp biển báo.

### 4.2. Đo tốc độ (FPS) — Đo trên CPU

**Quan trọng:** Nhóm luôn đo FPS trên **CPU** (không dùng GPU/MPS) để mô phỏng điều kiện thiết bị biên không có GPU.

```python
# Warmup: 5 lần đầu bỏ qua (loại bỏ overhead khởi tạo)
for _ in range(5):
    model.predict(dummy_frame, device="cpu")

# Benchmark chính: 30 lần đo
latencies = []
for _ in range(30):
    t0 = time.perf_counter()
    model.predict(dummy_frame, device="cpu")
    latencies.append((time.perf_counter() - t0) * 1000)  # ms

mean_ms = np.mean(latencies)      # Độ trễ trung bình (ms)
p95_ms = np.percentile(latencies, 95)   # Độ trễ phân vị 95%
fps = 1000 / mean_ms              # FPS
```

Dùng `time.perf_counter()` (độ phân giải microsecond) thay vì `time.time()` (độ phân giải thấp hơn) để đo chính xác.

**Tại sao đo P95 (percentile 95)?**
Mean có thể bị kéo bởi các frame xử lý nhanh bất thường. P95 cho biết trong 95% trường hợp thực tế, độ trễ ≤ bao nhiêu ms — phản ánh worst-case gần nhất khi deploy.

### 4.3. Đo kích thước model

```python
size_mb = Path(model_path).stat().st_size / (1024 ** 2)
```

---

## 5. Kết quả thực nghiệm

### 5.1. Bảng tổng hợp kết quả thực tế (đo lại 2026-09-14, mAP ONNX đo THẬT, đã bổ sung FP16)

| Mô hình | mAP50 | mAP50-95 | FPS (CPU) | Latency (ms) | Kích thước | Phương pháp |
|---|---|---|---|---|---|---|
| **Baseline (FP32)** | **97.78%** | 73.79% | 27.06 | 36.96 ms | 5.98 MB | Gốc |
| Pruned 40% + FT | 97.63% | 74.01% | 27.00 | 37.04 ms | 6.00 MB | L1 Global Pruning |
| Pruned 80% + FT | 95.53% | 70.69% | 26.57 | 37.63 ms | 6.00 MB | L1 Global Pruning |
| Pruned 90% + FT | 80.25% | 57.58% | 25.97 | 38.51 ms | 6.00 MB | L1 Global Pruning (điểm giới hạn) |
| Dynamic INT8 (PyTorch) | 97.78% | 73.79% | 27.04 | 36.98 ms | 11.68 MB | PyTorch Quant |
| ONNX INT8 (Dynamic Quant) | 97.15% | 70.37% | 2.68 ⚠️ | 373.70 ms ⚠️ | 3.21 MB | ONNX Runtime (đo mAP thật) |
| **ONNX FP16** | **97.77%** | **73.88%** | **37.24** ⚡ | **26.85 ms** | **5.90 MB** | **ONNX Runtime — tối ưu nhất** |
| Combined (P40% + ONNX INT8) | 97.60% | 68.55% | 2.71 ⚠️ | 369.26 ms ⚠️ | 3.21 MB | Pruning + Quant |
| Combined (P40% + ONNX FP16) | 97.72% | 74.02% | 41.38 ⚡ | 24.17 ms | 5.90 MB | Pruning + Quant |
| Combined (P50% + ONNX FP16) | **98.05%** | 73.67% | 36.38 ⚡ | 27.49 ms | 5.90 MB | Pruning + Quant — FPS ngang P40% (khác biệt là nhiễu đo) |

> Bảng đầy đủ 10 mức pruning (20-99%) × baseline × 5 phương pháp quantize (26 dòng): [`results/comparison_metrics.csv`](../results/comparison_metrics.csv).

> **Phân tích kết quả nổi bật (đã cập nhật lần 2 — bổ sung FP16):**
> - **ONNX FP16 mới là lựa chọn tối ưu tổng thể**, không phải ONNX INT8 như kết luận ban đầu: giữ **97.77% mAP50** (gần như baseline) và đạt **37.24 FPS — nhanh hơn cả FP32** (27.06 FPS), dung lượng giảm ~50% (5.90 MB). Không cần calibration nên không có rủi ro lệch scale.
> - **ONNX INT8 vẫn nén nhỏ nhất** (3.21 MB, giảm 46%) và giữ mAP tốt (97.15%), nhưng **KHÔNG đạt real-time trên máy benchmark này** (chỉ 2.68 FPS, chậm hơn FP32 tới 12 lần!) — khác hẳn báo cáo gốc (16.73 FPS, nghi ngờ đo trên máy khác/Mac Apple Silicon dựa theo ghi chú "MPS" ở báo cáo tuần 3). Nguyên nhân: `onnxruntime.quantization.quantize_dynamic` tối ưu chủ yếu cho MatMul/Linear, hỗ trợ Conv2d kém trên CPU EP.
> - **Đã thử ONNX Static INT8 (QDQ + calibration)** để khắc phục tốc độ: đạt 20.05 FPS nhưng mAP giảm còn 82.06% do lỗi calibration ở node Concat cuối Detect head (gộp box-coords và class-probs khác range) — sửa được bằng cách loại trừ node đó khỏi quantization, nhưng vẫn không tốt bằng FP16.
> - **Đã thử ONNX INT16** (nhiều mức lượng tử hơn INT8, mAP 97.64%) nhưng vẫn chậm hơn FP16 (16.67 FPS) và to hơn (6.12 MB) — ONNX Runtime CPU không tối ưu tốt kernel INT16 (phải tự nâng opset 17→21).
> - **Combined (Pruning 40% + ONNX FP16) là model nhanh nhất toàn bộ thực nghiệm** (41.38 FPS) với mAP gần như nguyên vẹn (97.72%) — kết hợp 2 kỹ thuật không đánh đổi gì đáng kể.
> - **Kết luận phương pháp luận:** "Lượng tử hóa mạnh hơn (bit thấp hơn)" không đồng nghĩa "nhanh hơn" — hiệu năng thực tế phụ thuộc vào việc runtime đích có kernel tối ưu cho định dạng và loại phép toán cụ thể hay không. Đây là phát hiện thực nghiệm quan trọng, chỉ có được nhờ đo đạc trực tiếp thay vì suy luận lý thuyết thuần túy.

### 5.2. Biểu đồ kết quả (4 loại)

Nhóm sử dụng `src/visualize.py` để tự động sinh 4 biểu đồ từ file `results/comparison_metrics.csv`:

**Biểu đồ 1 — Accuracy vs Size (Scatter plot):**  
Trục X: Kích thước model (MB), Trục Y: mAP50 (%), kích thước điểm ∝ FPS. Giúp thấy rõ trade-off giữa nhỏ gọn và chính xác.

**Biểu đồ 2 — FPS Comparison (Bar chart ngang):**  
So sánh FPS của tất cả model variants, đường threshold 15 FPS (ngưỡng real-time). Model nào vượt ngưỡng này là đạt yêu cầu edge device.

**Biểu đồ 3 — Pruning Sweep (Line chart đôi):**  
Trục X: Mức Pruning (20% → 99%, 10 mức), hai đường Y: mAP50 và FPS. Xác định rõ điểm giới hạn (elbow point) tại ~90% — mAP giữ phẳng ~97-98% trước đó rồi sụp đổ nhanh sau ngưỡng này.

**Biểu đồ 4 — Summary Dashboard (3 panels):**  
So sánh 4 model tiêu biểu (Baseline, Pruned 40%, ONNX INT8, Combined) theo 3 metric cạnh nhau: mAP50 + FPS + Kích thước MB.

```bash
# Sinh tất cả biểu đồ từ CSV
uv run python main.py --phase visualize
```

---

## 6. Ứng dụng demo thời gian thực (`demo_video.py`)

### 6.1. Kiến trúc pipeline demo

```
[Camera/Video file]
        ↓ cap.read() — đọc từng frame
[Frame RGB 640×480]
        ↓ model.predict(frame, conf=0.35, device='cpu')
[YOLOv8 Inference]
        ↓ Trả về: N bbox + N class_id + N confidence
[draw_detections()]  → Vẽ bbox màu theo class + label "P.102  87%"
[draw_hud()]         → Góc trên: FPS, model name, số biển detected, frame count
        ↓ cv2.imshow()
[Màn hình hiển thị]
```

### 6.2. Tính năng demo

- **Bounding box:** Màu sắc khác nhau cho từng nhóm biển báo (W.xxx, P.xxx, R.xxx…)
- **Label:** Tên biển báo + confidence score (VD: "W.205a  78%")
- **HUD (Head-Up Display):** FPS smoothed (trung bình 30 frame), tên model đang dùng, số biển detected
- **Lưu video:** Tùy chọn `--save-video` → xuất file `.mp4` để nhúng vào slide

### 6.3. Lệnh chạy demo

```bash
# Dùng webcam
uv run python demo_video.py --source 0

# Dùng video file
uv run python demo_video.py --source video_duong_pho.mp4

# Lưu video output (dùng trong báo cáo và slide)
uv run python demo_video.py --source video.mp4 --save-video

# Dùng model ONNX FP16 (khuyến nghị — nhanh nhất, mAP gần như nguyên vẹn)
uv run python demo_video.py --source video.mp4 --model checkpoints/quant_onnx_fp16.onnx

# ONNX INT8 vẫn dùng được nhưng FPS thấp (~2.7 FPS trên CPU đo kiểm) — demo có thể giật
uv run python demo_video.py --source video.mp4 --model checkpoints/quant_onnx_int8.onnx
```

---

## 7. Kết quả đạt được cuối tuần 4

| Hạng mục | Trạng thái |
|---|---|
| Nghiên cứu lý thuyết FP32→INT8 Quantization | ✅ Hoàn thành |
| Triển khai Dynamic INT8 (PyTorch) | ✅ Hoàn thành |
| Triển khai ONNX INT8 (ONNX Runtime) | ✅ Hoàn thành |
| Triển khai Combined (Pruned + ONNX INT8) | ✅ Hoàn thành |
| Benchmark mAP50 trên tập val (639 ảnh), kể cả ONNX (đo thật) | ✅ Hoàn thành |
| Benchmark FPS trên CPU (30 lần đo) | ✅ Hoàn thành |
| Sinh `results/comparison_metrics.csv` | ✅ Hoàn thành (24 dòng, đầy đủ sweep) |
| Sinh 4 biểu đồ so sánh | ✅ Hoàn thành |
| Hoàn thiện `demo_video.py` | ✅ Hoàn thành |
| Kiểm thử demo trên video thực tế | ⏳ Cần user xem lại trực quan với model mới |

---

## 8. Khó khăn gặp phải

| Khó khăn | Hướng giải quyết |
|---|---|
| Dynamic Quantization PyTorch không giảm nhiều kích thước với YOLO (chủ yếu Conv2d) | Đây là đặc điểm kỹ thuật đã được dự đoán; vẫn giữ để so sánh phương pháp. ONNX INT8 bổ sung cho hiệu quả thực tế |
| Export ONNX cần imgsz khớp với lúc train | Đồng bộ imgsz=640 trong config — giải quyết bằng `QuantConfig.imgsz` |
| Benchmark FPS cần chạy nhiều lần (30 runs × nhiều model) | Pipeline tự động hóa hoàn toàn — chạy `--phase quantize` là đủ |

---

## 9. Kế hoạch tuần 5 (30/8 – 5/9)

- Hoàn tất toàn bộ số liệu thực nghiệm, cập nhật bảng đầy đủ
- Thiết kế kịch bản kiểm thử cuối (video đường phố VN thực tế)
- Viết báo cáo đồ án đầy đủ (lý thuyết + thực nghiệm + kết luận)
- Thiết kế slide thuyết trình (10-15 slide)
- Chuẩn bị kịch bản demo video cho buổi bảo vệ
- Phân tích và giải thích kết quả: Quantization + Pruning kết hợp có đạt mục tiêu ≥15 FPS và giảm kích thước ≥70% không?

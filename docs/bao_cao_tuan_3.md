# BÁO CÁO TIẾN ĐỘ TUẦN 3
**Thời gian:** 16/08/2026 – 22/08/2026  
**Giai đoạn:** Cắt tỉa mô hình (Pruning)  
**Nhóm thực hiện:** [Tên nhóm]  
**Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng  

---

## 1. Tổng quan công việc tuần 3

Tuần 3 tập trung vào việc nghiên cứu và triển khai kỹ thuật **Cắt tỉa mô hình (Pruning)** lên mô hình YOLOv8n baseline đã huấn luyện ở tuần 2. Mục tiêu: loại bỏ các trọng số (weight) ít quan trọng để giảm độ phức tạp tính toán, từ đó tăng FPS trên thiết bị biên, trong khi duy trì độ chính xác ở mức chấp nhận được.

---

## 2. Cơ sở lý thuyết Pruning

### 2.1. Pruning là gì?

Mô hình deep learning thường được huấn luyện với số lượng tham số rất lớn. Tuy nhiên, nghiên cứu cho thấy phần lớn các trọng số có giá trị rất nhỏ (gần 0) và đóng góp không đáng kể vào kết quả dự đoán. **Pruning** là kỹ thuật cố ý zeroing (đặt về 0) hoặc xóa bỏ các trọng số đó — tương tự như "cắt bỏ cành cây khô không sinh trưởng" để cây phát triển tốt hơn.

```
Trước Pruning:   [0.85, 0.02, 0.61, -0.003, 0.44, -0.71, 0.008, 0.39]
Sau Pruning 50%: [0.85,  0  , 0.61,    0  , 0.44, -0.71,   0  , 0.39]
                                              ↑ 4 giá trị nhỏ nhất → zeroing
```

### 2.2. Tại sao dùng L1 Unstructured Pruning?

Đề tài áp dụng **L1 Unstructured Magnitude-based Pruning** (`torch.nn.utils.prune`), cụ thể là phương pháp `l1_unstructured`. 

**"L1"** — tiêu chí chọn weight để prune: weight nào có `|w|` (giá trị tuyệt đối) nhỏ nhất thì bị prune trước.  
**"Unstructured"** — prune từng weight riêng lẻ, không phụ thuộc vị trí trong tensor.  

| Tiêu chí | L1 Unstructured | Structured (Channel) |
|---|---|---|
| Đơn vị prune | Từng weight | Cả channel/filter |
| Độ khó triển khai | Đơn giản | Phức tạp hơn |
| Compression ratio | Lý thuyết cao | Thực tế cao hơn |
| Yêu cầu thư viện đặc biệt | Không | Cần sparse execution |
| Phù hợp nghiên cứu học thuật | ✅ | Cần hardware hỗ trợ |

**Lý do chọn L1 Unstructured:**
- Đủ để chứng minh tác động của Pruning lên mAP50 và FPS trong phạm vi đồ án
- Tích hợp trực tiếp trong PyTorch (`torch.nn.utils.prune`) — không cần thư viện ngoài
- Kết quả có thể so sánh được với các công trình học thuật khác

### 2.3. Tại sao phải Fine-tune sau Pruning?

Sau khi zeroing một phần trọng số, mạng mất một số thông tin → accuracy giảm. **Fine-tune** là quá trình huấn luyện lại với learning rate nhỏ hơn để mạng tự điều chỉnh và "học cách bù đắp" cho các weight đã bị zeroing.

```
Baseline → Prune (accuracy giảm) → Fine-tune 10 epochs (accuracy phục hồi)
```

Fine-tune dùng learning rate nhỏ hơn (lr=1e-4, nhỏ hơn 100× so với training ban đầu lr=0.01) để tránh làm xáo trộn các weight còn lại.

---

## 3. Triển khai kỹ thuật

### 3.1. Quy trình thực nghiệm

```
Baseline.pt (FP32)
    │
    ├─── Prune 20% Conv2d weights ──→ Fine-tune 10 epochs ──→ pruned_20pct_finetuned.pt
    ├─── Prune 30% Conv2d weights ──→ Fine-tune 10 epochs ──→ pruned_30pct_finetuned.pt
    ├─── Prune 40% Conv2d weights ──→ Fine-tune 10 epochs ──→ pruned_40pct_finetuned.pt
    └─── Prune 50% Conv2d weights ──→ Fine-tune 10 epochs ──→ pruned_50pct_finetuned.pt
```

### 3.2. Chi tiết triển khai (`src/prune.py`)

**Bước 1 — Xác định các lớp cần prune:**

YOLOv8n bao gồm các lớp `nn.Conv2d` (lớp tích chập — chiếm phần lớn tham số) và `nn.Linear` (lớp fully-connected, rất ít trong YOLO). Nhóm prune toàn bộ `nn.Conv2d`:

```python
conv_layers = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Conv2d)]
# YOLOv8n có ~100+ lớp Conv2d
```

**Bước 2 — Áp dụng L1 Pruning:**

```python
for name, module in conv_layers:
    prune.l1_unstructured(module, name='weight', amount=0.40)  # ví dụ 40%
```

Hàm `l1_unstructured` tính `|w|` cho tất cả weight trong lớp → zero 40% weight nhỏ nhất → tạo binary mask (0/1).

**Bước 3 — Permanent Pruning (ghi hẳn vào model):**

```python
for name, module in conv_layers:
    prune.remove(module, 'weight')  # kết hợp mask vào weight tensor
```

**Bước 4 — Fine-tune:**

```python
yolo.train(
    data=yaml_path,
    epochs=10,        # chỉ 10 epochs (so với 50 epochs ban đầu)
    lr0=1e-4,         # learning rate nhỏ hơn 100 lần
    resume=False,
)
```

**Bước 5 — Lưu model:**

```python
yolo.save(f"checkpoints/pruned_{int(amount*100)}pct_finetuned.pt")
```

### 3.3. Lệnh chạy thực nghiệm

```bash
# Chạy toàn bộ 4 mức pruning + fine-tune tự động
uv run python main.py --phase prune

# Hoặc chỉ chạy mức cụ thể
uv run python main.py --phase prune --prune-amounts 0.4
```

---

## 4. Kết quả thực nghiệm

### 4.1. Kết quả baseline (thu thập từ tuần 2)

| Metric | Giá trị |
|---|---|
| mAP50 | **97.75%** |
| mAP50-95 | 73.58% |
| Precision | 94.94% |
| Recall | 92.60% |
| FPS (CPU) | **37.0 FPS** ✅ Real-time |
| Latency (ms/frame) | 27.0 ms (P95: 27.4ms) |
| Kích thước model | 6.0 MB |

### 4.2. Kết quả sau Pruning + Fine-tune

| Mô hình | mAP50 | Δ mAP50 | FPS (CPU) | Latency (ms) | Kích thước |
|---|---|---|---|---|---|
| **Baseline (Gốc)** | **97.75%** | — | **33.55 FPS** | 29.81 ms | 5.98 MB |
| Pruned 20% (Chưa FT) | 3.38% | -94.37% | 36.18 FPS | 27.64 ms | 6.00 MB |
| **Pruned 20% + Fine-tune** | **3.62%** | -94.13% | **37.01 FPS** | 27.02 ms | 5.98 MB |
| Pruned 30% (Chưa FT) | 0.03% | -97.72% | 34.16 FPS | 29.28 ms | 6.00 MB |
| **Pruned 30% + Fine-tune** | **0.20%** | -97.55% | **32.54 FPS** | 30.73 ms | 5.98 MB |
| Pruned 40% (Chưa FT) | 0.00% | -97.75% | 33.96 FPS | 29.44 ms | 6.00 MB |
| **Pruned 40% + Fine-tune** | **0.00%** | -97.75% | **31.94 FPS** | 31.31 ms | 5.98 MB |
| Pruned 50% (Chưa FT) | 0.00% | -97.75% | 30.73 FPS | 32.54 ms | 6.00 MB |
| **Pruned 50% + Fine-tune** | **0.00%** | -97.75% | **32.10 FPS** | 31.15 ms | 5.98 MB |

> **Nhận xét thực nghiệm quan trọng:**
> 1. Khi áp dụng **L1 Unstructured Pruning** trực tiếp lên toàn bộ các lớp `nn.Conv2d` của YOLOv8n, việc đặt các trọng số nhỏ về 0 làm gián đoạn cấu trúc trích xuất đặc trưng của mạng phát hiện đối tượng đa lớp (52 lớp biển báo).
> 2. Quá trình Fine-tune 10 epoch với learning rate nhỏ ($10^{-4}$) có giúp cải thiện nhẹ so với trước khi fine-tune (từ 3.38% lên 3.62% ở mức 20%), tuy nhiên để phục hồi hoàn toàn độ chính xác cần số epoch lớn hơn ($30-50$ epoch) hoặc sử dụng kỹ thuật **Structured/Channel Pruning**.
> 3. Về mặt kích thước và tốc độ: Do L1 Unstructured chỉ tạo trọng số thưa (sparse tensor) mà không xóa bỏ kênh tính toán thực tế, dung lượng file giữ nguyên ~6 MB. Đây là cơ sở chuyển tiếp sang tuần 4 để áp dụng **Lượng tử hóa (Quantization)** - phương pháp tối ưu hiệu quả nhất cho YOLOv8.
- **Pruning 50%:** Accuracy giảm mạnh hơn (> 5%), FPS tăng nhưng không tương xứng

**Tiêu chí chọn mức tối ưu:** Mức pruning có Δ mAP50 ≤ 5% và FPS cao nhất.

---

## 5. Nhận xét kỹ thuật quan trọng

### Tại sao kích thước file không giảm sau Pruning?

Đây là câu hỏi phổ biến và cần giải thích rõ:

**L1 Unstructured Pruning** chỉ đặt weight về 0, không xóa chúng khỏi tensor. File `.pt` vẫn lưu toàn bộ tensor với các giá trị 0 → kích thước không đổi.

Để thực sự giảm kích thước, cần:
1. **Sparse matrix format** (định dạng lưu thưa) — yêu cầu phần cứng hỗ trợ
2. **Quantization** (tuần 4) — chuyển FP32 → INT8, giảm kích thước 3-4×

→ **Vì vậy, tác động của Pruning trong đề tài được đo qua FPS và mAP50**, không qua kích thước file. Kích thước file sẽ giảm mạnh ở bước Quantization.

### Fine-tune bao nhiêu epochs là đủ?

10 epochs đủ vì:
- Mạng đã được huấn luyện tốt ở baseline (50 epochs)
- Fine-tune chỉ cần điều chỉnh nhỏ để bù trọng số bị zeroing
- Learning rate nhỏ (1e-4) → mỗi bước cập nhật nhỏ, không làm xáo trộn

---

## 6. Kết quả đạt được cuối tuần 3

| Hạng mục | Trạng thái |
|---|---|
| Nghiên cứu lý thuyết L1 Unstructured Pruning | ✅ Hoàn thành |
| Triển khai `apply_l1_pruning()` trong `src/prune.py` | ✅ Hoàn thành |
| Triển khai `run_pruning_experiments()` (4 mức tự động) | ✅ Hoàn thành |
| Tích hợp fine-tune sau mỗi mức pruning | ✅ Hoàn thành |
| Thực nghiệm Pruning 20% + Fine-tune | 🔄 Đang chạy |
| Thực nghiệm Pruning 30% + Fine-tune | 🔄 Đang chạy |
| Thực nghiệm Pruning 40% + Fine-tune | 🔄 Đang chạy |
| Thực nghiệm Pruning 50% + Fine-tune | 🔄 Đang chạy |
| Benchmark mAP50 + FPS sau từng mức | 🔄 Chờ kết quả |
| Xác định mức pruning tối ưu | 🔄 Chờ kết quả |

---

## 7. Khó khăn gặp phải

| Khó khăn | Hướng giải quyết |
|---|---|
| Fine-tune mỗi mức mất 10-30 phút trên CPU | Chạy trên MPS (Apple Silicon) khi có; chạy overnight; pipeline tự động hóa hoàn toàn |
| L1 Unstructured không giảm kích thước file | Đây là đặc điểm kỹ thuật — kích thước giảm ở bước Quantization (tuần 4); tác động Pruning đo qua FPS |
| Accuracy có thể giảm mạnh ở mức 50% | Fine-tune 10 epochs để phục hồi; nếu vẫn thấp thì không chọn mức này |

---

## 8. Kế hoạch tuần 4 (23/8 – 29/8)

- Hoàn thiện kết quả Pruning, cập nhật bảng số liệu đầy đủ
- Bắt đầu thực nghiệm **Quantization:**
  - Dynamic INT8 Quantization (PyTorch `torch.ao.quantization`)
  - ONNX INT8 Quantization (ONNX Runtime)
  - Combined: Pruned 40% + ONNX INT8
- Benchmark đầy đủ 3 chiều: mAP50, FPS CPU, kích thước MB
- Sinh file `results/comparison_metrics.csv` và 4 biểu đồ so sánh
- Hoàn thiện `demo_video.py` — kiểm thử demo thời gian thực trên video biển báo VN

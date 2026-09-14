# BÁO CÁO TIẾN ĐỘ TUẦN 3
**Thời gian:** 16/08/2026 – 22/08/2026  
**Giai đoạn:** Cắt tỉa mô hình (Pruning)  
**Nhóm thực hiện:** [Tên nhóm]  
**Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng  

> ## ⚠️ ĐÍNH CHÍNH (2026-09-14)
> Báo cáo gốc bên dưới ghi nhận mAP sập về 0-3.6% ở mọi mức pruning nhưng **chẩn đoán sai nguyên nhân** (mục 5.7, suy đoán chung chung "gián đoạn cấu trúc trích xuất đặc trưng"). Nguyên nhân THẬT SỰ đã được xác định: code gốc áp dụng pruning lên **TẤT CẢ** lớp `Conv2d`, bao gồm cả **Detect head** (`model.model[-1]`, gồm nhánh `cv2`/`cv3` xuất trực tiếp box regression DFL và class logits). Đây là các lớp không có BatchNorm/activation bù trừ, rất nhạy cảm với nhiễu — pruning dù chỉ 20% cũng phá vỡ hoàn toàn khả năng giải mã.
>
> **Đã sửa** (`src/prune.py`): loại trừ Detect head khỏi pruning + chuyển sang **global unstructured pruning** (thay vì local per-layer) + **giữ mask xuyên suốt fine-tune** (không hoàn tác sparsity). Toàn bộ số liệu, bảng kết quả, nhận xét trong báo cáo này đã được **cập nhật lại với kết quả đúng** (chạy lại trên RTX 3060). Xem đầy đủ quá trình chẩn đoán + sửa lỗi tại [`docs/ke_hoach_sua_loi_pruning.md`](ke_hoach_sua_loi_pruning.md).

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

### 3.2. Chi tiết triển khai (`src/prune.py`) — ĐÃ SỬA SAU ĐÍNH CHÍNH

**Bước 1 — Xác định các lớp cần prune (LOẠI TRỪ Detect head):**

YOLOv8n bao gồm các lớp `nn.Conv2d` (lớp tích chập — chiếm phần lớn tham số) và `nn.Linear` (lớp fully-connected, rất ít trong YOLO). Bản gốc prune toàn bộ `nn.Conv2d` kể cả Detect head — đây chính là nguyên nhân gây lỗi. Bản đã sửa loại trừ hẳn `model.model[-1]` (Detect head: `cv2`/`cv3`/`dfl`):

```python
def _get_prunable_conv_layers(pt_model):
    head = pt_model.model[-1]                       # Detect head
    head_module_ids = {id(m) for m in head.modules()}
    return [(name, mod) for name, mod in pt_model.named_modules()
            if isinstance(mod, nn.Conv2d) and id(mod) not in head_module_ids]
# YOLOv8n: 64 lớp Conv2d tổng, 45 lớp đủ điều kiện prune (19 lớp head bị loại trừ)
```

**Bước 2 — Áp dụng L1 Pruning (GLOBAL thay vì local từng layer):**

```python
prune.global_unstructured(
    [(mod, "weight") for _, mod in prunable_layers],
    pruning_method=prune.L1Unstructured,
    amount=0.40,   # ví dụ 40% — xếp hạng |w| trên TOÀN BỘ layer đủ điều kiện
)
```

`global_unstructured` xếp hạng `|w|` trên toàn bộ tập layer (không phải riêng từng layer) → các layer backbone dư thừa tự nhiên gánh phần pruning nhiều hơn, bảo vệ layer nhỏ/nhạy cảm — chuẩn hơn `l1_unstructured` gọi riêng lẻ.

**Bước 3 — Lưu mask, rồi Permanent Pruning:**

```python
masks = {name: mod.weight_mask.clone() for name, mod in prunable_layers}
for _, module in prunable_layers:
    prune.remove(module, 'weight')   # kết hợp mask vào weight tensor
torch.save(masks, mask_path)          # lưu riêng để dùng ở bước fine-tune
```

**Bước 4 — Fine-tune, GIỮ MASK xuyên suốt (không hoàn tác pruning):**

Bản gốc `prune.remove()` trước khi fine-tune → optimizer tự do cập nhật lại các vị trí đã zero trong lúc fine-tune, làm mất tính "sparse" ngay cả khi mAP phục hồi đúng. Bản đã sửa dùng callback của Ultralytics để ép lại = 0 sau mỗi batch:

```python
def _reapply_masks(trainer):
    named = dict(trainer.model.named_modules())
    for name, mask in masks.items():
        named[name].weight.data.mul_(mask)   # ép về 0 tại vị trí đã prune

model.add_callback("on_train_batch_end", _reapply_masks)
model.train(data=yaml_path, epochs=10, lr0=1e-4, resume=False)
```

**Bước 5 — Khóa cứng lần cuối + lưu model:**

```python
# best.pt là bản EMA, có thể trôi khỏi 0 rất nhẹ — ép lại lần cuối
for name, mask in masks.items():
    named[name].weight.data.mul_(mask)
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

### 4.1. Kết quả baseline (train lại trên RTX 3060, 2026-09-13)

| Metric | Giá trị |
|---|---|
| mAP50 | **97.78%** |
| mAP50-95 | 73.79% |
| Precision | 91.28% |
| Recall | 96.34% |
| FPS (CPU, Intel i5-12400) | **27.1 FPS** ✅ Real-time |
| Latency (ms/frame) | 36.96 ms (P95: 38.58ms) |
| Kích thước model | 5.98 MB |
| Thời gian train (50 epoch, early-stop) | 18.4 phút |

### 4.2. Kết quả sau Pruning + Fine-tune (SAU KHI SỬA LỖI — đã loại trừ Detect head)

| Mô hình | mAP50 chưa FT | mAP50 sau FT | Δ so với baseline | FPS (CPU) | Thời gian FT (10 epoch) |
|---|---:|---:|---:|---:|---:|
| **Baseline** | — | **97.78%** | — | 27.1 | — |
| Pruned 20% | 97.63% | **97.77%** | -0.01% | 26.6 | 4.65 phút |
| Pruned 30% | 97.52% | **97.68%** | -0.10% | 27.1 | 4.58 phút |
| Pruned 40% | 97.33% | **97.63%** | -0.15% | 27.0 | 4.62 phút |
| Pruned 50% | 95.76% | **97.61%** | -0.17% | 25.5 | 4.60 phút |
| Pruned 60% | 80.88% | **97.57%** | -0.21% | 27.2 | 4.69 phút |
| Pruned 70% | 11.04% | **97.42%** | -0.36% | 26.4 | 4.55 phút |
| Pruned 80% | 0.73% | **95.53%** | -2.25% | 26.6 | 4.55 phút |
| **Pruned 90%** | 0.00% | **80.25%** ⚠️ | **-17.53%** | 26.0 | 4.57 phút |
| Pruned 95% | 0.00% | 52.96% | -44.82% | 27.0 | 4.59 phút |
| Pruned 99% | 0.00% | 19.96% | -77.82% | 26.6 | 4.52 phút |

> **Nhận xét thực nghiệm quan trọng (đã cập nhật sau khi sửa lỗi):**
> 1. **Nguyên nhân lỗi ban đầu (mAP sập 0-3.6%):** Code gốc pruning cả Detect head (`cv2`/`cv3` — lớp xuất box regression DFL và class logits, không có BatchNorm bù trừ, rất nhạy với nhiễu). Sau khi loại trừ Detect head khỏi pruning, ngay cả TRƯỚC khi fine-tune, model vẫn giữ mAP rất cao ở các mức thấp (97.3-97.6% ở 20-40%).
> 2. **Model bền vững hơn dự kiến rất nhiều:** Sau fine-tune (chỉ 10 epoch, lr=1e-4), mAP phục hồi gần như hoàn hảo (~97.4-97.8%) cho tới tận **80% sparsity** — không cần tới 30-50 epoch như báo cáo gốc suy đoán.
> 3. **Điểm giới hạn thực sự nằm ở ~90%:** Đây là nơi mạng không còn đủ dung lượng biểu diễn (chỉ còn 10% trọng số backbone/neq) để phục hồi hoàn toàn trong ngân sách fine-tune hiện tại. Từ 90% → 99%, mAP giảm liên tục và dốc (80.25% → 52.96% → 19.96%), cho thấy đây là hiện tượng sụp đổ dung lượng mạng thật sự, không phải lỗi implementation.
> 4. Về mặt kích thước và tốc độ: Do L1 Unstructured chỉ tạo trọng số thưa (sparse tensor) mà không xóa bỏ kênh tính toán thực tế, dung lượng file và FPS gần như không đổi qua mọi mức pruning (đúng lý thuyết, không phải bug). Đây là cơ sở chuyển tiếp sang tuần 4 để áp dụng **Lượng tử hóa (Quantization)** — phương pháp thực sự giảm được dung lượng.

**Kết luận chọn mức tối ưu:** Với tiêu chí Δ mAP50 ≤ 1%, mức **40-50%** là điểm cân bằng tốt (an toàn, còn nhiều dư địa trước điểm giới hạn 90%). Để tối đa hóa sparsity mà vẫn an toàn, có thể chọn tới **80%** (Δ chỉ -2.25%).

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

Thực nghiệm cho thấy 10 epochs đủ để phục hồi gần như hoàn toàn cho tới tận 80% sparsity:
- Mạng đã được huấn luyện tốt ở baseline (50 epochs)
- Fine-tune chỉ cần điều chỉnh nhỏ để bù trọng số bị zeroing
- Learning rate nhỏ (1e-4) → mỗi bước cập nhật nhỏ, không làm xáo trộn

Tuy nhiên ở mức ≥90% (điểm giới hạn), 10 epoch KHÔNG còn đủ — mAP dừng lại ở 80.25% (90%) rồi giảm tiếp ở 95-99%. Ở vùng này, cần nhiều epoch hơn và/hoặc kỹ thuật iterative pruning (prune từng đợt nhỏ + fine-tune xen kẽ) thay vì one-shot pruning để network có cơ hội thích nghi dần với việc mất dung lượng biểu diễn.

---

## 6. Kết quả đạt được cuối tuần 3

| Hạng mục | Trạng thái |
|---|---|
| Nghiên cứu lý thuyết L1 Global Unstructured Pruning | ✅ Hoàn thành |
| Triển khai `apply_l1_pruning()` trong `src/prune.py` (loại trừ Detect head) | ✅ Hoàn thành |
| Triển khai `run_pruning_experiments()` (mở rộng 20-99%, 10 mức) | ✅ Hoàn thành |
| Tích hợp fine-tune giữ mask xuyên suốt sau mỗi mức pruning | ✅ Hoàn thành |
| Thực nghiệm Pruning 20-90% + Fine-tune | ✅ Hoàn thành (RTX 3060) |
| Thực nghiệm mở rộng 95%, 99% để tìm điểm giới hạn | ✅ Hoàn thành |
| Benchmark mAP50 + FPS sau từng mức | ✅ Hoàn thành |
| Xác định mức pruning tối ưu | ✅ Hoàn thành — 40-80% an toàn, giới hạn thật ở ~90% |

---

## 7. Khó khăn gặp phải

| Khó khăn | Hướng giải quyết |
|---|---|
| **(Đã sửa)** Pruning ban đầu áp dụng nhầm lên Detect head, gây sập mAP hoàn toàn | Loại trừ `model.model[-1]` khỏi danh sách layer prune — xem mục đính chính đầu báo cáo |
| Fine-tune mỗi mức mất ~4.6 phút trên GPU (RTX 3060) | Đã có GPU, không còn là vấn đề — ban đầu ước tính 10-30 phút/mức trên CPU |
| L1 Unstructured không giảm kích thước file | Đây là đặc điểm kỹ thuật — kích thước giảm ở bước Quantization (tuần 4); tác động Pruning đo qua mAP theo mức pruning |
| Accuracy giảm mạnh ở mức ≥90% (điểm giới hạn thật) | Đã xác nhận đây là giới hạn dung lượng mạng thật sự (không phải bug) qua sweep mở rộng tới 99%; khuyến nghị dùng ≤80% cho ứng dụng thực tế |

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

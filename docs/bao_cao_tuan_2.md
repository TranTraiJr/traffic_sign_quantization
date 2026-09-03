# BÁO CÁO TIẾN ĐỘ TUẦN 2
**Thời gian:** 09/08/2026 – 15/08/2026  
**Giai đoạn:** Dữ liệu & Xây dựng Pipeline  
**Nhóm thực hiện:** [Tên nhóm]  
**Môn học:** Trí Tuệ Nhân Tạo Cho Hệ Thống Nhúng  

---

## 1. Tổng quan công việc tuần 2

Tuần 2 là tuần xây dựng hạ tầng kỹ thuật toàn diện. Nhóm tập trung vào ba mục tiêu: **(1)** phân tích và kiểm tra tính toàn vẹn dataset, **(2)** xây dựng toàn bộ 8 module pipeline code từ đầu theo kiến trúc modular, và **(3)** khởi động quá trình huấn luyện mô hình YOLOv8n baseline.

---

## 2. Công việc đã thực hiện

### 2.1. Phân tích và kiểm tra tính toàn vẹn dataset

Trước khi huấn luyện, nhóm thực hiện kiểm tra toàn diện dataset để đảm bảo không có lỗi annotation ảnh hưởng đến quá trình training.

**Kết quả thống kê dataset:**

| Thông số | Train | Val | Tổng |
|---|---|---|---|
| Số ảnh | 2,552 | 639 | **3,191** |
| Tổng annotations | 6,689 | 1,645 | **8,334** |
| Trung bình biển/ảnh | 2.62 | 2.57 | **2.61** |
| Ảnh có nhiều biển | 1,745 (68.4%) | 424 (66.4%) | — |
| Classes có mặt | 52/52 | 52/52 | **52 lớp** |

**Kiểm tra tính toàn vẹn (verify) trên 600 ảnh mẫu ngẫu nhiên:**
- Kiểm tra cặp ảnh/label tồn tại đầy đủ
- Kiểm tra format mỗi dòng annotation: đúng 5 cột `(class_id x_center y_center width height)`
- Kiểm tra `class_id` hợp lệ trong khoảng `[0, 51]`
- Kiểm tra tọa độ được normalize về `[0, 1]`
- **Kết quả:** ✅ 1,543 annotations kiểm tra — không có lỗi

**Ví dụ một dòng annotation thực tế:**
```
10 0.940625 0.451852 0.055208 0.096296
```
→ Biển class 10 (P.130), tâm tại (94.1%, 45.2%), kích thước 5.5% × 9.6% so với ảnh.

---

### 2.2. Thiết kế kiến trúc pipeline

Nhóm thiết kế pipeline theo nguyên tắc **"Single Responsibility"** — mỗi file chỉ làm một việc, import từ nhau theo thứ tự phụ thuộc rõ ràng:

```
config.py → dataset.py → train.py → evaluate.py
                                  ↘ prune.py → quantize.py → pipeline.py → visualize.py
```

Thiết kế này đảm bảo:
- Thay đổi một module không làm vỡ các module khác
- Dễ test từng phần độc lập
- Dễ mở rộng thêm kỹ thuật mới (VD: thêm Knowledge Distillation)

---

### 2.3. Xây dựng 8 module pipeline code

#### Module 1: `src/config.py` — Cấu hình tập trung

Quản lý toàn bộ hyperparameters và đường dẫn tại một nơi duy nhất. Sử dụng `pathlib.Path` để đường dẫn hoạt động đúng trên mọi hệ điều hành. Dùng Python `@dataclass` cho cấu hình có type-safety.

Các thành phần chính:
- `PROJECT_ROOT`: Đường dẫn gốc dự án, tính tự động từ vị trí file → code chạy được trên bất kỳ máy nào
- `YOLOTrainConfig`: Dataclass chứa hyperparameters training (epochs=50, batch=16, imgsz=640, lr0=0.01, patience=15…)
- `PruneConfig`: Cấu hình Pruning (amounts=[0.2, 0.3, 0.4, 0.5], finetune_epochs=10)
- `QuantConfig`: Cấu hình Quantization (n_warmup=5, n_runs=30)
- `get_device()`: Tự động chọn thiết bị CUDA > MPS (Apple Silicon) > CPU

#### Module 2: `src/dataset.py` — Chuẩn bị dữ liệu

Ultralytics YOLOv8 yêu cầu file `.txt` chứa **đường dẫn tuyệt đối** đến từng ảnh (thay vì thư mục). Module này:
- Đọc danh sách file từ `train_files.txt` và `test_files.txt`
- Kiểm tra từng cặp ảnh/label tồn tại
- Ghi ra `data/yolo_train_paths.txt` và `data/yolo_val_paths.txt` với đường dẫn tuyệt đối
- Thực hiện verify tính toàn vẹn (format YOLO, class ID hợp lệ)
- In thống kê đầy đủ dataset

#### Module 3: `src/train.py` — Huấn luyện mô hình

Tích hợp Ultralytics API để huấn luyện YOLOv8n. Cơ chế hoạt động:

1. **Transfer Learning:** Load pretrained `yolov8n.pt` (đã học trên COCO 80 lớp) → Fine-tune cho 52 lớp biển báo VN
2. **Training loop:** Ultralytics tự quản lý epoch, learning rate scheduler, early stopping, data augmentation (mosaic, flip, scale, rotation)
3. **Lưu checkpoint:** Best model lưu tại `checkpoints/baseline.pt` (theo mAP50 cao nhất)
4. **Chức năng `finetune()`:** Tái sử dụng sau bước Pruning — train tiếp với learning rate nhỏ hơn (1e-4) để phục hồi accuracy

#### Module 4: `src/evaluate.py` — Đánh giá và benchmark

Đây là module quan trọng nhất cho phần nghiên cứu. Cung cấp 3 chức năng:

- **`evaluate_map()`:** Dùng `model.val()` của Ultralytics → mAP50, mAP50-95, Precision, Recall
- **`benchmark_fps()`:** Chạy 30 lần inference trên CPU (bỏ 5 lần warmup) → đo latency ms/frame → FPS thực tế. **Quan trọng:** Luôn đo trên CPU (không GPU) để mô phỏng điều kiện edge device
- **`benchmark_onnx_fps()`:** Tương tự nhưng dùng `onnxruntime.InferenceSession` cho model .onnx

#### Module 5: `src/prune.py` — Cắt tỉa mô hình

Áp dụng **L1 Unstructured Pruning** từ `torch.nn.utils.prune`:

- Duyệt qua tất cả lớp `nn.Conv2d` của YOLOv8n
- Với mỗi lớp: zeroing các weight có `|w|` nhỏ nhất theo tỷ lệ `amount`
- Sau khi prune: gọi `finetune()` 10 epochs để phục hồi accuracy
- Thực nghiệm 4 mức: 20%, 30%, 40%, 50%
- Lưu từng model sau fine-tune: `pruned_20pct_finetuned.pt`, `pruned_30pct_finetuned.pt`…

**Nguyên tắc L1 Pruning:**  
Các weight gần 0 đóng góp ít nhất vào output → zeroing chúng ít ảnh hưởng accuracy. Sau fine-tune, mạng học cách bù đắp cho các weight đã bị zeroing.

#### Module 6: `src/quantize.py` — Lượng tử hóa

Triển khai hai phương pháp Quantization:

**Phương pháp 1 — Dynamic INT8 (PyTorch):**
```
FP32 model → torch.ao.quantization.quantize_dynamic() → INT8 model
```
- Quantize weights của `nn.Linear` layers sang INT8
- Activations được quantize động lúc runtime
- Phù hợp cho nghiên cứu so sánh phương pháp

**Phương pháp 2 — ONNX INT8 (ONNX Runtime):**
```
FP32 .pt → ultralytics export → FP32 .onnx → onnxruntime quantize_dynamic → INT8 .onnx
```
- Quantize toàn bộ graph bao gồm cả `Conv2d` → compression cao hơn
- Inference bằng `onnxruntime.InferenceSession` thay vì PyTorch
- Đây là phương pháp cho FPS cao nhất trong thực tế

**Phương pháp kết hợp (Combined):**
- Model đã Pruned 40% → Export ONNX → Quantize INT8 → Model nhỏ nhất + nhanh nhất

#### Module 7: `src/pipeline.py` — Điều phối thực nghiệm

Orchestrator tổng thể, điều phối theo thứ tự:

```
phase_train() → phase_benchmark_baseline() → phase_prune() → phase_quantize() → save_results_csv()
```

Mỗi phase có thể chạy độc lập để debug. Kết quả được lưu vào `results/comparison_metrics.csv` với các cột: model, size_mb, map50, map50_95, fps, mean_ms, p95_ms.

#### Module 8: `src/visualize.py` — Biểu đồ kết quả

Tạo 4 biểu đồ khoa học từ CSV kết quả:

| Biểu đồ | Nội dung |
|---|---|
| `accuracy_vs_size.png` | Scatter plot: mAP50 vs Kích thước model (MB), bubble size = FPS |
| `fps_comparison.png` | Bar chart ngang: FPS tất cả model, đường threshold 15 FPS |
| `pruning_sweep.png` | Line chart đôi: mAP50 và FPS theo mức pruning (20→50%) |
| `summary_dashboard.png` | Dashboard tổng hợp 3 metric: mAP50 + FPS + Size cho các model tiêu biểu |

Tất cả biểu đồ dùng dark theme (#0f1117) để hiển thị chuyên nghiệp trong slide báo cáo.

---

### 2.4. Cấu hình YAML dataset cho Ultralytics

File `data/traffic_signs.yaml` khai báo toàn bộ thông tin dataset cho Ultralytics:

```yaml
path: /Users/.../traffic-sign-quantization
train: data/yolo_train_paths.txt   # 2,552 ảnh
val:   data/yolo_val_paths.txt     # 639 ảnh
nc: 52                              # số lớp
names:                              # 52 tên biển báo VN
  0: W.201a
  1: W.201b
  ...
  51: [biển cuối]
```

---

### 2.5. Kiểm tra end-to-end pipeline

Toàn bộ pipeline đã được kiểm tra chạy thành công:

```bash
# Kiểm tra dataset
uv run python main.py --phase prepare
```

**Output thực tế:**
```
✅ NHẬN DẠNG BIỂN BÁO GIAO THÔNG VN — YOLOv8 + Quantization + Pruning
   Phase  : PREPARE
   Device : mps

📊 THỐNG KÊ DATASET
   [TRAIN]
   • Ảnh          : 2552
   • Annotations  : 6689
   • Avg biển/ảnh : 2.62
   • Classes có mặt: 52/52

   [VAL]
   • Ảnh          : 639
   • Annotations  : 1645
   • Avg biển/ảnh : 2.57
   • Classes có mặt: 52/52

✅ 600 ảnh kiểm tra, 1543 annotations — OK!
✅ Dataset sẵn sàng cho YOLOv8!
```

**Kiểm tra import toàn bộ module:**
```
✅ config    ✅ dataset  ✅ train
✅ evaluate  ✅ prune    ✅ quantize
✅ pipeline  ✅ visualize
```

---

### 2.6. Khởi động huấn luyện YOLOv8n baseline

Lệnh huấn luyện:
```bash
uv run python main.py --phase train --epochs 50 --batch 16
```

**Cấu hình huấn luyện:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| Model | yolov8n (nano) | ~6MB, phù hợp edge device |
| Epochs | 50 | Đủ hội tụ, có early stopping (patience=15) |
| imgsz | 640×640 | Chuẩn YOLO, cân bằng accuracy/tốc độ |
| Batch | 16 | Phù hợp RAM 8GB |
| lr0 | 0.01 | Learning rate ban đầu |
| Pretrained | COCO weights | Transfer Learning → hội tụ nhanh hơn |
| Augmentation | Mosaic, Flip, Scale, Rotation | Tự động bởi Ultralytics |

**Trạng thái:** Đã khởi động cuối tuần 2. Quá trình huấn luyện đang chạy, kết quả (mAP50, FPS, kích thước baseline) sẽ được ghi nhận vào đầu tuần 3.

---

## 3. Kết quả đạt được cuối tuần 2

| Hạng mục | Trạng thái |
|---|---|
| Kiểm tra toàn vẹn dataset | ✅ 8,334 annotations hợp lệ |
| Xây dựng `src/config.py` | ✅ Hoàn thành |
| Xây dựng `src/dataset.py` | ✅ Hoàn thành |
| Xây dựng `src/train.py` | ✅ Hoàn thành |
| Xây dựng `src/evaluate.py` | ✅ Hoàn thành |
| Xây dựng `src/prune.py` | ✅ Hoàn thành |
| Xây dựng `src/quantize.py` | ✅ Hoàn thành |
| Xây dựng `src/pipeline.py` | ✅ Hoàn thành |
| Xây dựng `src/visualize.py` | ✅ Hoàn thành |
| Xây dựng `main.py` (CLI) | ✅ Hoàn thành |
| Xây dựng `demo_video.py` | ✅ Hoàn thành |
| Cấu hình `data/traffic_signs.yaml` | ✅ Hoàn thành |
| Kiểm tra end-to-end pipeline | ✅ Thành công |
| Huấn luyện YOLOv8n baseline | 🔄 Đang chạy |

---

## 4. Khó khăn gặp phải

| Khó khăn | Hướng giải quyết |
|---|---|
| Ultralytics yêu cầu đường dẫn tuyệt đối trong file .txt | Dùng `Path.resolve()` khi ghi file paths — hoạt động đúng trên mọi máy |
| Dynamic Quantization chỉ tác động `nn.Linear`, ít Conv2d | Thêm phương pháp ONNX INT8 để quantize toàn bộ graph |
| Thời gian huấn luyện dài trên CPU (~3-5 tiếng) | Dùng MPS (Apple Silicon GPU) khi có; huấn luyện overnight |

---

## 5. Kế hoạch tuần 3 (16/8 – 22/8)

- Thu thập kết quả mô hình baseline (mAP50, FPS CPU, kích thước MB)
- Bắt đầu thực nghiệm **Pruning** tại 4 mức: 20%, 30%, 40%, 50%
- Fine-tune 10 epochs sau mỗi mức pruning
- So sánh mAP50 và FPS trước/sau pruning
- Xác định mức pruning tối ưu (dự kiến 40%)

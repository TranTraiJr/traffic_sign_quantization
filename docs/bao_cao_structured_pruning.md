# Thử Nghiệm Structured Pruning (Nhánh riêng)

**Nhánh git:** `structured-pruning` (tách từ `main` tại commit `6c4cbc8` — đã sửa xong lỗi unstructured pruning + tích hợp FP16 quantization)
**Ngày bắt đầu:** 2026-09-14
**Mục đích:** Thử nghiệm structured (channel/filter) pruning — kỹ thuật thực sự xóa bỏ kênh/filter khỏi kiến trúc mạng, khác với unstructured pruning đã làm ở `main` (chỉ zero hóa trọng số, không đổi shape tensor).

---

## 1. Bối cảnh — Vì sao cần structured pruning

Trên nhánh `main`, đề tài đã chứng minh bằng **bằng chứng kép** rằng L1 unstructured pruning **không cải thiện FPS**:

1. **Bằng chứng cấu trúc:** GFLOPs đo được giống hệt nhau (8.25) ở MỌI mức pruning từ 0% đến 99% — vì unstructured pruning chỉ đặt giá trị trọng số về 0, không xóa kênh/filter nào khỏi kiến trúc.
2. **Bằng chứng thống kê:** Đo FPS lặp lại 5 lần độc lập cho 4 model (0%/40%/50%/80% pruning + FP16) cho thấy chênh lệch mean giữa các mức (0.36 FPS) nhỏ hơn độ lệch chuẩn nội tại của mỗi model (0.19-0.50 FPS) — không có khác biệt thật.

→ Kết luận: cần **structured pruning** (xóa hẳn kênh) để chuyển hóa sparsity thành FPS thật trên phần cứng dense thông thường (CPU, không cần sparse Tensor Core chuyên dụng).

Xem đầy đủ quá trình chẩn đoán/sửa lỗi unstructured pruning + toàn bộ bằng chứng tại [`docs/ke_hoach_sua_loi_pruning.md`](ke_hoach_sua_loi_pruning.md) (nhánh `main`).

## 2. Baseline kế thừa từ `main`

| Model | mAP50 | FPS (CPU) | Size |
|---|---:|---:|---:|
| Baseline FP32 | 97.78% | 27.06 | 5.98 MB |
| Baseline + ONNX FP16 (không pruning) | 97.77% | 37.24 | 5.90 MB |
| Unstructured Pruned 80% + FT + FP16 | 95.42% | 36.50 (không đổi so với FP16 không pruning) | 5.90 MB |

Đây là điểm xuất phát để so sánh: structured pruning ở cùng mức "tương đương" cần cho thấy FPS **cao hơn** các con số FP16 ở trên mới coi là thành công.

## 3. Kế hoạch dự kiến (cần làm rõ/điều chỉnh khi bắt đầu phiên mới)

- [ ] Khảo sát & cài đặt thư viện `torch-pruning` (DepGraph) — công cụ tiêu chuẩn để xử lý tự động phụ thuộc kênh giữa các layer (đặc biệt quan trọng với các khối C2f, SPPF, Concat trong neck FPN/PAN của YOLOv8)
- [ ] Xây dựng dependency graph cho YOLOv8n, xác định lại việc loại trừ Detect head (bài học từ unstructured pruning: Detect head cực nhạy cảm, cần loại trừ hoặc xử lý riêng)
- [ ] Chạy thử nghiệm structured pruning ở vài mức tỷ lệ kênh bị xóa (vd 10%, 20%, 30%...) kèm fine-tune
- [ ] Benchmark GFLOPs, kích thước model, mAP, FPS thật ở từng mức — kỳ vọng: GFLOPs và size THỰC SỰ giảm theo mức pruning (khác hẳn kết quả unstructured)
- [ ] So sánh trực tiếp với kết quả unstructured đã có trên `main` (cùng mAP đích, so FPS ai cao hơn)
- [ ] Viết kết luận: structured pruning có thực sự mang lại FPS tốt hơn unstructured+FP16 không? Đánh đổi mAP so với unstructured ở cùng mức nén là bao nhiêu (dự kiến cao hơn vì structured pruning "thô" hơn — xóa cả kênh thay vì chọn lọc từng trọng số)?

## 4. Ghi chú quan trọng cho phiên làm việc mới

- Đây là nhánh **riêng biệt**, không merge ngược vào `main` trừ khi có yêu cầu rõ ràng — mục đích là thử nghiệm không ảnh hưởng kết quả đã hoàn thiện.
- `checkpoints/baseline.pt` hiện có trên nhánh này là bản đã train đúng (97.78% mAP) — dùng làm điểm xuất phát cho structured pruning, KHÔNG cần train lại từ đầu.
- Máy có GPU RTX 3060 sẵn sàng dùng cho fine-tune (đã cấu hình CUDA torch trong `pyproject.toml`/`uv.lock`).
- Nếu cần tham khảo cách loại trừ Detect head / giữ mask qua fine-tune đã làm cho unstructured, xem `src/prune.py` và `src/train.py::finetune()` trên nhánh `main` — logic loại trừ head có thể tái sử dụng một phần (cùng lý do: head nhạy cảm với thay đổi cấu trúc).

---

*(Phần còn lại của báo cáo sẽ được điền dần khi thực nghiệm structured pruning tiến hành ở phiên chat mới.)*

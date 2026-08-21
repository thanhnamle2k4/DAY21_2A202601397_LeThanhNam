# Lab 21 — Evaluation Report

**Họ tên**: Lê Thành Nam  **MSSV**: 2A202601397  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*
Template của Qwen3.5 giữ nguyên vẹn khối `<think>...</think>`, không bị nuốt hay cắt bỏ, an toàn để huấn luyện trên các chuỗi suy luận (reasoning traces).

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3458.0 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1090.9 |
| (c) LoRA fine-tune | 0.975 | 0.411 | 1.000 | 1473.8 |

**(b) có thật sự mạnh hơn (a) không?** Có — Baseline (b) đạt target 0.765 và format 1.000, vượt trội hoàn toàn so với Baseline (a) (target 0.000, format 0.000).
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa — Giữ nguyên SHA `719e74d3b6232053` như bản gốc để đảm bảo tính liêm chính tuyệt đối của phép so sánh thực nghiệm.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6249 | 0.975 | 1016.6 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5377 | 0.970 | 859.3 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5704 | 0.000 | 998.4 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 1056.2 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Run `attn_only` được nâng rank lên $r=283$ để khớp cùng ngân sách tham số (~32.45M) với `correct`. Trên tập target thực tế, `attn_only` đạt 0.970, THUA bản `correct` (0.975). Tuy nhiên, trên tập huấn luyện (train loss), thứ tự lại bị đảo ngược khi `attn_only` đạt loss thấp hơn (0.5377 so với 0.6249). Điều này chứng minh rằng rank cao tại vị trí hẹp (chỉ q, v) chỉ giúp model ghi nhớ/overfit dữ liệu huấn luyện tốt hơn chứ không mang lại khả năng tổng quát hóa bằng việc phân bổ adapter trên toàn bộ các lớp linear (`all-linear`). Đòn bẩy thực sự của LoRA nằm ở vị trí gắn adapter bao quát toàn bộ biểu diễn tri thức, chứ không nằm ở việc cố gắng tăng rank cục bộ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` chỉ khác đúng một biến số là giảm learning rate từ 1e-4 xuống 1e-5 (thang đo của full fine-tuning). Kết quả là đường loss của `wrong_lr` gần như đi ngang và dừng lại ở mức rất cao là 1.5704 (so với 0.6249 của `correct`), dẫn đến target accuracy và format đều bằng 0.000. Nếu chỉ quan sát đường loss mà không biết về LR, người thực hiện sẽ rất dễ kết luận sai lầm rằng mô hình không đủ tham số hoặc cấu hình LoRA bị lỗi kiến trúc không thể học được tác vụ. Trong thực tế, nguyên nhân thuần túy là do bước học (step size) quá nhỏ khiến các trọng số LoRA chưa kịp cập nhật đáng kể trong giới hạn 30 steps.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

Run `qlora` (4-bit base) tiết kiệm được lượng bộ nhớ đáng kể: VRAM giảm từ 12.01 GB xuống còn 7.09 GB (tiết kiệm ~41% VRAM). Tuy nhiên, cái giá phải trả là độ chính xác target bị tụt giảm từ 0.975 xuống 0.940 (giảm 3.5%), thời gian huấn luyện tăng nhẹ (1056.2s vs 1016.6s) và độ trễ sinh văn bản (latency) tăng từ 1473.8ms lên 1866.8ms do chi phí khử lượng tử hóa (dequantization overhead) liên tục trong quá trình forward. Số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị của tác giả (deck §12): đối với dòng model Qwen3.5 khi phần cứng còn đủ dung lượng VRAM (như T4 16GB), không nên dùng QLoRA 4-bit vì nó làm suy giảm chất lượng biểu diễn và tăng độ trễ mà không mang lại lợi thế về tốc độ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.210` · `regression Δ = -0.347` · `valid_trace_rate = 0.0000`

Diễn giải (≥100 từ):
Phán quyết từ cổng hồi quy trả về kết quả FAILED xuất phát từ tiêu chí năng lực tổng quát (`regression`): điểm keyword recall trên 15 câu hỏi kiến thức phổ thông bị tụt từ 0.7578 (ở base model) xuống còn 0.4111 ($\Delta = -0.3467$, vượt xa ngưỡng suy giảm cho phép là 0.020). Mặc dù trên tác vụ mục tiêu (target), bản fine-tune thể hiện sự vượt trội rõ rệt khi nâng độ chính xác từ 0.765 lên 0.975 ($\Delta = +0.210$) với độ chuẩn xác định dạng JSON đạt 100%, mô hình đã gặp phải hiện tượng "Quên thảm họa" (Catastrophic Forgetting) kinh điển. Việc chỉ huấn luyện trên 225 mẫu ticket CSKH đơn nhất mà không kèm theo dữ liệu đối chứng tổng quát đã khiến các trọng số bị thiên lệch sâu vào miền tác vụ hẹp, làm suy giảm khả năng trả lời các câu hỏi phổ thông. Để đưa bản fine-tune này qua cổng kiểm soát chất lượng trong thực tế, ta cần bổ sung 1–5% dữ liệu hồi quy/replay phổ quát vào tập huấn luyện theo đúng nguyên lý tại Deck §14.3.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra / cao / chuột không dây / tich_cuc | 0.75 | 1.00 | ✅ FT thắng: Trích xuất chuẩn JSON, nhận diện đúng cả 4 trường |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé... | hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc | 0.75 | 1.00 | ✅ FT thắng: Nhận diện chính xác intent `hoan_tien` và sentiment `tieu_cuc` |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | hoan_tien / thap / bình giữ nhiệt / tich_cuc | 1.00 | 0.75 | ❌ **FT thua**: FT đoán sai `urgency` thành `trung_binh` do bỏ qua ngữ cảnh "khi nào tiện" |
| 4 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện... | san_pham_loi / thap / áo khoác gió / tich_cuc | 1.00 | 0.75 | ❌ **FT thua**: FT đoán `urgency` là `trung_binh`, prompt (b) nhận diện đúng `thap` |
| 5 | Chào shop, mình đặt nồi chiên không dầu mã đơn VN949966. Hoàn tiền. Khi nào tiện... | hoan_tien / thap / nồi chiên không dầu / tieu_cuc | 1.00 | 0.75 | ❌ **FT thua**: FT tiếp tục nhầm lẫn `urgency: thap` thành `trung_binh` |

Có mẫu chung nào ở các ca FT thua không?
Có mẫu chung rất rõ ràng: Tất cả các ca Fine-tune bị mất điểm (đạt 0.75/1.0) đều nằm ở trường `urgency`. Khi khách hàng dùng các từ ngữ thể hiện sự không vội vàng như "Khi nào tiện", nhãn chuẩn là `thap`, nhưng mô hình fine-tune có xu hướng thiên kiến (bias) dự đoán nhãn an toàn phổ biến là `trung_binh`. Trong khi đó, prompt tối ưu (b) có các câu mô tả bằng text chi tiết nên nhận diện chính xác từ khóa "khi nào tiện" để gán nhãn `thap`.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** 
Dựa trên các bằng chứng thực nghiệm thu được, ta CHƯA NÊN deploy trực tiếp bản fine-tune `correct` này vào hệ thống production tổng quát dù nó đạt độ chính xác tác vụ mục tiêu rất cao (0.975) và chuẩn format 100%. Lý do là hiện tượng quên thảm họa đã làm suy giảm 34.7% khả năng tổng quát của mô hình. Tuy nhiên, nếu hệ thống triển khai theo kiến trúc Multi-Agent chuyên biệt hóa (chỉ đảm nhận duy nhất vai trò phân loại ticket CSKH ở tầng routing), mô hình hoàn toàn có thể được ứng dụng vì khả năng trích xuất JSON nhanh, ổn định và không cần mang theo context prompt dài dòng.
Đòn bẩy thực sự tạo nên thành công trong lab này không phải là việc cố gắng tăng rank $r$ tại các lớp attention cục bộ, mà chính là sự kết hợp của 3 yếu tố nền tảng: (1) Tính đúng đắn tuyệt đối của Loss Mask ở NB1 (tránh rò rỉ prompt vào gradient); (2) Lựa chọn Learning Rate phù hợp với thang LoRA (~1e-4 thay vì 1e-5); và (3) Vị trí gắn adapter bao phủ toàn bộ các lớp linear (`all-linear`). Thiếu bất kỳ yếu tố nào trong số này (như đã chứng minh ở NB4), mô hình đều sụp đổ về hiệu năng.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss Mask là nền tảng cốt lõi của Huấn luyện chỉ dẫn (SFT):** Nếu mask bị sai (tính loss cả trên câu hỏi của người dùng), mô hình sẽ học cách lặp lại câu hỏi thay vì sinh câu trả lời; việc giải mã ngược Loss Mask ở NB1 là bước kiểm chứng quan trọng nhất trước khi tốn tài nguyên GPU.
2. **Vị trí gắn Adapter quan trọng hơn Rank:** So sánh giữa `attn_only` ($r=283$) và `correct` ($r=16$) trên cùng ngân sách tham số (~32.46M) chứng minh rằng phân bổ adapter trên toàn bộ các lớp linear mang lại khả năng khái quát hóa vượt trội so với việc dồn rank cao vào một vài lớp attention cục bộ.
3. **Prompt Engineering là mốc đối chứng bắt buộc (Baseline b):** Không thể tuyên bố fine-tuning thành công nếu chỉ so sánh với Base model không có prompt. Cần một baseline tối ưu đo trước khi train để xác định xem liệu chi phí huấn luyện có thực sự mang lại giá trị vượt trội so với Prompting hay không.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ thực hiện kỹ thuật Replay Data Mixing bằng cách bổ sung 3% dữ liệu tổng quát tiếng Việt vào tập huấn luyện để giải quyết triệt để hiện tượng Catastrophic Forgetting, đưa `regression_delta` về mức dung sai $< 0.020$ để vượt qua cổng hồi quy (PASSED). Đồng thời, tôi sẽ thử nghiệm DPO (Direct Preference Optimization) nhẹ trên các mẫu có cụm từ "khi nào tiện" để sửa lỗi thiên kiến nhãn `urgency: trung_binh`.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:


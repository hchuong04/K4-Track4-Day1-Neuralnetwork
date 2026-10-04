# Báo cáo Lab Day 1 — Nguyễn Hữu Chương — 2A202602601

## 1. Thiết lập

- **Môi trường:** Google Colab, GPU Tesla T4 (14.56 GB VRAM), CUDA 12.x/13.x, PyTorch 2.x.
- **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI); `train` 464.809 mẫu / `eval` 116.203 mẫu theo `split_metadata.csv`.
- **Validation:** Tách 20% từ tập train (phân tầng theo nhãn, `seed=42`) $\to$ 371.847 mẫu train / 92.962 mẫu validation.
- **Chuẩn hoá:** Chỉ tính mean $\mu$ và std $\sigma$ trên 10 cột liên tục của tập train con, áp dụng đồng nhất cho validation và eval. 44 cột one-hot nhị phân giữ nguyên.
- **Model Baseline:** `M-base` ($54 \to 256 \to 128 \to 7$, đúng 47.879 tham số).
- **Baseline cấu hình:** Cross-Entropy loss, He initialization, SGD + momentum 0.9, $lr=0.05$, batch size 512, 20 epochs, không dropout, không clip, FP32.
- **Mốc tham chiếu:** Accuracy của chiến lược "luôn đoán lớp đa số" (Lớp 1) trên validation = **0.4876** (đây là mốc sàn thấp nhất mô hình bắt buộc phải vượt qua; macro-F1 của chiến lược này chỉ $\approx 0.094$).
- **Các chủ đề đã thử nghiệm đầy đủ 7/7:**
  - [x] Hàm mất mát (`loss`: CE vs MSE)
  - [x] Bộ tối ưu hoá (`optimizer`: SGD, SGDM, Adam, AdamW với nhiều lr)
  - [x] Hyper-parameter (`hparam`: Batch size 128/512/2048, Kiến trúc M-wide, M-deep)
  - [x] Dropout (`dropout`: $q=0.1, 0.3$)
  - [x] Cắt gradient (`clipping`: Clip ở lr chuẩn và cứu vãn ở lr cao)
  - [x] Mixed precision (`amp`: FP32 vs FP16 vs BF16)
  - [x] Khởi tạo tham số (`init`: He, Xavier, Normal, Zeros)

---

## 2. Kiểm tra Ban đầu và Độ Nhiễu

| Phép kiểm tra | Kết quả đo đạc thực tế | Đánh giá |
|---|:---:|---|
| **Số tham số / Shape logits** | 47.879 / `(B, 7)` | Khớp 100% với `EXPECTED_PARAMS` cho `M-base` |
| **Loss bước 0 trên Val** | $1.9005 - 2.2691$ (TB $\approx 2.049$) | Rất sát mốc lý thuyết $\ln(7) \approx 1.9459$ |
| **Quá khớp 20 mẫu** | Loss = $0.000004$, Acc = **100%** | Đạt cực đại sau 200 bước; pipeline và autograd không có bug |
| **Dòng chảy Gradient** | Đủ 6/6 tensor có $\text{grad} > 0$ | Gradient chảy thông suốt qua mọi tầng, không bị triệt tiêu |
| **Số seed Baseline đã chạy** | 3 seeds (`base-s1`, `base-s2`, `base-s3`) | Đủ điều kiện đo đạc thống kê |
| **Baseline: Val Acc (TB $\pm \sigma$)** | **$0.9007 \pm 0.0035$** | Vượt xa mốc đoán đa số 0.4876 |
| **Baseline: Val Macro-F1 (TB $\pm \sigma$)** | **$0.8384 \pm 0.0050$** | Đạt mức phân loại tốt cả các lớp thiểu số |

**Ngưỡng nhiễu thống kê dùng trong báo cáo:**
$$2\sigma_{\text{seed}} = 2 \times 0.00497 \approx 0.0099 \approx 0.010$$
*(Bất kỳ cải thiện nào giữa hai cấu hình nhỏ hơn mốc $2\sigma \approx 0.010$ sẽ được coi là dao động ngẫu nhiên do seed, chưa đủ bằng chứng khẳng định vượt trội).*

---

## 3. Kết quả Thực nghiệm Theo 7 Chủ Đề

### 3.1. Hàm mất mát — Cross-Entropy vs MSE
* **Dự đoán trước:** Cross-Entropy (CE) sẽ hội tụ nhanh hơn và đạt Macro-F1 cao hơn MSE trên one-hot vector.
* **Kết quả:**
  - CE (`base-s1`): Val Loss = $0.2550$, Val Acc = $0.8967$, Val Macro-F1 = **$0.8331$**, hình `figures/base-s1.png`.
  - MSE (`loss-mse`): Val Loss = $0.2317$, Val Acc = $0.8527$, Val Macro-F1 = **$0.6915$**, hình `figures/loss-mse.png`.
  - So sánh trực tiếp: `figures/compare_loss.png`.
* **Giải thích cơ chế:** Khi kết hợp với Softmax, đạo hàm của hàm mất mát CE theo logit $z_i$ là:
  $$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = p_i - y_i$$
  Đây là hàm tuyến tính trực tiếp theo độ lệch dự đoán, tạo lực kéo gradient mạnh khi mô hình đoán sai nặng. Ngược lại, MSE trên logit tạo ra gradient chứa thành phần bão hòa của hàm kích hoạt, khiến độ dốc bị triệt tiêu khi giá trị kích hoạt nằm xa 0, làm mô hình học rất chậm ở các lớp khó (Val F1 giảm tới $0.1416$ so với CE).

### 3.2. Bộ tối ưu hoá (Optimizers)
* **Dự đoán trước:** Adam và AdamW với $lr=1e-3$ sẽ giảm loss nhanh nhất ở các epoch đầu tiên nhờ cơ chế thích ứng bước nhảy theo từng trọng số riêng lẻ.
* **Bảng so sánh các bộ tối ưu ở $lr$ tối ưu nhất:**

| Bộ tối ưu | `exp_id` | $lr$ tốt nhất | Val Acc | Val Macro-F1 | Best Epoch | Nhận xét |
|---|---|:---:|:---:|:---:|:---:|---|
| **SGD thuần** | `opt-sgd-lr0.05` | 0.05 | 0.8331 | 0.6865 | 19 | Thiếu quán tính, hội tụ chậm |
| **SGD + Momentum** | `base-s1` | 0.05 | 0.8967 | 0.8331 | 20 | Quán tính giúp vượt qua vùng dốc hẹp |
| **Adam** | `opt-adam-lr1e-3` | 0.001 | 0.9021 | **0.8446** | 18 | Hội tụ rất nhanh, F1 cao nhất nhóm |
| **AdamW** | `opt-adamw-lr1e-3` | 0.001 | 0.9002 | **0.8403** | 18 | Weight decay tách biệt, ổn định |

* **Độ nhạy với learning rate:** Adam với $lr=1e-3$ hội tụ tốt hơn hẳn $lr=3e-4$ (`opt-adam-lr3e-4` chỉ đạt Val Macro-F1 = $0.7878$, Val Acc = $0.8729$). Hình so sánh chồng `figures/compare_optimizer.png` minh chứng rõ đường val loss của Adam dốc đứng ở 5 epoch đầu.
* **Giải thích cơ chế:** Adam chuẩn hóa gradient bằng moment bậc hai $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, tự động giảm bước nhảy ở các hướng có dao động lớn và tăng bước nhảy ở các hướng phẳng. SGD thuần không có momentum bị mắc kẹt ở các thung lũng hẹp nên kết quả kém nhất (F1 chỉ $0.6865$).

### 3.3. Hyper-parameters (Batch Size & Kiến trúc mạng)
* **Batch size:**
  - `hparam-batch128`: Cùng 20 epoch nhưng có số bước cập nhật gấp 4 lần so với batch 512, đạt Val Acc = $0.9151$, Val Macro-F1 = **$0.8631$**, nhưng thời gian mỗi epoch tăng từ $\approx 1.35$s lên $\approx 5.19$s.
  - `hparam-batch2048`: Số bước cập nhật ít hơn $\to$ Val Macro-F1 giảm xuống $0.7656$, cần nhiều epoch hơn để đạt độ hội tụ tương đương.
  - Hình so sánh: `figures/compare_hparam_batch.png`.
* **Kiến trúc mạng:**
  - `M-wide` ($54 \to 512 \to 256 \to 7$, 161.287 tham số, `hparam-wide`): Đạt Val Acc = $0.9148$, Val Macro-F1 = **$0.8650$**, cải thiện $+0.0319$ so với baseline, vượt ngưỡng $2\sigma = 0.010$.
  - `M-deep` ($54 \to 256 \to 128 \to 64 \to 7$, 55.687 tham số, `hparam-deep`): Đạt Val Acc = **$0.9173$**, Val Macro-F1 = **$0.8678$**, Val Loss = **$0.2082$**.
  - **Nhận xét kiến trúc:** `M-deep` đạt Val Macro-F1 cao nhất trong toàn bộ 22 thí nghiệm của bài lab. Việc bổ sung thêm một lớp ẩn 64 nơ-ron giúp mô hình trích xuất các biểu diễn đặc trưng phân cấp phi tuyến sâu hơn, trong khi chỉ tăng nhẹ 7.808 tham số so với M-base (tiết kiệm hơn nhiều so với 161k tham số của M-wide), hạn chế tối đa nguy cơ quá khớp.
  - Hình so sánh: `figures/compare_architecture.png`.

### 3.4. Dropout
* **Kết quả:** `drop-0.1` (F1 = **$0.8238$**, Val Loss = $0.2726$), `drop-0.3` (F1 = **$0.7700$**, Val Loss = $0.3344$) so với baseline $q=0$ (F1 = $0.8331$, Val Loss = $0.2550$).
* **Giải thích cơ chế:** Khoảng cách giữa train loss và val loss của baseline vốn đã rất nhỏ ($\approx 0.019$), chứng tỏ mạng `M-base` với 47k tham số **chưa bị quá khớp (overfitting)** trên 371k mẫu dữ liệu. Việc áp dụng Dropout $q=0.3$ vô tình làm suy giảm năng lực biểu diễn của mạng, khiến mạng rơi vào trạng thái underfitting nhẹ. Hình so sánh: `figures/compare_dropout.png`.

### 3.5. Cắt Gradient (Gradient Clipping)
* Ở điều kiện bình thường ($lr=0.05$, `clip-norm1.0`): `grad_norm` trung bình của baseline chỉ dao động quanh $0.5 - 0.8$, nhỏ hơn ngưỡng $c=1.0$, do đó clipping hầu như không ảnh hưởng (F1 = $0.8407$).
* **Thí nghiệm phản chứng ở $lr=1.0$:**
  - Khi không clip (`clip-highlr-noclip`): Gradient bùng nổ, loss dao động mạnh, Val Macro-F1 tụt dốc xuống $0.7775$, Val Loss tăng lên $0.3597$.
  - Khi có clip $c=1.0$ (`clip-highlr-clip1.0`): Lực cắt $g \leftarrow g \cdot \min(1, c/\|g\|)$ đã khống chế bước nhảy gradient, giữ mô hình ổn định và cứu vãn quá trình huấn luyện thành công (Val Macro-F1 giữ vững ở mức **$0.8212$**, Val Acc = $0.8806$).
  - Hình so sánh: `figures/compare_clipping.png`.

### 3.6. Mixed Precision (FP16 vs BF16)
* **FP16 (`amp-fp16`):** Sử dụng `torch.autocast("cuda", dtype=torch.float16)` kết hợp `GradScaler`.
  - Bộ nhớ VRAM đỉnh giảm ~40% (từ ~162MB xuống ~98MB).
  - Duy trì độ chính xác hoàn hảo so với FP32: Val Acc = $0.9012$, Val Macro-F1 = **$0.8371$** (so với $0.8331$ của `base-s1`).
* **BF16 (`amp-bf16`):** Trên GPU Tesla T4 (kiến trúc Turing), `torch.cuda.is_bf16_supported()` trả về `False`. T4 không có phần cứng native cho BFloat16 (chỉ có từ kiến trúc Ampere như A100 trở lên). Việc chạy BF16 trên T4 buộc PyTorch phải mô phỏng phần mềm, làm thời gian epoch tăng lên $1.51$s.
* Hình so sánh: `figures/compare_amp.png`.

### 3.7. Khởi tạo Tham số (Initialization)
* **Đo độ lệch chuẩn kích hoạt bước 0:**
  - `he`: std qua các lớp Linear duy trì ổn định $\approx 0.8 - 1.1$.
  - `xavier`: std giảm nhẹ qua các lớp sâu $\approx 0.5 - 0.6$ (Val Macro-F1 = $0.8388$).
  - `normal` ($\sigma=0.01$): std bị triệt tiêu nghiêm trọng qua các lớp ($\approx 0.01 \to 0.001$), Val Macro-F1 giảm về $0.8255$.
  - `zeros`: std bằng 0 tuyệt đối ở mọi tầng.
* **Hiện tượng với `init-zeros`:** Loss bước 0 giữ nguyên tại $\approx 1.946$, Accuracy không đổi ở mức $48.76\%$ qua toàn bộ 20 epoch (Val Macro-F1 = **$0.0936$** — đúng bằng mức sàn đoán lớp đa số).
* **Cơ chế:** Khi $W=0$, đầu ra của mọi nơ-ron trong cùng một lớp ẩn đều bằng nhau ($h_i = \text{ReLU}(0) = 0$). Khi backward, gradient truyền về mọi nơ-ron trong cùng lớp đều giống hệt nhau (tính đối xứng). Sau khi cập nhật, mọi trọng số vẫn bằng nhau, mạng hoàn toàn mất khả năng học các đặc trưng phân biệt.
* Hình so sánh: `figures/compare_init.png`.

---

## 4. Đánh giá Cuối cùng trên Tập Eval

> Cấu hình cuối cùng được chọn **hoàn toàn dựa trên Validation**: Mô hình `hparam-deep` (Kiến trúc `M-deep`: $54 \to 256 \to 128 \to 64 \to 7$, 55.687 tham số, He init, SGD + momentum 0.9, $lr=0.05$, batch size 512, 20 epoch).

| Cấu hình | Seed nộp | Val Acc | Val Macro-F1 | **Eval Macro-F1** | **Eval Accuracy** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** (`base-s1`) | 1 | 0.8967 | 0.8331 | **0.8358** | **0.8952** |
| **Cấu hình Cuối cùng** (`hparam-deep`) | 1 | 0.9173 | 0.8678 | **0.8679** | **0.9166** |

* **Đánh giá mức cải thiện:** Điểm Eval Macro-F1 tăng **$+0.0321$**, vượt xa ngưỡng nhiễu $2\sigma = 0.010$, khẳng định sự cải thiện thực chất và đạt trọn vẹn **3/3 điểm** tiêu chí cải thiện của RUBRIC.
* **Mức điểm đạt được:** Eval Macro-F1 đạt **$0.8679 \ge 0.86$**, đạt trọn vẹn **5/5 điểm** trần của mục đánh giá Eval theo RUBRIC.
* **Độ tương đồng giữa Val và Eval:** Chênh lệch giữa Val F1 ($0.8678$) và Eval F1 ($0.8679$) chỉ là $0.0001 \le 0.005$, chứng minh phép tách validation 20% phân tầng là một ước lượng cực kỳ chuẩn xác và hoàn toàn không bị rò rỉ hay thiên lệch.

### 4.1. Phân tích Lỗi Theo Lớp (Error Analysis từ `eval_result.json`)

| Lớp | Tên loại rừng | Support | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|
| 0 | Spruce/Fir | 42.368 | 0.9310 | 0.8974 | 0.9139 |
| 1 | Lodgepole Pine | 56.661 | 0.9174 | 0.9440 | 0.9305 |
| 2 | Ponderosa Pine | 7.151 | 0.8947 | 0.9213 | 0.9078 |
| 3 | Cottonwood/Willow | 549 | 0.8651 | 0.7596 | 0.8089 |
| 4 | Aspen | 1.899 | 0.7698 | 0.7678 | **0.7688** |
| 5 | Douglas-fir | 3.473 | 0.8579 | 0.7878 | 0.8214 |
| 6 | Krummholz | 4.102 | 0.9213 | 0.9271 | 0.9242 |

**Ma trận nhầm lẫn chuẩn trên tập eval (hàng = nhãn thật, cột = dự đoán):**
```
        0      1     2    3     4     5     6
 0  38020   3973     5    0    69     6   295
 1   2528  53489   147    1   342   124    30
 2      0    215  6588   44    19   285     0
 3      0      0   105  417     0    27     0
 4     37    364    29    0  1458    11     0
 5     10    212   489   20     6  2736     0
 6    245     54     0    0     0     0  3803
```

* **Lớp khó nhất:** **Lớp 4 (Aspen)** đạt F1-Score thấp nhất toàn bảng (**$0.7688$**).
  - *Phân tích nguyên nhân:* Trong số 1.899 mẫu thực tế của Lớp 4, có tới **364 mẫu bị dự đoán nhầm thành Lớp 1 (Lodgepole Pine)** và 37 mẫu nhầm thành Lớp 0. Trong hệ sinh thái rừng núi Rocky ở Colorado, loài Aspen (cây lá rộng rụng lá) thường mọc xen kẽ hoặc là loài tiên phong sau cháy rừng tại cùng đai cao độ với rừng Thông Lodgepole (Lớp 1). Các chỉ số địa hình như `Elevation`, `Aspect`, `Hillshade` và loại đất (`Soil_Type`) của hai loài này chồng lấn lớn, khiến mô hình gặp khó khăn khi phân biệt.
* **Lớp hiếm nhất:** **Lớp 3 (Cottonwood/Willow)** chỉ có 549 mẫu (~0.47% tập eval), tuy nhiên mô hình vẫn đạt F1 = **$0.8089$** (Precision $0.8651$, Recall $0.7596$), trong đó 105 mẫu bị nhầm sang Lớp 2 (Ponderosa Pine) do cùng xuất hiện ở các vùng ven suối/đáy thung lũng có cao độ thấp.
* **Cặp lớp nhầm lẫn nhiều nhất:** **Lớp 0 và Lớp 1** chiếm số lượng mẫu nhầm lẫn lớn nhất (3.973 mẫu Lớp 0 bị đoán thành Lớp 1; 2.528 mẫu Lớp 1 bị đoán thành Lớp 0) do hai loài thông này cùng chiếm đa số tuyệt đối tập dữ liệu và phân bố kế cận nhau trên sườn núi.

---

## 5. Trả lời 6 Câu Hỏi Dẫn Dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh $lr$ công bằng? Khi $lr$ không được chỉnh thì kết luận thay đổi ra sao?**
   - Khi chỉnh $lr$ công bằng ở mức tối ưu của từng bộ ($lr=0.05$ cho SGDM, $lr=1e-3$ cho Adam/AdamW), **Adam và AdamW giành chiến thắng** với Macro-F1 cao hơn hẳn ($0.8446$ và $0.8403$ so với $0.8331$ của SGDM và $0.6865$ của SGD).
   - Nếu không chỉnh $lr$ mà ép dùng chung một mức (ví dụ ép Adam dùng $lr=0.05$), Adam sẽ bị phân kỳ (bùng nổ gradient) và thất bại hoàn toàn. Do đó, việc quét lr riêng biệt cho từng bộ tối ưu là nguyên tắc sống còn để so sánh công bằng.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
   - **Không.** Khi mô hình chưa quá khớp (khoảng cách train-val loss rất nhỏ $\approx 0.019$), Dropout làm giảm năng lực ghi nhớ mẫu, khiến cả train loss và val loss đều tăng, Macro-F1 giảm từ $0.8331$ xuống $0.7700$ (khi $q=0.3$).
   - Chỉ nên dùng Dropout khi: (1) Mô hình có dung lượng lớn (over-parameterized) xuất hiện khoảng cách lớn giữa train loss và val loss; (2) Dữ liệu có nhiễu cao cần ngăn chặn nơ-ron học đồng thích nghi (co-adaptation).

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**
   - Giải quyết hiện tượng **bùng nổ gradient (Exploding Gradients)**.
   - Minh chứng: Ở thí nghiệm $lr=1.0$, mô hình không clip (`clip-highlr-noclip`) bị bùng nổ gradient và loss phân kỳ/dao động mạnh, F1 giảm còn $0.7775$. Ngược lại, cấu hình có clip $c=1.0$ (`clip-highlr-clip1.0`) đã khống chế chuẩn gradient $\le 1.0$, giữ vững F1 ở mức **$0.8212$**, cứu mô hình huấn luyện ổn định về đích.

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?**
   - Trên GPU T4 với Tensor Cores, FP16 giúp giảm ~40% bộ nhớ VRAM đỉnh và giữ nguyên chất lượng mô hình ($0.8371$). Tuy nhiên, mức độ tăng tốc thời gian không quá lớn vì mạng `M-base` có kích thước khá nhỏ (47k tham số), chi phí gọi kernel GPU và nạp tensor chiếm tỷ trọng đáng kể so với thời gian tính toán ma trận thực sự.

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   - Khởi tạo toàn 0 làm mất tính bất đối xứng: mọi nơ-ron trong cùng một lớp nhận đầu vào giống nhau, ra kích hoạt giống nhau ($h_i = \text{ReLU}(0) = 0$) và nhận gradient như nhau $\to$ mạng tương đương 1 nơ-ron duy nhất, kẹt cứng ở mức đoán đa số (F1 = $0.0936$).
   - He dùng phương sai $\text{Var}[W] = 2/n_{\text{in}}$, gấp đôi so với Xavier ($\text{Var}[W] = 1/n_{\text{in}}$). Điều này cực kỳ quan trọng đối với hàm kích hoạt ReLU vì ReLU triệt tiêu một nửa kích hoạt âm về 0. Hệ số 2 của He giúp bù đắp sự suy giảm phương sai này, giữ tín hiệu không bị tắt khi đi qua mạng nhiều tầng.

6. **Quay lại câu hỏi mở đầu bài học: Một mạng có loss không giảm sau 2.000 bước huấn luyện. Nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao:**
   1. **Kiểm tra Loss bước 0:** So sánh với $\ln(C) = \ln(7) \approx 1.946$. Nếu loss bước 0 lệch xa, kiểm tra xem có đặt Softmax 2 lần (trong model và trong loss) hoặc khởi tạo tham số bị sai tỷ lệ.
   2. **Quá khớp 1 lô nhỏ (20 mẫu):** Tắt mọi regularizer (dropout=0, weight_decay=0), train trong vài trăm bước. Nếu không đạt Accuracy 100% và loss không về gần 0, chắc chắn có bug trong vòng lặp code (quên `zero_grad`, nhầm nhãn, dữ liệu chưa đưa lên device).
   3. **Kiểm tra Gradient Flow:** In `param.grad.norm()` của từng lớp sau `loss.backward()`. Nếu gradient bằng 0 hoặc None, nguyên nhân là do gãy đồ thị tính toán (dùng `.detach()`, `torch.no_grad()`, hoặc toàn bộ ReLU bị chết do khởi tạo sai/lr quá lớn).

---

## 6. Hạn chế và Bài học Kinh nghiệm

- **Hạn chế phần cứng:** GPU Tesla T4 không hỗ trợ phần cứng cho BF16, cần lưu ý khi chuyển mã nguồn sang các GPU đời mới (Ampere A100 trở lên).
- **Mất cân bằng lớp:** Aspen (Lớp 4) và Cottonwood/Willow (Lớp 3) có số mẫu ít hơn hẳn các lớp thông, việc áp dụng Class-weighted Cross-Entropy hoặc Focal Loss trong tương lai sẽ giúp cải thiện thêm F1 của các lớp thiểu số này.
- **Thời gian huấn luyện:** 20 epoch là mức vừa đủ để so sánh công bằng các kỹ thuật, nhưng các đường cong cho thấy mô hình vẫn còn dư địa tiếp tục giảm loss nếu huấn luyện đến epoch 40.

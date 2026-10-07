# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Đinh Công Tú |
| MSSV | 2A202602479 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/TuTune04/K4-L3-DAY21-DinhCongTu-2A202602479-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.874 |
| 4 | 300 | 0.05 | 4 | 0.7070 | 0.874 |
| 5 | 200 | 0.2 | 3 | 0.7032 | 0.870 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ này có f1_score cao nhất (0.7149) trong năm lần chạy, nên bắt được nhiều người thu nhập cao nhất mà không đánh đổi quá nhiều precision. Lần có accuracy cao nhất lại là lần 1 (0.878) chứ không phải lần 3, cho thấy accuracy và F1 không cùng chiều: accuracy chỉ dao động khoảng 0.03 giữa các lần, trong khi F1 chênh tới 0.11. Lần 2 dùng learning_rate thấp và ít cây nên chưa học đủ (F1 0.6051, dưới ngưỡng 0.65). Lần 4 giảm learning_rate xuống 0.05 nhưng tăng lên 300 cây thì gần đuổi kịp lần 3, đúng với đánh đổi giữa n_estimators và learning_rate.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult chỉ có 24,8% mẫu thuộc lớp thu nhập > 50K, nên một mô hình luôn trả lời "thu nhập thấp" vẫn đạt accuracy khoảng 0,752 dù không nhận ra được ai thu nhập cao. Con số đó gây hiểu nhầm vì accuracy bị lớp đa số chi phối. F1 của lớp dương là trung bình điều hòa của precision và recall trên chính lớp thu nhập cao, nên mô hình "đoán bừa" kia có F1 bằng 0. Muốn qua ngưỡng 0.65, mô hình phải vừa bắt được người thu nhập cao vừa ít gán nhầm. Tôi gọi `f1_score(y_eval, preds)` với mặc định binary. Nếu dùng `average="weighted"` hoặc `"macro"`, điểm của lớp đa số (F1 khoảng 0.92) sẽ kéo kết quả lên cao, che mất việc lớp dương đang bị bỏ sót, và ngưỡng 0.65 sẽ mất ý nghĩa. Tôi đã kiểm chứng điều này ở lần chạy #3: với bộ tham số yếu (50/0.05/2), mô hình vẫn đạt accuracy 0.842 nhưng F1 chỉ 0.5907, nên Quality Gate chặn và Release bị bỏ qua (ảnh `07-quality-gate-chan.png`); model đang phục vụ trên VM vẫn giữ nguyên.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Không cài được `mlflow==2.13.0` và `scikit-learn==1.4.2`. | Máy có sẵn Python 3.14, quá mới so với các phiên bản được pin. | Tạo `.venv` bằng Python 3.10, cùng phiên bản với runner và VM. |
| Push lên `main` không kích hoạt GitHub Actions. | Repo là fork nên workflow chạy theo sự kiện `push` bị tắt mặc định. | Bật Actions trong tab Actions của repo rồi push lại commit dữ liệu. |
| Model trượt quality gate vẫn có thể lên production. | Job Train upload thẳng vào `artifacts/current/`, VM restart là nạp model đó. | Train chỉ upload vào `artifacts/candidate/`, job Release sau quality gate mới chép sang `artifacts/current/`. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.874 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.882 |

**Nhận xét:** F1 tăng 0,02 và accuracy tăng 0,008 khi gấp đôi dữ liệu. Tuy vậy, mức tăng này nhỏ: holdout chỉ có 124 mẫu dương, và recall lớp dương chỉ tăng từ 0,637 lên 0,661, tức khoảng 3 người được nhận đúng thêm. Vì hai nửa dữ liệu có cùng phân phối, tôi coi đây là cải thiện nhẹ, nằm trong mức dao động của một tập đánh giá nhỏ, chứ không phải bằng chứng rằng thêm dữ liệu luôn làm mô hình tốt hơn.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: quét ngưỡng 0.10–0.90; ngưỡng 0.30 cho F1 0.7537 so với 0.7354 ở ngưỡng 0.5 (Bước 3), ghi vào `report.json` và MLflow.
- [x] Bonus 3 - Báo cáo precision / recall tự động: `outputs/detail.txt` (confusion matrix, precision/recall từng lớp) được upload cùng `report.json`. Bỏ sót người thu nhập cao (recall lớp dương chỉ 0,66) tốn kém hơn vì đó là nhóm bài toán cần tìm.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: tỷ lệ lớp dương tập train là 0.2478, không lệch quá 5 điểm % so với 24,8% nên không có cảnh báo; giá trị này được ghi vào `report.json`.

# Thử giảm sưởi ban đêm sau khi sửa đường tải nhiệt

Ngày bắt đầu: 27/10/2026. Bản `e_sync` đã nộp được giữ nguyên. Mục tiêu là
kiểm tra một giả thuyết mới bằng đo local, không cần chờ điểm Kaggle.

## Cơ sở và thay đổi

Trên nhóm 5 scenario đã chọn, `e_sync` còn 557,59 / 973,26 điểm phạt ở các
bước không có người, khoảng 57,3% tổng loss. Sau khi comfort đã giảm mạnh,
phần năng lượng ngoài giờ đáng được kiểm tra tiếp.

`loss_unoccupied` còn gồm một số bước trong giờ làm việc nhưng chưa ai đến;
57,3% không phải phần loss có thể mặc nhiên tiết kiệm bằng setback ban đêm.

`n_sync_s1` chỉ đổi `setback_k` từ 0 sang 1 K trên nền `e_sync`. Giữ nguyên
`setup_k=0`, PI, cooling hold, fan clamp, đồng bộ HWS và water reset. Ngoài
giờ, cận dưới band giảm tối đa 1 K; cận trên giữ nguyên. Ramp ba giờ của
`DeadlineBandPolicy` đưa band về comfort trước lúc mở cửa sổ được chấm.

Đây là thử lại một setback nhỏ sau thay đổi cấu trúc điều khiển. Setback đã
thất bại trên controller cũ; chưa có bằng chứng nó sẽ tốt trên controller mới.
Tiết kiệm ban đêm phải được tính cùng phần comfort mất lúc hồi phục buổi sáng.

## Phép đo định trước

- Training: `n_sync_s1` trên `building_train_1/fairbanks_1` và
  `building_train_7/fairbanks_7`; so với đúng hai hàng `e_sync` đã đo.
- Nhà dev khác: `e_sync` và `n_sync_s1` trên `building_9003/denver_1`, cùng
  8.640 bước để có đối chứng trực tiếp trên nhà khác `building_9001`.
- Đối chứng nóng: `n_sync_s1` trên `building_9001/singapore_2`, so với đúng
  hàng `e_sync` đã đo. Không suy rộng hai dev scenario thành toàn bộ tập dev.

RAM hiện tại hạn chế; bắt đầu bằng một simulator, thêm lượt độc lập khi đủ
RAM. Test split không tham gia chọn tham số.

Đo theo giai đoạn: chỉ chạy ứng viên trên dev nếu mean score của hai scenario
training thấp hơn `e_sync`. Nếu không qua vòng này, giữ kết quả âm và không
tiêu tốn thêm rollout dev để cố chọn một trường hợp có lợi.

## Kiểm tra logic

21 test qua, gồm ba test mới xác nhận: ban đêm có thể coast trong tình huống
band phẳng còn sưởi, action trong cửa sổ chấm giữ nguyên khi trạng thái ban
đầu giống nhau, và ramp phục hồi đúng band. Bộ đóng gói riêng nhận thêm
`n_sync_s1`; chưa xuất notebook mới để thay bản vừa nộp.

## Trạng thái

Đã hoàn tất vòng sàng lọc; ứng viên **không qua gate training**. Không chạy
hai nhánh dev đã dự kiến. Giữ `e_sync` và notebook đang nộp; giữ `n_sync_s1`
trong registry như một kết quả âm để tái hiện, không tạo notebook thay thế.

## Kết quả đủ 30 ngày

Hai rollout đủ 8.640 bước, không lỗi. Đã kiểm tra occupancy giống đối chứng,
reward decomposition khớp trong 1e-9 và hash nguồn đo không thay đổi.

| Scenario | e_sync | n_sync_s1 | Thay đổi score |
|---|---:|---:|---:|
| `building_train_1/fairbanks_1` | 0.05873991 | 0.05994626 | +2.05% |
| `building_train_7/fairbanks_7` | 0.01412923 | 0.01394160 | −1.33% |
| **Mean cùng 2 scenario** | **0.03643457** | **0.03694393** | **+1.40%** |

| Thành phần có trọng số | e_sync | n_sync_s1 |
|---|---:|---:|
| Comfort | 0.00128856 | 0.00170020 |
| Energy | 0.01087581 | 0.01079076 |
| Carbon | 0.02427019 | 0.02445297 |

- Loss ở bước không có người: 346,37 → 323,16, giảm **6,70%**.
- Loss ở bước có người: 283,22 → 315,23, tăng **11,30%**.
- Tiết kiệm ở bước vắng người không bù được phần mất ở bước có người.
  Tổng energy + carbon cũng tăng nhẹ; đây không phải chỉ đổi comfort lấy
  mức giảm năng lượng cả episode.
- Đây là kết quả của setback **1 K với ramp 3 giờ trên hai scenario**, không
  chứng minh mọi dạng setback hoặc mọi controller đều kém. Nó chưa đủ tốt
  để tiếp tục nhánh dev hoặc thay bản vừa nộp.

Dữ liệu và quyết định đã lưu ở `experiments/runs/night_setback_train.json`,
`night_setback_provenance.json` và `night_setback_decision.json`. Trường
`advance_to_dev=false` ghi lại việc dừng theo gate đã nêu trước khi xem kết quả.

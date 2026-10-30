# Đồng bộ AHU–HWS và giới hạn tổn thất thông gió

Ngày triển khai: 26/10/2026. Tiếp nối `HEAT_DELIVERY_REVIEW.md`; chưa cần
điểm nộp Kaggle để phát triển và đo đối chứng local.

## Hai thay đổi độc lập

`CoordinatedHeatPolicy` kế thừa `HeatDeliveryPolicy` và giữ nguyên các biến
thể `d_*` để còn đối chiếu được với batch đã đo.

1. **Đồng bộ nước nóng**: khi thực sự bật AHU bổ sung cho reheat, bật HWS và
   khôi phục áp suất bơm. Dùng lại water reset của controller cha, không hạ
   nhiệt độ nước mà một AHU đang sưởi khác yêu cầu. Tôn trọng pump pressure cap
   đã cấu hình. Không cập nhật tích phân lần thứ hai trong cùng một bước.
2. **Chặn thông gió bổ sung có hiệu suất sưởi thấp**: ước tính nhiệt độ khí trộn
   từ nhiệt độ phòng và ngoài trời. Với giới hạn khí ra coil 308.15 K, tỷ lệ
   tối đa nhiệt coil đi vào phòng là
   `(308.15 - target) / (308.15 - mixed_air)`, chặn trên ở 1. Nếu dưới 0,5,
   không bật thêm AHU. Vòng sưởi cha vẫn hoạt động bình thường. Đây là giả
   thuyết hiệu suất; chưa mô hình hóa đầy đủ tải phòng hay công suất coil.

Giới hạn recirculation 70% và discharge 35 °C là giá trị mặc định simulator,
được đưa thành tham số constructor. Controller chỉ đọc metadata, sensor và
trạng thái PI; không dùng tên scenario hoặc nhãn thời tiết để quyết định.
Khi thiếu sensor ngoài trời, biến thể có guard giữ quyết định của vòng cha.

| Biến thể | Đồng bộ HWS | Guard thông gió |
|---|---|---|
| `d_heat` đã đo | Không | Không |
| `e_sync` | Có | Không |
| `e_guard_air` | Không | Có |
| `e_guard_sync` | Có | Có |

## Phép đo

- Ba biến thể mới × `building_train_7/fairbanks_7` và
  `building_train_1/fairbanks_1`: 6 rollout đủ 8.640 bước, tách trường hợp
  được lợi và regression lạnh sâu. Kết quả: `ablation_heat_sync_probe.json`.
- Bổ sung `e_sync` trên `fairbanks_2`, `denver_2`, `atlanta_8` để so cùng
  bộ 5 scenario của batch ban đầu. Lưu hai file `heat_sync_train_fairbanks2.json`
  và `heat_sync_train_remaining.json`; gộp thành `heat_sync_train5.json` sau
  khi xác nhận không thiếu, trùng hoặc lỗi rollout.
- `e_sync` × `denver_1`, `fairbanks_3`, `singapore_2` của `building_9001`
  trên dev: 3 rollout đủ 8.640 bước, gồm hai trường hợp có nhu cầu sưởi và một
  đối chứng nóng. So với đúng 3 hàng `h_288` trong `h288_dev.json`, không so
  với trung bình cả 20 scenario dev. Kết quả: `heat_sync_dev.json`.
- Giữ test split và bản nộp hiện tại để tránh chọn policy theo chúng. Một
  building dev chưa đại diện cho toàn bộ tập dev hoặc private leaderboard.

Ban đầu có chạy dev 5 weather cho `e_guard_sync`. Sau khi `e_sync` hoàn tất
cả hai scenario training với kết quả tốt hơn `h_288`, ưu tiên xác nhận bản
đơn giản này trên 3 scenario dev nêu trên. Đã dừng lượt dev guard chưa hoàn
tất; lưu log và `heat_sync_dev_probe_aborted.json`, không tính nó thành một
điểm dev 5 scenario. Test split không tham gia việc chọn này.

## Kiểm tra triển khai

18 unit test qua, gồm 7 test cũ, 8 test đồng bộ/guard mới, 1 test runner
tuần tự và 2 test đóng gói. Có kiểm tra bật HWS cùng bước, pump pressure cap, không có demand,
AHU đã sưởi, thiếu weather, guard cho hai mức nhiệt độ, và ablation tắt đồng bộ
tái hiện `d_heat`.

Session Windows hiện tại từ chối tạo pipe của multiprocessing. Đã bổ sung
đường chạy tuần tự thật khi `--workers 1`, không tạo process pool. CLI đặt
giới hạn BLAS trước khi import numpy để tránh chạy nhiều thread không cần
thiết. Lỗi tạo pipe xảy ra trước rollout; batch đã được chạy lại theo đường
tuần tự, không trộn kết quả lỗi với kết quả đo.

Đã thêm builder riêng:

```powershell
python -m hvaclab.submission.experimental --variant e_sync --verify
```

Nó ghép đúng chuỗi policy và tham số registry, xuất vào
`experiments/submission/e_sync/`, giữ bundle mặc định. Test kiểm tra action
parity cho cả ba biến thể và không còn import nội bộ repo trong mã được ghép.
Đồng thời sửa lỗi notebook đặt future import sau inference-server imports:
nay hoist lên đầu cell và compile toàn bộ cell trước khi ghi file.

Tái hiện bảng kết quả sau khi các sweep hoàn tất bằng
`python -m hvaclab.heat_sync_analysis`. Công cụ kiểm tra hash nguồn đã đo,
đủ số scenario, horizon, lỗi và reward decomposition trước khi lập bảng.

## Trạng thái

Đã hoàn tất các phép đo bên dưới. Giữ bundle mặc định; notebook mới nằm riêng trong `experiments/submission/e_sync/`.

## Kết quả đủ episode

Tất cả hàng bên dưới đủ 8.640 bước, không lỗi; thành phần reward được cộng lại và đối chiếu trong ngưỡng 1e-9. Mức giảm chỉ áp dụng cho các scenario được ghi rõ.

### Cùng 5 scenario của batch ban đầu

| Scenario | h_288 | e_sync | Thay đổi score |
|---|---:|---:|---:|
| `building_train_1/fairbanks_1` | 0.07685394 | 0.05873991 | -23.57% |
| `building_train_2/fairbanks_2` | 0.05984411 | 0.01665602 | -72.17% |
| `building_train_4/denver_2` | 0.04853577 | 0.01005284 | -79.29% |
| `building_train_7/atlanta_8` | 0.05341082 | 0.01306810 | -75.53% |
| `building_train_7/fairbanks_7` | 0.06747863 | 0.01412923 | -79.06% |
| **Mean cùng 5 scenario** | **0.06122466** | **0.02252922** | **-63.20%** |

### Cùng 3 scenario dev trên nhà mới

| Scenario | h_288 | e_sync | Thay đổi score |
|---|---:|---:|---:|
| `building_9001/denver_1` | 0.06160402 | 0.01795161 | -70.86% |
| `building_9001/fairbanks_3` | 0.06289711 | 0.01456941 | -76.84% |
| `building_9001/singapore_2` | 0.01177561 | 0.01190436 | +1.09% |
| **Mean cùng 3 scenario** | **0.04542558** | **0.01480846** | **-67.40%** |

### Tách tác dụng của hai thay đổi trên 2 scenario đối lập

| Policy | Mean score | Comfort | Energy | Carbon |
|---|---:|---:|---:|---:|
| `d_ctrl` | 0.07216629 | 0.04248440 | 0.00939902 | 0.02028286 |
| `d_heat` | 0.06487903 | 0.02845626 | 0.01133956 | 0.02508321 |
| `e_guard_air` | 0.05429765 | 0.02110515 | 0.01042288 | 0.02276962 |
| `e_guard_sync` | 0.04134449 | 0.00973437 | 0.00991588 | 0.02169424 |
| `e_sync` | 0.03643457 | 0.00128856 | 0.01087581 | 0.02427019 |

Dữ liệu: `ablation_heat_sync_probe.json`, `heat_sync_train5.json`, `heat_sync_dev.json`; baseline: `ablation_heat.json` và `h288_dev.json`. Đã so cùng scenario và cùng horizon, không dùng bảng baseline cũ.

Đây là xác nhận sơ bộ trên 5 scenario training được chọn có chủ đích và 3 scenario của một nhà dev. Chưa phải kết quả full shipped-30, full dev, test hoặc Kaggle.

## Kết luận và ứng viên

- So với `d_heat` vừa đo, `e_sync` giảm thêm **45.87%** trên cùng 5 scenario. Cả 5 đều tốt hơn `h_288`.
- Comfort có trọng số giảm **98.13%** trên nhóm training và **94.71%** trên nhóm dev; energy + carbon tăng. Đây vẫn là đánh đổi có lợi cho tổng điểm.
- Dev Singapore tăng 1,09% (0.01177561 → 0.01190436), tức chênh lệch tuyệt đối khoảng 0.000129. Hai scenario dev cần sưởi cải thiện rõ.
- Ưu tiên `e_sync`: guard không cải thiện thêm trong ablation đã đo. Giữ hai biến thể guard làm bản đối chứng nghiên cứu.
- Notebook ứng viên: `experiments/submission/e_sync/submission.ipynb`. Đã kiểm tra 576 cặp action trên mỗi scenario Fairbanks 1 và 7, tổng 1.152 cặp khớp chính xác với registry. Notebook compile được; manifest ghi checksum của artifact và nguồn ghép. Chưa nộp Kaggle và giữ bundle mặc định.

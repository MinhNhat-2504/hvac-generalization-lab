# Rà paper, source và lịch sử thí nghiệm

Ngày rà: 27/10/2026. Mục tiêu là chọn hướng nghiên cứu có căn cứ, tránh lặp
các thất bại cũ và phân biệt bằng chứng trong source với phỏng đoán cơ chế.
Không thay controller hoặc notebook `e_sync` đang nộp.

## Phạm vi đã đọc và giới hạn

- Đã đọc toàn văn 581 dòng `reports/paper/main.tex`, đối chiếu
  `reports/KE_HOACH.md`, registry, các policy, runner và đóng gói.
- Đã rà trực tiếp đường chạy policy → action → thiết bị → thermal solver →
  observation/reward, đặc biệt AHU, VAV, HWS và thứ tự cập nhật cảm biến.
- Lập inventory 122 file Python không sinh tự động, gồm source, test và script
  audit, tại `source_inventory.json`. Inventory không có nghĩa mọi dòng của
  93 file starter kit, khoảng 29.800 dòng, đều đã được review ngữ nghĩa. Không
  khẳng định đã đọc từng dòng toàn bộ công cụ vẽ, I/O, generator và test.
- Lập `experiment_catalog.json`: 67 summary của 51 policy trong 26 file kết
  quả có schema sweep. Có ghi horizon, lỗi, tập scenario, hash file dữ liệu.
  Tham số trong catalog là registry hiện tại; không thay cho provenance lịch
  sử ở các run cũ chưa lưu source hash.

## Bảng những hướng đã thử

Nhóm A gồm cùng ba scenario `building_train_1/fairbanks_1`,
`building_train_7/singapore_6`, `building_train_8/phoenix_4`, đủ 8.640 bước.
Mỗi hàng phải so với đúng controller nền của nó, không so ngang các thời kỳ.

| Hướng | Nền | Thử nghiệm | Dữ liệu / ý nghĩa |
|---|---:|---:|---|
| Tắt/coast theo occupancy | pid_water 0.13478 | occ 0.18062 | Nhóm A, `occ_v2_probe.json` |
| Setback 3/5/8 K | flat 0.12321 | 0.16563 / 0.16017 / 0.18320 | Nhóm A, đều kém flat |
| Biên comfort 0,5 K | flat 0.12321 | 0.23605 | Nhóm A, comfort cũng xấu đi |
| Ép water floor cao | fan 0.04967 | 0.05362 | Nhóm A, thêm carbon |
| Bỏ fan economy lúc khẩn cấp | fan 0.04967 | 0.07357 | Nhóm A |
| Cap pump pressure | fan 0.04967 | 0.05090 | Nhóm A, `w_pump00` |
| Damper mở nhanh 1 K / 0,5 K | 0.05034 | 0.08965 / 0.11167 | Nhóm A |
| Tăng gain lên 1,0 | 0.05034 | 0.07937 | Nhóm A |
| Sensor band | 0.05034 | 0.05478 | Nhóm A |
| Worst-zone | 0.05034 | 0.09702 | Nhóm A |
| Gate damper theo supply sensor | z_ctrl 0.05034 | 0.06258 | Nhóm A; xem vấn đề sensor bên dưới |
| P mềm nhưng bỏ integral | p_ctrl 0.05034 | 0.05304 | Nhóm A; không bác bỏ PI kết hợp |
| Forced cooling + hold | h_288 0.03846 | all_288 0.05061 | Nhóm A; hai cơ chế can thiệp nhau |
| Guard thông gió + water sync | e_sync 0.03643 | 0.04134 | Hai Fairbanks 1/7, không phải nhóm A |
| Setback 1 K trên e_sync | e_sync 0.03643 | 0.03694 | Hai Fairbanks 1/7, +1,40%; không qua gate |

`h_6`, `h_96`, `w_pump-05`, `w_kp025`, `w_damper4/8` và một số key khác có
đăng ký nhưng không có kết quả lưu trong catalog. Không gọi một key đã đăng
ký là một thử nghiệm đã hoàn tất.

`occ_v1.json` có **28/30 run lỗi**. Summary của nó chỉ từ hai run thành công;
không dùng số đó như kết quả đầy đủ của 30 scenario hoặc bằng chứng đánh giá
controller. `occ_v2_probe.json` mới có ba run đầy đủ cho bảng trên.

Thử lại một ý tưởng sau khi controller nền đổi có thể hợp lý, nhưng phải ghi
rõ điều gì đã thay đổi, bằng chứng mới và gate dừng. Lần setback 1 K là một
phép thử lại trên `e_sync`, không phải hướng chưa ai thử; kết quả tiếp tục âm.

## Paper đang là snapshot trước Version 4

1. Các bảng chính 30 training / 20 dev / 8 test mô tả `h_288`, không phải
   `e_sync`. Các mức giảm 63,2% và 67,4% mới chỉ thuộc 5 training được chọn
   và 3 dev trên một nhà, không được thay trực tiếp vào bảng full-set.
2. Kết luận cold tail và “các thay đổi sau không tác động đến sưởi” mô tả
   controller cũ. `e_sync` đã sửa một phần đường tải nhiệt, nên cần chẩn đoán
   lại phần loss còn lại trước khi chọn hướng nghiên cứu.
3. Caption ablation nói mỗi hàng thêm một thay đổi vào hàng trước, nhưng
   “cooling clamp only” và “heating clamp only” là hai nhánh từ flat, không
   phải hai hàng cộng dồn. Cần vẽ lại quan hệ đối chứng.
4. “Monotonically worse” không khớp 3 K = 0.16563 và 5 K = 0.16017. Kết
   luận dữ liệu hỗ trợ là các setback đã thử đều kém flat trên nhóm A.
5. Phí comfort 0,12 cho lệch 1 K là khi toàn bộ trọng số occupancy chịu lệch
   đó. Một zone chiếm tỷ lệ occupancy `w` đóng góp `0.12*w`, không mặc nhiên
   0,12 cho từng zone. Trường hợp người đầu tiên đến thì `w` có thể bằng 1.
6. Fan power trong source là `flow * pressure / efficiency`. Chưa có cơ sở
   gán cơ chế tiết kiệm của hold cho một hạng startup penalty hoặc ống gió
   lạnh riêng. Kết quả hold có lợi vẫn là bằng chứng thực nghiệm hợp lệ;
   lời giải thích cơ chế phải bám mô hình thực sự được tính.
7. “Ít khí nóng luôn rẻ hơn vì gas như nhau” thiếu điều kiện khả thi. VAV có
   discharge cap 308.15 K; giảm flow vừa đổi tải khí ngoài, vừa đổi công suất
   sưởi tối đa. Kết quả fan clamp không chứng minh một định lý đơn điệu cho
   mọi tải phòng.
8. Test chỉ gồm Minneapolis là một estimate thiên về lạnh, không phải lower
   bound toán học của cải thiện trên private leaderboard. “Scalar dùng chung
   không thể overfit” trong kế hoạch cũng quá mạnh; global scalar vẫn có thể
   fit phân phối scenario hoặc quá trình chọn tham số.
9. RLS/MPC/CMA-ES hiện là hướng trong kế hoạch, chưa có implementation trong
   controller `hvaclab`. Paper và kế hoạch cần ghi rõ đâu là kết quả đã làm,
   đâu là công việc dự kiến. Chưa kiểm chứng các quy định/nguồn bên ngoài trong
   lượt audit local này.

Chưa tự viết lại manuscript với các kết luận chưa được đo. Các điểm trên là
danh sách sửa cho lần cập nhật paper sau, đồng thời giữ lại lịch sử thí nghiệm.

## Hai phát hiện về observation đã kiểm tra trực tiếp

### Supply sensor là setpoint

`smart_buildings/smart_control/simulator/air_handler.py:224` trả
`self._supply_air_temperature_setpoint` từ `supply_air_temperature_sensor`.
Đường request observation đi qua `SmartDevice.get_observation`, không thay nó
bằng kết quả `get_supply_air_temp`.

Ví dụ đã chạy: return 295 K, outside 260 K, recirculation 0,7, SAT setpoint
305 K. Sensor báo **305 K**, khí thực sau trộn/cooling là **284,5 K**.
`ZoneAwarePolicy._supply_air_temp` đang đọc sensor này như khí thực. Vì vậy
`z_gate` thất bại chưa đủ để bác bỏ nguyên tắc xét nhiệt độ khí thực; phép
triển khai đã dùng một đại lượng khác với điều nó giả định.

Quan sát airflow setpoint của VAV cũng là design `max_air_flow_rate`, không
phải lưu lượng thực đo. Mô hình nhận dạng cần kiểm tra semantics của từng
feature thay vì dựa vào tên field.

### Nhiệt độ phòng có trễ một bước

`SimulatorFlexibleGeometries.execute_step_sim` lấy zone temperature và gọi
`vav.update_settings` trước thermal solve. VAV lưu giá trị đó làm sensor, còn
reward lấy thermal field mới. Probe 3 bước đã kiểm tra:

| Bước | Nhiệt độ thực trước bước | Sensor sau bước | Nhiệt độ thực sau bước |
|---|---:|---:|---:|
| 0 | 294.00000 K | 294.00000 K | 297.39479 K |
| 1 | 297.39479 K | 297.39481 K | 299.30731 K |
| 2 | 299.30731 K | 299.30731 K | 300.64045 K |

Sensor sau bước khớp thermal state trước bước trong sai số float, tức feedback
trễ một bước 300 giây trên đường chạy đã kiểm tra. Đây là constant-maximum-
action diagnostic, không phải trajectory hoặc score của `e_sync`.
True simulator state chỉ dùng để audit offline; policy khi nộp chỉ được dùng
observation, metadata và lịch sử lệnh của chính nó.

Tái hiện: `python reports/scripts/observation_probe.py`. Dữ liệu:
`experiments/runs/observation_semantics_probe.json`.

## Đồng bộ HWS còn một edge case, nhưng chưa có bằng chứng lợi ích

Đã tái hiện bằng policy thật với metadata giả lập: sau khi arming cooling
hold, cho AHU có mean temperature trong band và một zone lạnh. Kết quả AHU
run=1, van zone mở 35%, PI muốn sưởi, HWS run=0; `e_sync` không sync nước vì
không có heat override mới. Đây là một nhánh chưa được sửa, không phải điểm
Kaggle hay bằng chứng nó thường xảy ra.

Đã thêm observer không sửa action và chạy hai probe 1.440 bước:

| Đại lượng | train_7/fairbanks_7 | dev_9001/singapore_2 |
|---|---:|---:|
| Bước PI active | 684 | 684 |
| Held cooling có yêu cầu reheat | 141 | 36 |
| Held cooling + reheat nhưng HWS tắt | 0 | 26 |
| Trong các bước đó có sensor zone dưới comfort low | 0 | 0 |

Ở Fairbanks, handler khác bật nước nên edge case không xuất hiện trong probe.
Ở Singapore có 26 bước nhưng chưa chứng minh mất comfort: yêu cầu PI nhắm tâm
band khác với vi phạm comfort, và sensor còn có trễ. Chưa có căn cứ bật nước
cho mọi PI request hoặc công bố sẽ cải thiện score.

Probe Fairbanks có mean reward khớp chính xác probe cũ chưa thêm observer;
unit test cũng kiểm tra observer không đổi action, PI integral hoặc hold.
Probe ngắn không được so score với episode đủ 8.640 bước.

## Hướng tiếp theo và thứ tự

1. **Đo rộng bản e_sync cố định trước.** Mới có 5/30 training và 3/20 dev;
   còn 25 + 17 rollout. Đã chuẩn bị manifest `experiments/plans/e_sync_coverage.json`
   với cache và danh sách thiếu; trạng thái `prepared_not_started`. Không dùng
   các phần trăm của tập chọn để dự báo trực tiếp leaderboard.
2. **Ưu tiên căn chỉnh dữ liệu và ước lượng trạng thái/bù trễ.** Từ lịch sử
   observation và action, ước lượng nhiệt độ hiện tại trước khi PI ra quyết
   định. Đây khác các thử nghiệm tăng gain, margin hoặc setback đã thất bại.
   RLS nhỏ có thể là một implementation để đánh giá; chưa có bằng chứng nó
   cải thiện score. Cần kiểm tra prediction trên dữ liệu đúng độ trễ, chạy
   shadow prediction trước khi cho model sửa action, rồi giữ fallback e_sync.
3. **Đo tính nhất quán giữa cooling hold và nhu cầu reheat**, đặc biệt thời
   điểm có occupancy và comfort regret, sau khi căn chỉnh observation/reward.
   Chỉ làm ablation sửa interlock nếu trace cho thấy phần loss đủ đáng kể.
4. Nếu mở lại gate theo SAT, phải dùng nhiệt độ khí thực ước lượng đúng và
   lưu thành variant mới. Giữ `z_gate` cũ làm kết quả lịch sử, không sửa nó
   rồi gán số cũ cho implementation mới.

Version 4 chấm xong sẽ bổ sung một phép kiểm tra bên ngoài. Nó không phải
điều kiện để đọc source, chẩn đoán observation hoặc hoàn thiện local coverage.
Không tạo hay nộp thêm notebook trong lượt audit này.

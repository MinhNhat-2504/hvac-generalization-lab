# Giám sát nền cho đợt gated reheat

Chạy từ thư mục repo bằng PowerShell:

```powershell
powershell -NoProfile -File reports\scripts\start_study_supervisor.ps1
```

Launcher mở tiến trình ẩn độc lập, không đổi power plan hay cấu hình khởi động
Windows. Supervisor dùng khóa PID/thời điểm tạo để ngăn mở trùng, kiểm tra mỗi
15 giây và tiếp quản scheduler đã chạy. Nếu scheduler mất nhưng worker còn sống,
nó chờ worker kết thúc để tránh hai tiến trình ghi cùng kết quả. Khi không còn
tiến trình và thí nghiệm chưa hoàn tất, nó kiểm tra hash nguồn/tham chiếu,
RAM >= 1.75 GiB và dung lượng >= 1 GiB trước khi chạy scheduler từ checkpoint.

Mỗi lần mở scheduler mới có file `scheduler_<UTC>.log`. Lịch sử sự kiện nằm ở
`supervisor_events.jsonl`, trạng thái ở `supervisor_status.json`, trong
`experiments/runs/gated_reheat_calibration`. Exit code được lưu, thử lại sau
15–300 giây; năm lần thoát liên tiếp trước khi chạy ổn định năm phút sẽ chặn
retry để kiểm tra lỗi. Không tự sửa mã hoặc bỏ qua hash. Không coi thiếu RAM
hay máy ngủ là lỗi rollout, không giết tiến trình chỉ vì heartbeat chậm.

Tạo `supervisor.stop` trong thư mục run để kết thúc giám sát; scheduler/worker
đang chạy được giữ nguyên. Launcher sẽ bỏ marker này khi được gọi lại.
Supervisor không tự đăng ký chạy sau reboot/logoff và không ngăn người dùng
chủ động cho máy ngủ. Máy ngủ thì mô phỏng ngừng tiến triển; sau reboot cần
chạy lại launcher. Đóng chat không phải thao tác dừng supervisor đã tách nền.
Các tiến trình/worker đang được tiếp quản có thể không có log scheduler từ
trước; log đầy đủ áp dụng cho các lần khởi chạy mới qua supervisor.

Kiểm thử gồm child Python thoát thật với code 7, lưu log, retry và child tiếp
theo hoàn tất, cùng các kiểm tra không trùng scheduler, orphan workers,
hash/RAM gate, retry giới hạn và stop marker.

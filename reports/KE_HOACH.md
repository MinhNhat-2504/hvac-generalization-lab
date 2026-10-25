# Kế hoạch giành giải nhất cả hai hạng mục

> Tài liệu này viết bằng tiếng Việt vì là ghi chú làm việc nội bộ.
> `README.md` giữ tiếng Anh vì đó là mặt tiền repo cho ban giám khảo đọc.

Hai hạng mục, hai mục tiêu khác nhau, nhưng dùng chung một khối công việc.

- **Main leaderboard** — 25.000 USD, giải nhất 5.750 USD. Hạn cuối 15/01/2027.
  Chấm hoàn toàn bằng metric trên 30 kịch bản ẩn.
- **Paper track** — 15.000 USD, Final Paper giải nhất 3.000 USD, cộng vòng
  Poster có hạn **13/11/2026 — sớm hơn hạn code rất nhiều**. Chấm theo phương
  pháp, tính mới, chất lượng nghiên cứu và độ rõ ràng, không chấm theo metric.

Hạn poster là ràng buộc quyết định thứ tự làm việc. Cái gì đo được và chứng
minh được trước đầu tháng 11 thì cái đó lên poster.

---

## 1. Điểm nằm ở đâu — bốn phát hiện định lượng

Đây là phần quan trọng nhất của toàn bộ kế hoạch. Tất cả đều suy ra từ việc
đọc mã nguồn hàm thưởng, không phải phỏng đoán.

### 1.1 Điểm thoải mái bị CHẶN theo số người, chặn tuyệt đối

Trong `smart_control/reward/base_setpoint_energy_carbon_reward.py`:

```python
if total_occupancy > 0.0:
    normalized_productivity_regret = actual/max - 1.0
else:
    normalized_productivity_regret = 0.0
```

Nghĩa là **bước nào tòa nhà không có người thì thành phần thoải mái bằng đúng
0**, bất kể nhiệt độ các phòng ra sao. Khi đó phần thưởng của bước chỉ còn:

```
r = -0.2 * năng_lượng  -  0.2 * carbon
```

Người đến từ giờ 6 đến 12, về từ 13 đến 19, và `is_work_day` buộc mọi người
vắng mặt vào cuối tuần và ngày lễ. Trên một đoạn 30 ngày, **khoảng 60% trong
8.640 bước được tính điểm là lúc tòa nhà trống rỗng**. Mỗi joule đốt trong
những bước đó mua về đúng con số không.

Cả 4 baseline đều giữ dải eco suốt 24/7, nên cả 4 đều trả hóa đơn này.

Đo trên `building_train_1/fairbanks_1` (kịch bản baseline tệ nhất), tách phần
lỗ của `pid_water` theo trọng số:

| thành phần | đóng góp |
|---|---|
| thoải mái | 0,087 |
| năng lượng | 0,077 |
| carbon | 0,122 |

→ **69% phần lỗ là năng lượng + carbon**, và hạng mục lớn nhất là carbon từ
lò hơi gas: giữ 289 K chống lại không khí −42 °C suốt đêm, không nhận được
một điểm thoải mái nào.

### 1.2 Khi có người, thoải mái đắt hơn năng lượng khoảng 35 lần

Regret là **trung bình có trọng số** độ thiếu hụt từng phòng, và độ thiếu hụt
bằng `lệch / 5 K` trong vùng tuyến tính. Nên mỗi phòng có người lệch 1 K khỏi
dải tốn `0,6 × 0,2 = 0,12` phần thưởng mỗi bước.

Một kWh điện tốn: `0,2×(0,16/6,37) + 0,2×(0,103/15,14) = 0,0064`.

→ Bù 1 K lệch toàn nhà đáng giá **≈19 kWh mỗi bước 5 phút, tức 225 kW chạy
liên tục**, trên tổng công suất 320 kW. Khi có người, năng lượng gần như miễn
phí so với thoải mái. Khi trống, thoải mái miễn phí và năng lượng là toàn bộ
hóa đơn.

**Sự bất đối xứng này chính là toàn bộ chiến lược: tiêu thoải mái khi bị chấm,
không tiêu gì khi không bị chấm.**

### 1.3 Quạt là loại năng lượng đắt nhất

Chi phí điểm mỗi kWh:

| dạng | chi phí điểm |
|---|---|
| điện quạt / bơm | 0,0064 |
| nhiệt từ gas | 0,0034 |
| lạnh từ chiller (COP 3,5) | 0,0018 |

Quạt không có COP để chia, nên **áp suất tĩnh là đòn điều khiển cần canh nhất**.
Gas rẻ hơn điện về tiền nhưng carbon gấp đôi (0,181 vs 0,103 kg CO₂/kWh).

### 1.4 Điểm là trung bình 30 kịch bản đồng trọng số, và cái đuôi quyết định

4 kịch bản tệ nhất chiếm ~38% tổng điểm lỗ của `pid_water`. **Cải thiện trường
hợp trung bình chỉ đáng một phần nhỏ so với sửa trường hợp tệ nhất.**

### 1.5 Hệ quả quan trọng cho buổi sáng

Regret chuẩn hóa theo **tổng** số người. Nên khi mới có 3 người đến và chưa ai
khác, 3 phòng đó gánh toàn bộ điểm thoải mái. Người đến sớm có phương sai rất
cao và rủi ro rất lớn.

→ **Mọi phòng phải nằm trong dải ngay tại `morning_start_hour`, không phải
trung bình là đủ.**

---

## 1.6 ⚠ KẾT QUẢ ÂM: "tắt máy khi nhà trống" KHÔNG hiệu quả

Đã triển khai và đo đầy đủ. Kết quả **tệ hơn** `pid_water` trên cả 3 scenario
tham chiếu. Hai nguyên nhân độc lập, không nguyên nhân nào thấy được từ con số
tổng:

**(a) Nhà trống *trong* khung giờ được chấm không phải là cơ hội.**
Người đến theo các lần rút ngẫu nhiên độc lập từ giờ 6, nên nhà thường trống
vài bước *sau khi* khung mở. Và regret chuẩn hóa theo **tổng** số người, nên
người đầu tiên bước vào gánh **toàn bộ** điểm thoải mái của phòng họ. Tắt máy
lúc 06:05 vì "chưa ai tới" nghĩa là đang tắt, mọi phòng ngoài dải, đúng lúc họ
tới.

**(b) Ép dải comfort ngoài khung giờ còn tốn hơn là để trôi.**
Bản đầu nhắm dải comfort (294–297 K) mỗi khi thấy cần hồi phục, trong khi các
baseline dùng dải eco rộng (289–298 K). Ở Fairbanks đó là **sưởi thêm 5 K suốt
đêm**. Và nó hiện ra đúng chỗ phải hiện: lỗ ở các bước nhà trống **tăng** từ
447 lên 830, trong khi thoải mái chỉ tốt hơn 0,010.

→ **Đòn điều khiển không phải cái công tắc bật/tắt. Nó là chính cái DẢI.**

## 1.7 Thiết kế thay thế: dải theo hạn chót (`deadline_band`)

Không còn chế độ nào. Mỗi bước tính đúng một đại lượng — dải cần giữ lúc này —
rồi giao cho luật điều khiển kế thừa:

| khi nào | dải |
|---|---|
| trong khung được chấm | đúng dải comfort |
| ngoài khung, hạn còn xa | dải comfort nới ra toàn bộ setback |
| ngoài khung, đang tới hạn | nới một phần, thu hẹp tuyến tính về dải comfort đúng lúc tới hạn |

Mấu chốt: luật kế thừa **vốn đã** làm đúng khi dải rộng — khi mọi phòng nằm
trong dải, `pid` cho AHU nghỉ, đóng damper, đóng van reheat, hạ vòng nước về
mức thấp nhất và xóa mọi lệnh chạy. **Một dải đủ rộng để chứa tòa nhà CHÍNH LÀ
hành động tắt**, và nó đi qua cùng một đường mã như mọi trường hợp khác, nên
không có nhánh tắt riêng nào để làm sai.

Độ sâu setback là **tham số được đo**, không phải tính toán khả thi. Bản đầu
suy nó ra từ ước lượng tốc độ hồi phục trực tuyến, và chính ước lượng đó là chỗ
yếu: nó đo từ việc *phòng tệ nhất* thu hẹp độ lệch, điều này **phóng đại** khả
năng của một hệ thiết bị dùng chung cho cả tòa nhà.

## 1.8 Hướng mới từ vật lý: quạt là nhiên liệu đắt nhất (`fan_economy`)

Mọi policy trong chuỗi `pid` đặt `áp_suất = clip(2|u| − 1, −1, 1)` — tức **tăng
nhiên liệu đắt nhất theo cường độ nhu cầu**. Nhưng `abs(u)` không phân biệt hai
chiều, mà hai chiều **không đối xứng**:

- **Sưởi.** AHU *không thể* sưởi; mọi nhiệt đến từ van reheat. Trong giá lạnh
  thì cả hành động nhiệt độ khí cấp cũng vô hiệu: AHU cấp
  `min(khí_trộn, setpoint)`, và với recirculation chặn ở 0,70, khí hồi ~293 K,
  ngoài trời 231 K thì khí trộn là `0,70×293 + 0,30×231 = 274 K`, **đã thấp hơn
  mức sàn 285 K** mà setpoint có thể ra lệnh. Tăng áp suất quạt là đẩy **thêm**
  khí 274 K qua coil, và từng mét khối đó phải được hâm từ 274 K trước khi giúp
  được gì cho phòng. **Tốn điện quạt, rồi tốn gas để sửa cái khí mà quạt vừa
  đẩy vào.**

  Nhiệt mà phòng cần là cố định theo tổn thất của nó. Nó có thể đến dưới dạng
  *nhiều khí hơi ấm* hoặc *ít khí rất nóng* — và cách thứ hai **rẻ hơn hẳn**,
  vì lượng gas như nhau còn quạt thì không.

- **Làm lạnh.** Ngược lại, và khí là *phương tiện duy nhất*, nên phải mua lưu
  lượng. Nhưng đánh đổi vẫn nghiêng về nhiệt độ hơn lưu lượng.

**Đo lần đầu** (1 ngày không người trên fairbanks_1, nên là so sánh thuần
năng lượng/carbon, không có rủi ro thoải mái): **0,00908 → 0,00499**, năng
lượng **−57%**, carbon **−38%**.

Hệ quả bắt buộc: hạ lưu lượng mà không nâng nhiệt độ nước thì sẽ **thiếu
nhiệt**. Nên có thêm tham số `heating_water_floor`. Chính docstring của
`pid_water` cũng tới cùng kết luận từ phía kia: *"Trong các kịch bản lạnh nhất,
nước mát tốn NHIỀU hơn nước nóng: coil không cấp đủ, nên quạt chạy full áp suất
cả episode để đẩy khí không đủ ấm."* — họ nhận ra nhưng **không sửa luật quạt**.

## 2. Giai đoạn 1 — Chặn theo số người (ĐÃ ĐO, kết quả âm — xem 1.6)

`hvaclab/policies/occupancy_gated.py`. Ba chế độ, giữ nguyên luật điều khiển
`pid_water` kế thừa để phép so sánh chỉ khác đúng một yếu tố:

- **ACTIVE** — có người. Chạy hết công suất, vì thoải mái đắt gấp 35 lần.
- **RECOVER** — trống, nhưng không còn kịp trôi thêm mà vẫn vào dải đúng hạn.
  Cùng luật đó, nhưng nhắm dải *comfort* thay vì dải eco.
- **COAST** — trống và vẫn còn kịp hồi phục. Tắt hết.

COAST bị chặn bởi **điều kiện khả thi**, không phải nhiệt độ setback cố định:

```
hồi_phục_được = tốc_độ_sưởi × số_bước_tới_hạn × AN_TOÀN
sàn            = comfort_low − hồi_phục_được
```

`sàn` là nhiệt độ thấp nhất mà *lúc này* tòa nhà được phép có mà vẫn kịp vào
dải. Đầu đêm hạn còn xa nên sàn nằm dưới mọi mức thời tiết có thể gây ra, máy
tắt. Hạn càng gần sàn càng dâng lên gặp nhiệt độ thật, và đúng lúc giao nhau
thì bắt đầu hồi phục. **Đường dốc buổi sáng không được lập trình ở đâu cả — nó
tự nảy ra từ phép thử khả thi.**

`tốc_độ_sưởi` **đo từ chính phản ứng của tòa nhà**, không fit sẵn ngoại tuyến.
Đây là điều cho phép cùng một đoạn mã chuyển được sang vỏ nhà và hệ thiết bị
chưa từng thấy. Khởi tạo cố ý bi quan: đi sớm thì tốn vài giờ gas một lần, đi
muộn thì ăn hình phạt thoải mái vô hạn mỗi buổi sáng.

Không dùng `comfort_mode_soon` cho việc này vì nó chỉ nhìn trước đúng 60 phút,
còn trong giá lạnh tòa nhà cần lâu hơn nhiều.

**Mốc cần vượt: `pid_water` = 0,0541.**

---

## 3. Giai đoạn 2 — Sửa hai bệnh lý của baseline

Cả hai đã chẩn đoán xong, và cả hai đáng giá hơn mức sàn oracle 0,0473.

### 3.1 Sưởi và làm lạnh đánh nhau

Trên `singapore_4`, `pid_water` được 0,109 còn `fixed_setpoint` 0,024 — tệ hơn
ở cả ba thành phần, với **carbon gấp 9 lần**, trong một khí hậu không bao giờ
cần sưởi.

Nguyên nhân: AHU chỉ chọn được MỘT nhiệt độ khí cấp cho mọi phòng nó phục vụ.
Vòng worst-zone kéo nhiệt độ đó xuống thật lạnh để lo cho phòng nóng nhất, các
phòng bị lạnh quá liền mở van reheat, và tòa nhà trả tiền gas để phá đi cái
lạnh mà nó vừa trả tiền điện để tạo ra.

**Cách sửa:** khóa liên động theo từng AHU — khi *đa số* phòng của nó bị lạnh
quá thì nâng nhiệt độ khí cấp lên chứ không reheat, và chỉ reheat cho phòng
ngoại lai thật sự.

### 3.2 Đốt gas ban đêm trong giá lạnh

Trên `fairbanks_1`, `pid_water` 0,212 còn `fixed_setpoint` 0,147. Giai đoạn 1
xử lý phần lớn chuyện này.

Phần còn lại là hồi phục buổi sáng, nơi hành động nhiệt độ khí cấp **vô hiệu
khi ngoài trời dưới −6,8 °C** (khí trộn đã lạnh hơn mức sàn 285 K). Lúc đó đòn
điều khiển thật chỉ còn nhiệt độ nước nóng và các van reheat.

---

## 4. Giai đoạn 3 — Dùng đúng những giờ có người

Phát hiện 1.2 nói các baseline quá rụt rè khi có người:

- `pid` và hậu duệ co áp suất quạt theo `|u|`. Đúng cho năng lượng, sai cho tốc
  độ hồi phục. Với thoải mái gấp 35 lần năng lượng, nhánh **có người** nên
  nghiêng về đóng khoảng lệch cho nhanh, nhánh **trống** nên nghiêng về rẻ. Một
  bộ điều khiển không thể tinh chỉnh cho cả hai; việc tách chế độ ở Giai đoạn 1
  biến nó thành hai.
- Quyền điều khiển từng phòng là nơi duy nhất tồn tại quyết định cấp phòng.
  Damper và reheat nên lấy dải từ **cảm biến của chính phòng đó**, vì đó là cái
  bộ chấm điểm đọc.
- Tiền điều hòa nên nhắm vào **bên trong** dải một chút, đừng nhắm sát mép, để
  trôi tự nhiên không làm vượt mép trong giờ có người đầu tiên.

---

## 5. Giai đoạn 4 — Nhận dạng hệ thống trực tuyến (lõi của paper)

Đây vừa là phần kỳ vọng lợi nhiều nhất, vừa là đóng góp khoa học của paper.

Policy **không thể** tinh chỉnh theo từng tòa nhà, vì nó không bao giờ thấy các
tòa nhà đánh giá. Nên những hằng số quan trọng — hằng số thời gian nhiệt, tốc
độ sưởi/làm lạnh đạt được, tổn thất vỏ nhà trên mỗi K chênh lệch, hệ số ghép
của từng phòng với AHU của nó — **phải được ước lượng từ phản ứng của chính
tòa nhà ngay trong lúc chạy**.

Giai đoạn 1 đã làm phiên bản đơn giản nhất cho tốc độ hồi phục. Dạng tổng quát:
hồi quy bình phương tối thiểu đệ quy (RLS) cho từng phòng trên

```
dT_phòng = a·(T_cấp − T_phòng)·lưu_lượng + b·reheat + c·(T_ngoài − T_phòng) + d
```

fit trực tuyến, rồi dùng cho (i) phép thử khả thi, (ii) MPC chân trời ngắn trên
nhiệt độ khí cấp và nhiệt độ nước, (iii) phát hiện **phòng không thể với tới**
— thứ mà docstring của `pid_water` chỉ ra là một kiểu hỏng thật, và tốt nhất là
đừng đuổi theo.

Một episode 30 ngày là 8.640 bước, và buổi sáng đầu tiên không có rủi ro điểm
nếu bộ điều khiển thận trọng, nên dư dữ liệu. Điểm hấp dẫn: nó chuyển được
**do cấu trúc**, chứ không phải nhờ hy vọng phân phối huấn luyện bao được phân
phối kiểm tra.

---

## 6. Giai đoạn 5 — Tìm tham số, KHÔNG dùng deep RL

Bộ điều khiển trên có cỡ 15–25 tham số vô hướng (hệ số, biên an toàn, dải,
ngưỡng liên động). Fit chúng bằng **CMA-ES** trên nhiều tòa nhà *sinh ra* thì
hiệu quả mẫu cao hơn hẳn việc học một mạng policy, và không thể overfit bố trí
tòa nhà vì các tham số đều vô hướng không chiều và dùng chung.

Deep RL là thứ hiển nhiên nhưng có lẽ là thứ sai ở đây: không gian hành động là
`3K+2+2V` với K và V đổi theo từng nhà, episode dài 8.640 bước trên một
simulator chậm, và phần thưởng bị chi phối bởi sự kiện đuôi hiếm. Một GNN policy
sẽ cần rất nhiều compute mới đạt tới mức một bộ điều khiển có cấu trúc như trên.
**Residual RL đặt lên trên bộ điều khiển có cấu trúc** mới là phiên bản bảo vệ
được, và đó là Giai đoạn 6 nếu còn ngân sách.

Huấn luyện trên nhà sinh ra → chọn trên `dev` → báo cáo trên `test`.
**Không bao giờ chọn mô hình dựa trên 30 kịch bản có sẵn.**

---

## 7. Kỷ luật đo lường — cái bẫy lớn nhất của cuộc thi

30 kịch bản có sẵn là **tập huấn luyện**. Tinh chỉnh trên chúng rồi đọc kết quả
như tiến bộ chính là cái bẫy trung tâm, vì leaderboard chạy trên tòa nhà và
đoạn thời tiết KHÔNG có trong repo này.

`hvaclab/holdout.py` dựng hai tập thay thế từ vật liệu mà tập có sẵn không hề
dùng tới. Trong 48 đoạn thời tiết, `training_set.yaml` đã lấy 30, còn trống 18:

- **dev** — nhà mới (seed 9000+) × 10 đoạn chưa dùng ở 5 trạm đã thấy. Tinh
  chỉnh ở đây.
- **test** — nhà mới (seed 9500+) × `minneapolis`, một trạm mà **không kịch bản
  có sẵn nào dùng**. Nên cả tòa nhà lẫn khí hậu đều chưa từng thấy. Xem thật
  ít — mỗi lần xem là tiêu bớt giá trị của nó như một ước lượng không chệch.

**Mỗi kết quả báo cáo đều mang đủ ba con số: shipped-30, dev, test.** Thay đổi
nào làm tốt shipped-30 mà không tốt dev là overfit, và bị revert.

---

## 8. Ràng buộc kỹ thuật khi nộp bài

Kaggle code competition: notebook, tắt internet, ≤9 giờ CPU và ≤9 giờ GPU,
private leaderboard trên ~67% dữ liệu test, các bài đứng đầu bị chạy lại.

Những điều phải tôn trọng ngay từ đầu:

- Policy phải là **Python + numpy thuần khi suy luận**, không tải gì, không
  phụ thuộc artifact đã fit mà không commit được. Nhận dạng trực tuyến khớp
  hoàn hảo với điều này; một mạng pretrained lớn thì không.
- 30 kịch bản × 8.640 bước trong 9 giờ cho ngân sách khoảng **1,1 giây mỗi
  bước** cho mọi thứ kể cả simulator (vốn mất ~0,05 s). Dư cho RLS và MPC chân
  trời ngắn; không dư cho bất cứ thứ gì phải mô phỏng lại.
- `get_action` chạy 8.640 lần mỗi kịch bản. Bảng chỉ số dựng trong `__init__`.
- **Không được tham chiếu nhãn thời tiết, tên trạm, tháng hay id tòa nhà** —
  giao diện cố ý bỏ chúng, và hard-code một cái là bị loại.

### ⚠ Phát hiện quan trọng: `baseline_returns.json` đã LỖI THỜI

Simulator **tất định tuyệt đối** — chạy hai lần cùng một policy trên cùng kịch
bản cho kết quả giống nhau đến 10 chữ số thập phân. Nên mọi chênh lệch đều là
sai lệch hệ thống, không phải nhiễu.

Nhưng chạy `pid_water` trên `building_train_1/fairbanks_1` bằng mã hiện tại cho
**0,21392**, còn `baseline_returns.json` ghi **0,21168** — lệch 1,1%.

Nguyên nhân: file baseline được commit ở `3bb6c32` (commit starter kit đầu
tiên) và **chưa bao giờ cập nhật**. Sau đó commit `9567419` ("Fix simulation
tracker querying occupancy with VAV IDs") đã sửa chỗ tracker truy vấn occupancy
bằng VAV ID — việc đó "tạo ra dân số ảo và tiêu thụ các lần rút số ngẫu nhiên,
làm thay đổi quỹ đạo mô phỏng về sau". Tức là quỹ đạo đã đổi, nhưng bảng
baseline thì không.

**Hệ quả:** không được so điểm của mình với `baseline_returns.json`. Phải **tự
sinh lại baseline** bằng mã hiện tại. Đây cũng là điều có lợi thế cạnh tranh —
nhiều đội sẽ so với file lỗi thời và tự đánh giá sai tiến bộ của mình.

### ⚠ Phát hiện: generator mặc định sinh nhà quá lớn

`run_floor_generator` mặc định lấy footprint 400–700 × 300–500 ft, lớn hơn hẳn
6 nhà có sẵn (318–430 × 188–301). Seed 9000 ở mặc định ra **162 phòng và 17
AHU**, trong khi cuộc thi công bố tập đánh giá là **28–85 phòng, 3–9 AHU**.

Và vì scenario generator đặt `num_ahus = ceil(phòng/10)`, footprint quá lớn còn
làm chi phí mô phỏng tăng theo. Tập holdout đầu tiên vừa không đại diện vừa
chậm.

Đã sửa: `hvaclab/holdout.py` lấy footprint từ chính khoảng của 6 nhà có sẵn,
rồi **đo lại số phòng thực tế từ `config.gin`** và co giãn nếu lệch khoảng —
vì số phòng không phải hàm đóng của footprint (generator tự phân hoạch mặt
bằng).

### Ghi chú hạ tầng máy cá nhân

Máy có 16 lõi nhưng chỉ 15,7 GB RAM. Mỗi worker simulator đạt đỉnh ~0,6–0,8 GB,
và scipy SuperLU **báo lỗi `SUPERLU_MALLOC` thay vì chạy chậm** khi không cấp
phát được — nên quá tải RAM làm *sạch* hỏng cả lượt chạy, chứ không phải chậm.
Sweep đầu tiên hôm 24/10 chết 28/30 vì lý do này.

Đã xử lý: `hvaclab/runner.py` bước thẳng qua `env` thay vì qua
`SimulationTracker` (tracker giữ một dòng nhiệt độ mọi phòng mỗi bước và tính
lại `get_zone_average_temps()` toàn lưới mỗi bước), và `default_workers()` giới
hạn số worker theo RAM còn trống chứ không chỉ theo số lõi.

---

## 9. Trình tự theo mốc thời gian

| hạn | việc |
|---|---|
| ngay | Giai đoạn 1 đo đủ trên shipped-30, dev, test |
| | Giai đoạn 2–3, mỗi bước là một ablation một-yếu-tố, kèm đủ ba con số |
| cuối 10/2026 | Giai đoạn 4 nhận dạng trực tuyến — luận điểm lõi của paper |
| **13/11/2026** | **Nộp Poster vòng 1** — trình bày kết quả occupancy-gating và phương pháp nhận dạng, kèm bảng ablation |
| 11–12/2026 | Giai đoạn 5 tìm tham số trên các nhà sinh ra |
| 13/12/2026 | NeurIPS workshop |
| 12/2026–01/2027 | Giai đoạn 6 residual learning nếu đáng; làm chắc notebook, dư thời gian chạy, an toàn khi bị chạy lại |
| **15/01/2027** | **Nộp bài cuối** |

---

## 10. Luận điểm của paper

Luận điểm đã là thứ mạnh nhất ở đây, và **không phụ thuộc vào việc có thắng
leaderboard hay không**:

> Thành phần chi phối trong benchmark này là năng lượng tiêu khi hàm mục tiêu
> không quan sát; và bộ điều khiển thắng là bộ biết khi nào nó đang bị chấm
> điểm, cùng với việc tòa nhà của nó hồi phục nhanh đến mức nào.

Đó là một phát biểu về **cấu trúc hàm thưởng** và **thích nghi trực tuyến** —
đúng những gì paper track nói là họ thưởng.

Thứ nhất, nearshore/coastal vẫn là vùng khó với altimetry.
Các review gần đây đều nhấn mạnh rằng radar altimetry đã rất mạnh ở open ocean, nhưng gần bờ vẫn là thách thức kỹ thuật và khoa học, do land contamination, sóng ngắn, động lực ven bờ phức tạp, corrections khó hơn, và sai số tăng khi tiến sát bờ.
- https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1592765/full?utm_source=chatgpt.com
- https://os.copernicus.org/articles/21/133/2025/?utm_source=chatgpt.com
- https://www.sciencedirect.com/science/article/pii/S1569843225006296?utm_source=chatgpt.com
Thứ hai, vệ tinh không cho chuỗi thời gian dày liên tục như gauge/station.
Một điểm rất quan trọng với forecasting là temporal sampling. Nhiều hệ vệ tinh, kể cả SWOT, có revisit không liên tục; có ứng dụng mà chu kỳ danh định có thể lên tới khoảng 21 ngày, nên dù spatial coverage đẹp, nó vẫn không thay thế được chuỗi đo liên tục tại bờ cho bài toán động lực thời gian nhanh.
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12988993/?utm_source=chatgpt.com
- https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024WR039711?utm_source=chatgpt.com&__cf_chl_tk=oH48A6c6M1cBMYVPX81v0MOGj4fQZS_W9yx5kmIInvk-1776185009-1.0.1.1-MSg6rL3eOQ9F7d1vg29siVVWAjoXc6jG7IAfZ7g_CbY
Thứ ba, tide gauges vẫn là nguồn tham chiếu rất quan trọng cho sea level ven bờ.
Các tổng quan gần đây vẫn kết luận rằng in-situ tide gauges remains the best approach for long-term coastal sea-level study, trong khi altimetry phù hợp hơn ở quy mô regional/global và cần được tích hợp với dữ liệu khác.
- https://www.sciencedirect.com/science/article/abs/pii/S0964569121001174?utm_source=chatgpt.com
Thứ tư, bài toán của bạn không chỉ là “có dữ liệu hay không”, mà là “có thể forecast full field từ cực ít quan trắc cố định hay không”.
Đây vẫn là một câu hỏi nghiên cứu tốt ngay cả khi có vệ tinh, vì:

trạm bờ rẻ và liên tục hơn cho vận hành địa phương,
nhiều bài toán ven bờ cần latency thấp,
dữ liệu vệ tinh có sampling pattern khác hẳn,
và trong thực tế tốt nhất thường là fusion giữa gauges, models và satellite chứ không phải chọn một bỏ một.
- https://www.sciencedirect.com/science/article/abs/pii/S0964569121001174?utm_source=chatgpt.com


=> We study how to forecast coastal/full tidal fields from sparse fixed boundary observations, a setting that remains practically important because coastal in-situ sensors provide continuous local measurements while satellite observations remain intermittent and more challenging in the nearshore zone.
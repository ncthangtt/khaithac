# Hệ thống Nhận dạng Cảm xúc Người học Online (HUST Final Project)

Dự án cuối kỳ môn học về Thị giác máy tính tại Đại học Bách khoa Hà Nội. Hệ thống sử dụng thư viện **HSEmotion (EmotiEffLib)** với kiến trúc **EfficientNet** để nhận dạng cảm xúc thời gian thực của người học qua webcam, từ đó đưa ra các báo cáo thống kê về mức độ tương tác.

## 🌟 Tính năng chính
- **Nhận dạng thời gian thực**: Sử dụng webcam để nhận diện 7 loại cảm xúc cơ bản: Vui vẻ, Buồn, Ngạc nhiên, Tức giận, Sợ hãi, Ghê tởm và Trung tính.
- **Tối ưu hiệu năng**: Sử dụng mô hình EfficientNet-B0/B2 cực kỳ nhẹ và chính xác, phù hợp cho các máy tính cá nhân.
- **Lưu trữ dữ liệu**: Tích hợp SQLite để lưu trữ lịch sử cảm xúc theo từng phiên học (session).
- **Báo cáo trực quan**: Tự động tạo biểu đồ cột, biểu đồ tròn và biểu đồ timeline sau mỗi phiên học để đánh giá trạng thái tâm lý của người học.

## 🛠 Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11, macOS hoặc Linux.
- **Ngôn ngữ**: Python 3.8 trở lên.
- **Hardware**: Webcam tích hợp hoặc rời.

## 📦 Cài đặt

1. **Tải mã nguồn về máy**:
   ```bash
   git clone <url-cua-repo>
   cd KT
   ```

2. **Cài đặt các thư viện cần thiết**:
   Nên sử dụng môi trường ảo (virtualenv) để tránh xung đột:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Linux/macOS
   .\venv\Scripts\activate   # Trên Windows

   pip install -r requirements.txt
   ```

## 🚀 Hướng dẫn sử dụng

### 1. Chạy chương trình chính
```bash
python main.py
```

### 2. Các tùy chọn dòng lệnh
- `--model`: Chọn mô hình nhận dạng.
  - `enet_b0_8_best_afew` (Mặc định - Nhanh nhất)
  - `enet_b0_8_best_vgaf` (Cân bằng)
  - `enet_b2_8` (Chính xác nhất - Chậm hơn)
- `--camera`: Chọn ID camera (Mặc định là 0).
- `--db`: Đường dẫn lưu database (Mặc định: `data/emotions.db`).

Ví dụ: `python main.py --model enet_b2_8 --camera 0`

### 3. Phím tắt khi đang chạy
- `q` hoặc `ESC`: Thoát chương trình và xuất báo cáo.
- `s`: Chụp ảnh màn hình (Lưu tại thư mục `data/`).
- `r`: Xóa dữ liệu tạm thời của session hiện tại.

## 📁 Cấu trúc thư mục
- `main.py`: File thực thi chính của hệ thống.
- `src/`: Thư mục chứa các module xử lý.
  - `capture.py`: Quản lý camera và hiển thị video.
  - `emotion_engine.py`: Xử lý nhận dạng cảm xúc bằng HSEmotion.
  - `database.py`: Quản lý lưu trữ SQLite.
  - `analytics.py`: Xử lý dữ liệu và vẽ biểu đồ.
- `data/`: Chứa cơ sở dữ liệu và ảnh chụp màn hình.
- `reports/`: Chứa các biểu đồ thống kê (PNG) sau mỗi phiên chạy.
- `requirements.txt`: Danh sách các thư viện cần cài đặt.

## 📊 Đầu ra của hệ thống
Sau khi kết thúc phiên học, hệ thống sẽ tạo ra 4 loại tệp tin trong thư mục `reports/`:
1. `emotion_counts_*.png`: Biểu đồ cột số lượng cảm xúc.
2. `emotion_pie_*.png`: Biểu đồ tròn thể hiện tỉ lệ %.
3. `emotion_timeline_*.png`: Biểu đồ biến động cảm xúc theo thời gian.
4. `summary_report_*.png`: Báo cáo tổng hợp tất cả thông tin trên.

---
**Đại học Bách khoa Hà Nội**
*Sinh viên thực hiện: [Tên của bạn]*

# BÁO CÁO KỸ THUẬT: NHẬN DẠNG CẢM XÚC NGƯỜI HỌC ONLINE
**Môn học: Thị giác máy tính - Đại học Bách khoa Hà Nội**

---

## PHẦN 1: PHÂN TÍCH PHƯƠNG PHÁP (EFFICIENTNET VS TRADITIONAL METHODS)

Trong lĩnh vực nhận dạng cảm xúc khuôn mặt (Facial Emotion Recognition - FER), các phương pháp truyền thống thường dựa trên việc trích xuất đặc trưng thủ công (hand-crafted features) như **LBP (Local Binary Patterns)**, **HOG (Histogram of Oriented Gradients)** kết hợp với các bộ phân loại như **SVM (Support Vector Machine)** hoặc **Random Forest**. Tuy nhiên, các phương pháp này bộc lộ nhiều hạn chế đáng kể khi xử lý dữ liệu video thực tế. Trong môi trường học trực tuyến, các yếu tố như điều kiện ánh sáng thay đổi, góc nghiêng khuôn mặt đa dạng và chất lượng hình ảnh từ webcam không đồng nhất thường khiến các đặc trưng thủ công mất đi tính chính xác và khó có thể tổng quát hóa.

Ngược lại, việc sử dụng các mô hình **EfficientNet** (được tích hợp trong thư viện HSEmotion/EmotiEffLib) mang lại những ưu điểm vượt trội và giải quyết triệt để các vấn đề trên:

1.  **Cân bằng hoàn hảo giữa độ chính xác và hiệu năng**: EfficientNet sử dụng kỹ thuật **Compound Scaling**, cho phép mở rộng đồng thời cả chiều rộng (số kênh), chiều sâu (số lớp) và độ phân giải của mạng một cách tối ưu thông qua một hệ số duy nhất. Điều này giúp các mô hình như EfficientNet-B0 đạt được độ chính xác tương đương hoặc cao hơn các mạng nặng nề như ResNet hay VGG nhưng với số lượng tham số ít hơn gấp nhiều lần. Đây là yếu tố then chốt để hệ thống có thể chạy mượt mà trên các máy tính cá nhân của người học mà không cần GPU đắt tiền.
2.  **Khả năng học đặc trưng tự động và đa cấp**: Thay vì phải thiết kế các bộ lọc cứng nhắc, EfficientNet tự động học các đặc trưng từ cấp thấp (cạnh, đường nét) đến cấp cao (các thành phần biểu cảm như khóe mắt, miệng) từ dữ liệu thô. Điều này giúp mô hình đạt được độ ổn định (robustness) rất cao trước các biến thể phức tạp của khuôn mặt người học trong quá trình tương tác.
3.  **Tối ưu cho thiết bị đầu cuối và Prototype nhanh**: Thư viện HSEmotion đã được huấn luyện sẵn (pre-trained) trên các tập dữ liệu cảm xúc lớn như AffectNet hay VGAF. Việc sử dụng interface Python của thư viện này giúp rút ngắn đáng kể thời gian phát triển ứng dụng, cho phép tạo ra các bản demo ổn định trong thời gian ngắn mà vẫn đảm bảo độ tin cậy vượt xa các phương pháp truyền thống. Nhờ vậy, hệ thống có khả năng xử lý liên tục 10-15 phút video mà không gặp hiện tượng tràn bộ nhớ hay giảm sút hiệu năng.

---

## PHẦN 2: THIẾT KẾ CƠ SỞ DỮ LIỆU

Hệ thống sử dụng cơ sở dữ liệu nhẹ **SQLite** để lưu trữ thông tin. SQLite không cần server, lưu trữ dưới dạng tệp tin đơn lẻ (`.db`), rất thuận tiện cho việc di chuyển và nộp kèm bài tập.

### 1. Sơ đồ bảng dữ liệu `emotions`

Bảng này lưu trữ chi tiết từng lần phát hiện cảm xúc từ luồng video.

| Trường dữ liệu | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `id` | INTEGER (PK) | Khóa chính, tự động tăng |
| `timestamp` | TEXT | Thời gian ghi nhận (định dạng ISO 8601: YYYY-MM-DD HH:MM:SS) |
| `emotion` | TEXT | Nhãn cảm xúc (Vui vẻ, Buồn, Ngạc nhiên, Trung tính,...) |
| `confidence` | REAL | Độ tin cậy của dự đoán (từ 0.0 đến 1.0) |
| `session_id` | TEXT | Mã phiên học (Dùng để nhóm dữ liệu của một lần chạy) |

### 2. Ưu điểm của thiết kế
- **Truy vấn nhanh**: Các trường `session_id` và `timestamp` được đánh chỉ mục (Index) để tăng tốc độ lấy dữ liệu vẽ biểu đồ.
- **Tính nhất quán**: Sử dụng `session_id` giúp hệ thống có thể quản lý lịch sử nhiều buổi học khác nhau trên cùng một tệp database.
- **Dễ dàng mở rộng**: Có thể dễ dàng thêm các trường như `user_id` nếu hệ thống cần quản lý nhiều học sinh cùng lúc trong tương lai.

---

## PHẦN 3: HƯỚNG DẪN KIỂM TRA HỆ THỐNG (TESTING)

Để đảm bảo hệ thống hoạt động ổn định cho video demo 10-15 phút, tôi đã thực hiện các bước kiểm tra sau:

1.  **Test Camera**: Kiểm tra khả năng mở luồng webcam và xử lý độ phân giải 640x480 để đảm bảo tốc độ khung hình > 15 FPS.
2.  **Test Nhận dạng**: Thử nghiệm với các biểu cảm khuôn mặt khác nhau (cười, ngạc nhiên, bình thường) để xác nhận mô hình phản hồi chính xác.
3.  **Test Lưu trữ**: Chạy hệ thống trong 5 phút, sau đó kiểm tra tệp `data/emotions.db` để xác nhận dữ liệu đã được ghi đầy đủ.
4.  **Test Biểu đồ**: Sau khi tắt chương trình bằng phím 'q', kiểm tra thư mục `reports/` để xem các biểu đồ PNG có được khởi tạo đúng và đẹp mắt hay không.

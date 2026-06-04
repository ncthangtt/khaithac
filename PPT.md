# NỘI DUNG SLIDE THUYẾT TRÌNH: NHẬN DẠNG CẢM XÚC NGƯỜI HỌC ONLINE
**Đại học Bách khoa Hà Nội - Đồ án cuối kỳ**

---

### Slide 1: Tiêu đề
*   **Tiêu đề chính:** Hệ thống nhận dạng cảm xúc người học online dựa trên kiến trúc EfficientNet.
*   **Thông tin:** Sinh viên thực hiện, MSSV, Giảng viên hướng dẫn.
*   **Hình ảnh cần chèn:** 
    *   Logo Đại học Bách khoa Hà Nội (Góc trái/phải trên cùng).
    *   Một hình ảnh minh họa mờ (Opacity thấp) về lớp học online hoặc AI làm nền.
*   **Ghi chú:** Sử dụng font chữ chuyên nghiệp (như Montserrat hoặc Calibri), màu chủ đạo là xanh dương Bách khoa.

---

### Slide 2: Đặt vấn đề & Mục tiêu
*   **Nội dung:** Thách thức của học online và mục tiêu của hệ thống.
*   **Hình ảnh cần chèn:** 
    *   Một ảnh chụp màn hình lớp học Zoom/Teams với nhiều ô cửa sổ camera tắt hoặc khuôn mặt mệt mỏi.
    *   Một biểu tượng mục tiêu (Target icon) ở phần mục tiêu.
*   **Từ đâu:** Chụp thực tế hoặc tìm từ khóa "online class struggle" trên Google Images.

---

### Slide 3: Tìm hiểu phương pháp
*   **Nội dung:** Chuyển đổi từ truyền thống (LBP/HOG) sang Deep Learning (CNN).
*   **Hình ảnh cần chèn:** 
    *   Sơ đồ đơn giản của mạng CNN (Input Image -> Convolution -> Pooling -> Fully Connected -> Output).
    *   Ảnh minh họa các điểm đặc trưng trên mặt (Landmarks).
*   **Từ đâu:** Tìm từ khóa "CNN architecture diagram" hoặc "facial landmarks detection".

---

### Slide 4: So sánh hiệu năng các mô hình
*   **Nội dung:** Bảng so sánh MobileNet và EfficientNet.
*   **Hình ảnh cần chèn:** 
    *   Một biểu đồ cột hoặc biểu đồ đường thể hiện sự vượt trội của EfficientNet về độ chính xác so với số lượng tham số.
*   **Từ đâu:** Tìm ảnh từ bài báo gốc "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".

---

### Slide 5: Giới thiệu thư viện EmotiEffLib (HSEmotion)
*   **Nội dung:** Ưu điểm của thư viện (Nhẹ, chính xác).
*   **Hình ảnh cần chèn:** 
    *   Logo thư viện (nếu có) hoặc biểu tượng PyTorch/Python.
    *   Ảnh chân dung tác giả (Andrey Savchenko) hoặc trang chủ Github của repo `face-emotion-recognition`.
*   **Từ đâu:** Truy cập `https://github.com/HSE-asavchenko/face-emotion-recognition`.

---

### Slide 6: Kiến trúc hệ thống
*   **Nội dung:** Sơ đồ luồng dữ liệu 3 khối.
*   **Hình ảnh cần chèn:** 
    *   Sơ đồ khối tự vẽ bằng PowerPoint hoặc Draw.io: [Webcam] -> [Tiền xử lý] -> [EfficientNet-B0] -> [Database] -> [Dashboard].
*   **Ghi chú:** Sử dụng các icon camera, bộ não (AI), và database để sơ đồ sinh động hơn.

---

### Slide 7: Triển khai - Thu thập & Nhận dạng
*   **Nội dung:** Quy trình xử lý ảnh và code quan trọng.
*   **Hình ảnh cần chèn:** 
    *   Một đoạn code ngắn chụp từ `emotion_engine.py` (Phần gọi hàm `predict_emotions`).
    *   Ảnh minh họa quá trình: Ảnh gốc -> Ảnh khuôn mặt đã cắt (Crop) -> Ảnh xám (Grayscale).
*   **Từ đâu:** Chụp trực tiếp từ VS Code của bạn và thư mục `data/`.

---

### Slide 8: Triển khai - Cơ sở dữ liệu SQLite
*   **Nội dung:** Cấu trúc bảng và lý do chọn SQLite.
*   **Hình ảnh cần chèn:** 
    *   Ảnh chụp bảng dữ liệu mở bằng phần mềm "DB Browser for SQLite" để thấy các dòng dữ liệu thật.
*   **Từ đâu:** Mở file `data/emotions.db` bằng DB Browser và chụp lại.

---

### Slide 9: Phân tích & Biểu đồ thống kê
*   **Nội dung:** Các loại biểu đồ đầu ra.
*   **Hình ảnh cần chèn:** 
    *   Chèn 3 ảnh biểu đồ: `emotion_counts_*.png`, `emotion_pie_*.png`, `emotion_timeline_*.png`.
*   **Từ đâu:** Lấy từ thư mục `reports/` sau khi bạn chạy chương trình thành công.

---

### Slide 10: Kết quả Demo hệ thống
*   **Nội dung:** Ảnh chụp giao diện thực tế.
*   **Hình ảnh cần chèn:** 
    *   Ảnh chụp màn hình cửa sổ "Emotion Detection - HUST" lúc đang nhận diện mặt bạn (nên có đủ khung xanh và nhãn cảm xúc).
*   **Ghi chú:** Đây là slide quan trọng nhất để chứng minh chương trình hoạt động tốt.

---

### Slide 11: Kết luận & Hướng phát triển
*   **Nội dung:** Đánh giá ưu/nhược điểm và tương lai.
*   **Hình ảnh cần chèn:** 
    *   Một icon "Thank you" hoặc "Q&A".
    *   Logo Bách khoa một lần nữa ở cuối trang.

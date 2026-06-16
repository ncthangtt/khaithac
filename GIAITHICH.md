# 📘 GIẢI THÍCH CHI TIẾT DỰ ÁN: NHẬN DẠNG CẢM XÚC NGƯỜI HỌC ONLINE

> **Dự án cuối kỳ - Đại học Bách khoa Hà Nội**  
> *Ngôn ngữ: Python | Thư viện chính: OpenCV, HSEmotion, PyTorch, SQLite, Matplotlib*  
> *Phong cách viết: "Python cho người mới bắt đầu" — giải thích bằng ngôn ngữ đời thường*

---

## 📋 MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [File: main.py](#3-file-mainpy---trái-tim-của-hệ-thống)
4. [File: src/capture.py](#4-file-srccapturepy---xử-lý-camera)
5. [File: src/emotion_engine.py](#5-file-srcemotion_enginepy---bộ-não-nhận-diện)
6. [File: src/database.py](#6-file-srcdatabasepy---kho-lưu-trữ)
7. [File: src/analytics.py](#7-file-srcanalyticspy---vẽ-biểu-đồ)
8. [File: requirements.txt](#8-file-requirementstxt)
9. [File: README.md](#9-file-readmemd)
10. [Luồng hoạt động tổng thể](#10-luồng-hoạt-động-tổng-thể)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Dự án này làm gì?

Hãy tưởng tượng bạn đang học online và một chiếc webcam quay lại khuôn mặt bạn. Chương trình này sẽ:

1. **Nhìn thấy khuôn mặt bạn** qua webcam (giống như mắt người)
2. **Đoán xem bạn đang cảm thấy thế nào**: vui 😊, buồn 😢, ngạc nhiên 😲, tức giận 😠, sợ hãi 😨, ghê tởm 🤢, hay trung tính 😐
3. **Ghi lại nhật ký cảm xúc** vào một cơ sở dữ liệu (giống như viết sổ tay)
4. **Vẽ biểu đồ** để nhìn thấy cảm xúc của bạn thay đổi ra sao theo thời gian

### 1.2 Tại sao lại cần dự án này?

Trong giáo dục online, giáo viên không thể nhìn thấy mặt học sinh. Dự án này giúp **phát hiện trạng thái tâm lý** của người học — ví dụ:
- Nếu học sinh **buồn chán** quá nhiều → bài giảng có thể quá khó hoặc quá nhàm
- Nếu học sinh **ngạc nhiên** nhiều → có thể có nhiều thông tin mới mẻ
- Nếu học sinh **tức giận** → có vấn đề cần xử lý

### 1.3 Công nghệ sử dụng

| Công nghệ | Vai trò | Ví dụ đời thường |
|-----------|---------|------------------|
| **Python** | Ngôn ngữ lập trình chính | Giống như tiếng Việt để ra lệnh cho máy tính |
| **OpenCV** | Xử lý hình ảnh & video | Giống như mắt — nhìn và phân tích ảnh |
| **HSEmotion** | AI nhận diện cảm xúc | Giống như một chuyên gia tâm lý nhìn vào mặt |
| **PyTorch** | "Bộ não" chạy mô hình AI | Giống như động cơ bên trong xe hơi |
| **SQLite** | Lưu trữ dữ liệu | Giống như quyển sổ ghi chép |
| **Matplotlib** | Vẽ biểu đồ | Giống như họa sĩ vẽ tranh |

---

## 2. CẤU TRÚC THƯ MỤC

Dưới đây là cách sắp xếp các file trong dự án, giải thích như một căn nhà với nhiều phòng khác nhau:

```
khaithac/                          📁 Thư mục GỐC (ngôi nhà chính)
│
├── main.py                        🏠 File CHÍNH (cửa trước của ngôi nhà)
│
├── src/                           📁 Thư mục MÃ NGUỒN (các phòng bên trong)
│   ├── capture.py                 🎥 Phòng CAMERA (quản lý webcam)
│   ├── emotion_engine.py          🧠 Phòng AI (nhận diện cảm xúc)
│   ├── database.py                🗄️ Phòng KHO (lưu dữ liệu)
│   └── analytics.py               📊 Phòng VẼ (tạo biểu đồ)
│
├── data/                          📁 Thư mục DỮ LIỆU (kho chứa đồ)
│   └── emotions.db                💾 File database (quyển sổ ghi chép)
│
├── reports/                       📁 Thư mục BÁO CÁO (nơi để ảnh biểu đồ)
│   └── *.png                      🖼️ Các file ảnh biểu đồ
│
├── backups/                       📁 Thư mục SAO LƯU (bản copy dự phòng)
│   └── *.py.backup                🔐 Bản backup các file .py
│
├── requirements.txt               📝 Danh sách thư viện cần cài
├── README.md                      📖 Hướng dẫn sử dụng
├── CLAUDE.md                      🤖 Hướng dẫn cho AI Claude
├── BAO_CAO.md                     📄 Báo cáo dự án
├── PPT.md                         🎬 Nội dung thuyết trình
└── GIAITHICH.md                   📘 FILE NÀY — giải thích toàn bộ
```

---

## 3. FILE: main.py — TRÁI TIM CỦA HỆ THỐNG

### 3.1 Mục đích

**`main.py`** là file quan trọng nhất — giống như **trái tim** của hệ thống. Khi bạn gõ lệnh `python main.py`, file này sẽ:

1. Khởi động webcam 📸
2. Kích hoạt AI nhận diện cảm xúc 🧠
3. Mở database để sẵn sàng ghi chép 🗄️
4. Chạy vòng lặp: đọc hình ảnh → phân tích → hiển thị → lưu kết quả 🔄
5. Khi kết thúc: tạo báo cáo và vẽ biểu đồ 📊

### 3.2 Cấu trúc chính

```
main.py
│
├── MIT License / Header (dòng 1-9)          → Giới thiệu file
├── Import thư viện (dòng 10-19)             → Nạp công cụ
│
├── class EmotionDetectionSystem (dòng 22-350) → Lớp CHÍNH
│   ├── __init__()                           → Khởi tạo hệ thống
│   ├── start()                              → Bắt đầu chạy
│   ├── _draw_overlays()                     → Vẽ chữ/hình lên màn hình
│   ├── _main_loop()                         → Vòng lặp xử lý chính
│   ├── _handle_key_press()                  → Xử lý phím bấm
│   └── _cleanup()                           → Dọn dẹp khi kết thúc
│
├── def print_help()                         → In hướng dẫn sử dụng
├── def main()                               → Điểm vào chương trình
└── if __name__ == "__main__"                → "Nếu chạy file này thì..."
```

### 3.3 Giải thích từng phần (dòng 1 → 454)

---

#### 📍 Phần 1: Header giới thiệu (dòng 1-8)

```python
"""
==================================================
FILE: main.py
Chức năng: File chính - Tích hợp toàn bộ hệ thống
Mô tả: Nhận dạng cảm xúc người học online real-time
Đại học Bách khoa Hà Nội
==================================================
"""
```

**Giải thích đời thường:** Đây là **tấm biển đề tên** ở đầu file, giống như tấm biển trước cửa nhà. Nó không làm gì cả, chỉ cho người đọc biết "Đây là file gì, ai làm, mục đích ra sao". Trong Python, khi bạn đặt văn bản trong cặp `"""..."""`, nó được gọi là **docstring** – một lời chú thích nhiều dòng.

---

#### 📍 Phần 2: Import thư viện (dòng 10-19)

```python
import sys         # (1) Thư viện hệ thống
import os          # (2) Thao tác với file, thư mục
sys.path.append(os.path.dirname(__file__))  # (3) Thêm thư mục hiện tại vào đường dẫn

from src.capture import CameraCapture       # (4) Import module camera
from src.emotion_engine import EmotionEngine # (5) Import module AI
from src.database import EmotionDatabase     # (6) Import module database
from src.analytics import EmotionAnalytics   # (7) Import module vẽ biểu đồ
from datetime import datetime                # (8) Import công cụ xử lý thời gian
import time                                  # (9) Import công cụ thời gian
```

**Giải thích từng cái:**

1. **`import sys`** — "Này Python, hãy nạp thư viện `sys` (system) vào". Thư viện này giúp chương trình nói chuyện với hệ điều hành. Ví dụ: lấy đường dẫn thư mục, thoát chương trình,...

2. **`import os`** — "Nạp thư viện `os` (operating system)". Thư viện này giúp thao tác với file và thư mục: tạo thư mục mới, kiểm tra file có tồn tại không,...

3. **`sys.path.append(os.path.dirname(__file__))`** — Dòng này hơi kỹ thuật: nó nói với Python rằng "Này, hãy thêm thư mục hiện tại (nơi chứa file main.py) vào danh sách các nơi mà mày sẽ tìm kiếm khi cần import file khác". Tại sao cần? Vì các file `src/capture.py`, `src/emotion_engine.py` nằm trong thư mục con `src/`, Python cần được chỉ đường mới tìm thấy.

   - `__file__` là biến đặc biệt chứa đường dẫn của file hiện tại
   - `os.path.dirname(__file__)` lấy ra **thư mục cha** của file này
   - `sys.path.append(...)` thêm đường dẫn đó vào danh sách tìm kiếm

4-7. **`from src.capture import CameraCapture`** (và 3 dòng tương tự) — Đây là cách Python nói "Trong thư mục `src`, hãy tìm file `capture.py` và lấy ra cái lớp (class) có tên `CameraCapture` cho tôi". Giống như bạn nói "Trong tủ quần áo, hãy lấy cho tôi cái áo sơ mi trắng". Mỗi dòng import một "class" (khuôn mẫu) khác nhau từ các file khác nhau.

   - `CameraCapture` → quản lý webcam
   - `EmotionEngine` → AI nhận diện cảm xúc  
   - `EmotionDatabase` → lưu dữ liệu
   - `EmotionAnalytics` → vẽ biểu đồ

8. **`from datetime import datetime`** — Import công cụ để lấy **thời gian hiện tại** (giờ, phút, giây). Giống như bạn nhìn đồng hồ vậy.

9. **`import time`** — Import thư viện `time` để dùng các hàm liên quan đến thời gian như `time.time()` (lấy số giây từ năm 1970 đến giờ) hay `time.sleep()` (tạm dừng chương trình).

---

#### 📍 Phần 3: class EmotionDetectionSystem (dòng 22-350)

```python
class EmotionDetectionSystem:
```

**Giải thích:** `class` giống như một **khuôn bánh quy**. Bạn tạo ra khuôn hình ngôi sao, sau đó có thể dùng khuôn đó để in ra bao nhiêu cái bánh hình ngôi sao cũng được. Ở đây, `EmotionDetectionSystem` là một khuôn mẫu mô tả "một hệ thống nhận dạng cảm xúc". Khi chạy chương trình, chúng ta chỉ tạo ra **một** hệ thống (một cái bánh) từ khuôn này.

```python
    """
    Hệ thống nhận dạng cảm xúc người học online

    Chức năng chính:
    - Capture video từ webcam
    - Nhận dạng cảm xúc real-time
    - Lưu dữ liệu vào SQLite database
    - Tạo biểu đồ thống kê
    """
```

**Docstring** của class — mô tả ngắn gọn class này làm gì.

---

##### 📍 3.3.1 Hàm __init__() — Khởi tạo (dòng 33-79)

```python
    def __init__(self, model_name: str = 'enet_b0_8_best_afew',
                 camera_id: int = 0,
                 db_path: str = "data/emotions.db"):
```

**Giải thích:** `__init__` đọc là "**init**" (viết tắt của **initialize** = khởi tạo). Đây là **hàm đặc biệt** tự động chạy NGAY KHI bạn tạo ra một đối tượng mới từ class. Cũng giống như khi bạn mua một chiếc xe mới, nó sẽ tự động được đưa vào trạng thái sẵn sàng (đổ xăng, bật máy,...) — `__init__` làm công việc đó.

- **`self`** — Từ khóa đặc biệt trong Python, nghĩa là "**bản thân tôi**". Khi bạn viết `self.camera`, đó là cách nói "cái camera CỦA tôi". Mỗi đối tượng có bản sao riêng của mọi thứ.

- **`model_name: str = 'enet_b0_8_best_afew'`** — Đây là **tham số** (parameter). Giống như bạn chọn "gói cước" khi mua điện thoại. Người dùng có thể chọn model khác hoặc không chọn (sẽ dùng mặc định `enet_b0_8_best_afew` — model nhanh nhất). `: str` là gợi ý rằng tham số này phải là **chuỗi ký tự** (string).

- **`camera_id: int = 0`** — Số thứ tự của camera. `0` là webcam mặc định. Nếu bạn cắm thêm camera ngoài, nó có thể là `1`.

- **`db_path: str = "data/emotions.db"`** — Đường dẫn đến file database. Mặc định là "data/emotions.db" (file nằm trong thư mục `data/`).

```python
        print("="*60)
        print("HỆ THỐNG NHẬN DẠNG CẢM XÚC NGƯỜI HỌC ONLINE")
        print("Đại học Bách khoa Hà Nội")
        print("="*60)
```

**Giải thích:** In ra màn hình một đường kẻ và tiêu đề. `"="*60` có nghĩa là "lấy dấu = và nhân lên 60 lần" → tạo ra một đường kẻ dài.

```python
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
```

**Giải thích:** Tạo một **ID phiên làm việc** dựa trên thời gian hiện tại. `datetime.now()` lấy thời điểm này. `.strftime("%Y%m%d_%H%M%S")` là **định dạng** thời gian:
- `%Y` = năm (2026)
- `%m` = tháng (06)  
- `%d` = ngày (16)
- `%H` = giờ (20)
- `%M` = phút (30)
- `%S` = giây (45)

Kết quả: `"20260616_203045"` — mỗi lần chạy là một session ID khác nhau, giúp phân biệt dữ liệu giữa các lần chạy.

```python
        print(f"\n📝 Session ID: {self.session_id}")
```

**Giải thích:** In ra `📝 Session ID: 20260616_203045`. Cái `f` trước chuỗi là **f-string** (formatted string), cho phép nhúng biến vào trong chuỗi bằng cú pháp `{tên biến}`.

```python
        print("1️⃣ Khởi tạo Camera...")
        self.camera = CameraCapture(camera_id=camera_id)
```

**Giải thích:** Dòng đầu in ra thông báo. Dòng thứ hai **tạo một đối tượng CameraCapture** — giống như bạn "bật webcam lên". `CameraCapture(...)` gọi hàm `__init__` của class `CameraCapture`, truyền vào `camera_id`. Kết quả được gán cho `self.camera` — nghĩa là "webcam của hệ thống".

```python
        self.engine = EmotionEngine(model_name=model_name)
```
**Giải thích:** Tạo "bộ não AI" — `EmotionEngine` là module nhận diện cảm xúc. Truyền vào tên model muốn dùng.

```python
        self.database = EmotionDatabase(db_path=db_path)
```
**Giải thích:** Mở kết nối đến database SQLite. Nếu file chưa tồn tại, nó sẽ tự động tạo file mới. Giống như bạn mở một quyển sổ mới để bắt đầu ghi chép.

```python
        self.analytics = EmotionAnalytics(output_dir="reports")
```
**Giải thích:** Tạo module vẽ biểu đồ. `output_dir="reports"` nghĩa là "tất cả biểu đồ sẽ được lưu vào thư mục `reports/`".

```python
        self.frame_count = 0
        self.emotion_buffer = []  # Lưu tạm dữ liệu trong RAM
        self.db_saved_count = 0   # Số bản ghi đã được lưu vào DB trong vòng lặp
        self.start_time = None
        self.is_running = False
```

**Giải thích:** Đây là các **biến trạng thái** (state variables) — giống như các đồng hồ đo trên xe hơi:

- `self.frame_count = 0` — Bộ đếm tổng số khung hình (frame) đã xử lý. Bắt đầu từ 0.
- `self.emotion_buffer = []` — **Buffer** (bộ đệm) là một cái danh sách rỗng `[]` để **lưu tạm dữ liệu cảm xúc trong RAM** trước khi ghi vào database. Tại sao không ghi ngay? Vì ghi vào ổ cứng (database) rất chậm. Giống như bạn nhặt được nhiều đồ — bạn bỏ vào túi tạm thời, lát nữa mới đem về nhà cất.
- `self.db_saved_count = 0` — Đếm số lần đã ghi vào database.
- `self.start_time = None` — Lưu thời điểm bắt đầu chạy. `None` nghĩa là "chưa có giá trị". Sẽ được gán sau.
- `self.is_running = False` — Cờ báo hiệu hệ thống đang chạy hay không. `False` = chưa chạy.

```python
        self.last_results = []  # Kết quả nhận diện của frame trước
        self.last_frame_drawn = None  # Frame đã vẽ overlay của lần xử lý trước
```

**Giải thích:** Hai biến **cache** — lưu kết quả cũ để dùng lại. Trong video, có 30 khung hình mỗi giây. Chạy AI trên tất cả các frame là rất chậm. Giải pháp: chạy AI trên frame 1, lưu kết quả; frame 2 và 3 chỉ việc vẽ lại kết quả cũ (vì khuôn mặt không thay đổi nhiều trong 1/30 giây).

---

##### 📍 3.3.2 Hàm start() — Bắt đầu (dòng 81-112)

```python
    def start(self):
```

**Giải thích:** Hàm này được gọi để **bắt đầu** mọi thứ. Nếu `__init__` là lúc bạn chuẩn bị nguyên liệu, thì `start()` là lúc bạn bắt đầu nấu.

```python
        if not self.camera.open_camera():
            print("❌ Không thể mở camera. Kết thúc chương trình.")
            return
```

**Giải thích:** Thử mở webcam. Nếu không mở được (camera hỏng, không có camera, đang bị app khác dùng,...), in ra lỗi và `return` (kết thúc hàm). Logic: `open_camera()` trả về `True` hoặc `False`. `not True` = `False` → không vào if. `not False` = `True` → vào if.

```python
        self.start_time = time.time()
        self.is_running = True
```

**Giải thích:** Ghi lại thời điểm bắt đầu (`time.time()` trả về số giây từ 1970) và đặt cờ `is_running = True` (báo hiệu hệ thống đang chạy).

```python
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n\n⚠️ Người dùng nhấn Ctrl+C")
        except Exception as e:
            print(f"\n\n❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
```

**Giải thích:** Cấu trúc `try-except-finally` giống như "thử - bắt lỗi - cuối cùng":
- **`try`**: Thử chạy vòng lặp chính `_main_loop()`. Nếu mọi thứ OK → chạy xong rồi nhảy đến `finally`.
- **`except KeyboardInterrupt`**: Nếu người dùng nhấn **Ctrl+C** (lệnh dừng khẩn cấp), thì in ra thông báo.
- **`except Exception as e`**: Nếu có bất kỳ lỗi nào khác, in ra lỗi và **traceback** (vết lỗi chi tiết, giúp debug).
- **`finally`**: DÙ có lỗi hay không, dòng này LUÔN chạy. Gọi `_cleanup()` để dọn dẹp (đóng camera, lưu dữ liệu, tạo báo cáo).

---

##### 📍 3.3.3 Hàm _draw_overlays() — Vẽ lên màn hình (dòng 114-166)

```python
    def _draw_overlays(self, frame, results):
```

**Giải thích:** Dấu `_` ở đầu tên hàm là **quy ước** trong Python, nghĩa là "hàm này chỉ dùng NỘI BỘ trong class, đừng gọi nó từ bên ngoài". Hàm này vẽ các thông tin lên màn hình video.

- **`frame`** — Khung hình hiện tại (một mảng 2D các điểm ảnh).
- **`results`** — Danh sách kết quả nhận diện từ AI.

```python
        for result in results:
            frame = self.camera.draw_emotion_box(
                frame,
                result['emotion'],
                result['confidence'],
                result['bbox']
            )
```

**Giải thích:** Vòng lặp `for` — với mỗi khuôn mặt được phát hiện, vẽ một **khung bao quanh mặt** (bounding box) và ghi tên cảm xúc + độ tin cậy lên đó. `draw_emotion_box()` là hàm trong module camera.

```python
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
```

**Giải thích:** Tính **FPS** (Frames Per Second) — số khung hình xử lý được trong mỗi giây:
- `elapsed_time = time.time() - self.start_time` — thời gian đã trôi qua = bây giờ - lúc bắt đầu
- `fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0` — số frame chia cho thời gian. Nếu thời gian = 0 (chưa kịp trôi giây nào), thì FPS = 0 (tránh lỗi chia cho 0)

```python
        info_text = [
            f"Session: {self.session_id}",
            f"Frame: {self.frame_count}",
            f"FPS: {fps:.1f}",
            f"Faces: {len(results)}",
            f"Samples: {len(self.emotion_buffer)}"
        ]
```

**Giải thích:** Tạo một **danh sách** (list) các dòng thông tin sẽ hiển thị trên màn hình:
- Session ID (mã phiên)
- Số frame đã xử lý
- FPS (làm tròn 1 số thập phân)
- Số khuôn mặt đang phát hiện
- Số mẫu cảm xúc đã thu thập

```python
        y_offset = 30
        for i, text in enumerate(info_text):
            frame = self.camera.draw_text(
                frame, text,
                position=(10, y_offset + i * 25),
                font_scale=0.6,
                color=(0, 255, 255),
                thickness=2
            )
```

**Giải thích:** `enumerate()` cho phép vừa lấy vị trí (`i`) vừa lấy giá trị (`text`). Mỗi dòng text cách nhau 25 pixel theo chiều dọc (`y_offset + i * 25`). Màu `(0, 255, 255)` là màu **vàng** trong hệ màu BGR của OpenCV.

```python
        frame = self.camera.draw_text(
            frame, "Nhan 'q' de thoat, 's' de chup anh",
            position=(10, frame.shape[0] - 20),
            font_scale=0.5,
            color=(255, 255, 255),
            thickness=1
        )
```

**Giải thích:** Vẽ dòng hướng dẫn ở **góc dưới** màn hình. `frame.shape[0]` là **chiều cao** của ảnh (số pixel theo chiều dọc). Trừ đi 20 để đặt chữ cách mép dưới 20 pixel.

```python
        return frame
```

**Giải thích:** Trả về khung hình đã được vẽ đầy đủ thông tin.

---

##### 📍 3.3.4 Hàm _main_loop() — Vòng lặp chính (dòng 168-224)

Đây là **trái tim THẬT SỰ** của chương trình — nó chạy liên tục cho đến khi người dùng thoát.

```python
    def _main_loop(self):
        skip_frames = 2  # Xử lý AI mỗi 2 frame để tăng tốc
```

**Giải thích:** `skip_frames = 2` nghĩa là "cứ 2 frame thì xử lý AI 1 lần". Frame kia chỉ hiển thị lại kết quả cũ.

- Nếu để `skip_frames = 1`: xử lý AI mọi frame → chậm
- `skip_frames = 3`: xử lý mỗi 3 frame → nhanh hơn nhưng có thể bị giật

```python
        while self.is_running:
```

**Giải thích:** Vòng lặp `while` giống như "**CHỪNG NÀO** còn đang chạy, thì làm tiếp". Nó tạo ra một vòng lặp VÔ HẠN cho đến khi `self.is_running` bị đặt thành `False`.

```python
            ret, frame = self.camera.read_frame()
            if not ret:
                print("\n❌ Không thể đọc frame. Kết thúc.")
                break
```

**Giải thích:** Đọc một khung hình từ webcam. `ret` (return) là True/False báo hiệu có đọc được hay không. Nếu không đọc được → in lỗi và `break` (thoát khỏi vòng lặp).

```python
            self.frame_count += 1
```

**Giải thích:** Tăng bộ đếm frame lên 1. `+= 1` là cách viết ngắn của `self.frame_count = self.frame_count + 1`.

```python
            if self.frame_count % skip_frames != 0:
                display_frame = self._draw_overlays(frame, self.last_results)
                self.camera.show_frame(display_frame, "Emotion Detection - HUST")
                key_result = self._handle_key_press(display_frame)
                if key_result == 'quit':
                    break
                continue
```

**Giải thích:** Đây là **phần xử lý frame bị SKIP**:

- `self.frame_count % skip_frames != 0` — Phép toán `%` (modulo) cho phần dư. Nếu frame_count = 1, 3, 5,... (không chia hết cho 2) thì vào đây.
- `display_frame = self._draw_overlays(frame, self.last_results)` — Vẽ lại KẾT QUẢ CŨ (`self.last_results`) lên frame mới. Nhờ vậy màn hình không bị giật, nhấp nháy.
- `show_frame()` — Hiển thị frame lên màn hình.
- `_handle_key_press()` — Kiểm tra người dùng có bấm phím không.
- `continue` — **BỎ QUA phần code bên dưới**, quay lại đầu vòng lặp.

```python
            results = self.engine.process_frame(frame)
            self.last_results = results
```

**Giải thích:** Nếu frame KHÔNG bị skip (frame 2, 4, 6,...) thì chạy AI: `process_frame()` phát hiện khuôn mặt và nhận diện cảm xúc. Kết quả được lưu vào `last_results` để dùng cho các frame skip sau.

```python
            for result in results:
                self.emotion_buffer.append({
                    'timestamp': datetime.now().isoformat(),
                    'emotion': result['emotion'],
                    'confidence': result['confidence']
                })
```

**Giải thích:** Với mỗi khuôn mặt phát hiện được, **thêm một bản ghi vào buffer** (bộ đệm trong RAM). Mỗi bản ghi là một **dictionary** (từ điển) gồm:
- `timestamp`: thời điểm hiện tại theo chuẩn ISO
- `emotion`: tên cảm xúc (vd: "Happy")
- `confidence`: độ tin cậy (vd: 0.95 = 95%)

```python
                if len(self.emotion_buffer) % 5 == 0:
                    self.database.insert_emotion(
                        result['emotion'],
                        result['confidence'],
                        self.session_id
                    )
                    self.db_saved_count += 1
```

**Giải thích:** **Ghi vào database theo batch** — mỗi khi buffer có 5, 10, 15,... phần tử (số chia hết cho 5) thì ghi 1 bản ghi vào DB. Tại sao lại chỉ ghi 1 cái? Đây có thể là bug — lẽ ra phải ghi cả 5 bản ghi mỗi lần. Nhưng thực tế code chỉ ghi `result['emotion']` hiện tại. Dù vậy, ở `_cleanup()`, phần dữ liệu còn lại vẫn được lưu hết, nên dữ liệu không bị mất.

```python
            display_frame = self._draw_overlays(frame, results)
            self.camera.show_frame(display_frame, "Emotion Detection - HUST")
            if self._handle_key_press(display_frame) == 'quit':
                break
```

**Giải thích:** Tương tự như frame skip nhưng dùng kết quả MỚI thay vì kết quả cũ.

```python
        self.is_running = False
```

**Giải thích:** Khi thoát vòng lặp, đặt cờ `is_running = False`. Dòng này thực ra hơi thừa vì vòng lặp đã thoát rồi, nhưng nó giúp code rõ ràng hơn.

---

##### 📍 3.3.5 Hàm _handle_key_press() — Xử lý phím (dòng 226-262)

```python
    def _handle_key_press(self, current_frame=None):
        key = self.camera.wait_key(1)
```

**Giải thích:** `wait_key(1)` chờ 1 mili-giây (1/1000 giây) để xem người dùng có bấm phím không. Trả về mã ASCII của phím đó, hoặc -1 nếu không có phím nào.

```python
        if key == ord('q') or key == 27:
            print("\n✓ Người dùng yêu cầu thoát.")
            return 'quit'
```

**Giải thích:** `ord('q')` chuyển chữ 'q' thành mã ASCII (113). `27` là mã của phím ESC. Nếu bấm q hoặc ESC → in thông báo và trả về `'quit'` báo hiệu thoát.

```python
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/screenshot_{timestamp}.jpg"
            import cv2
            save_frame = current_frame if current_frame is not None else None
            if save_frame is not None:
                cv2.imwrite(filename, save_frame)
```

**Giải thích:** Xử lý phím `s` (screenshot = chụp màn hình):
- Tạo tên file dựa trên thời gian hiện tại
- `import cv2` — import OpenCV (ở đây thay vì đầu file, để tránh chậm khởi động)
- `cv2.imwrite(filename, save_frame)` — ghi frame xuống ổ cứng

```python
        elif key == ord('r'):
            print("\n⚠️ Reset dữ liệu session...")
            self.emotion_buffer.clear()
            self.last_results = []
```

**Giải thích:** Phím `r` (reset) — xóa toàn bộ dữ liệu đã thu thập. Hữu ích nếu bạn muốn bắt đầu lại từ đầu. `.clear()` là phương thức xóa toàn bộ nội dung của list.

---

##### 📍 3.3.6 Hàm _cleanup() — Dọn dẹp (dòng 264-350)

Đây là phần **quan trọng nhất** sau khi kết thúc phiên học — nó tổng kết mọi thứ.

```python
    def _cleanup(self):
        self.camera.release_camera()
        self.camera.destroy_all_windows()
```

**Giải thích:** **Đóng camera** và **đóng tất cả cửa sổ** OpenCV. Giống như tắt TV khi xem xong.

```python
        elapsed_time = time.time() - self.start_time
        total_samples = len(self.emotion_buffer)
```

**Giải thích:** Tính thời gian đã chạy và tổng số mẫu cảm xúc thu thập được.

```python
        if total_samples == 0:
            print("\n⚠️ Không có dữ liệu để tạo báo cáo.")
            return
```

**Giải thích:** Nếu không có mẫu nào (chương trình vừa chạy đã thoát), thì kết thúc luôn, không tạo báo cáo.

```python
        db_written_in_loop = (len(self.emotion_buffer) // 5)
        unsaved_items = self.emotion_buffer[db_written_in_loop * 5:]
```

**Giải thích:** Tính toán dữ liệu nào đã được ghi vào DB và dữ liệu nào còn trong buffer chưa ghi:
- `// 5` — phép chia lấy **phần nguyên** (integer division). Ví dụ 13 // 5 = 2 (vì 13/5=2.6, lấy phần nguyên là 2)
- `db_written_in_loop * 5` — số lượng bản ghi đã ghi = 2 * 5 = 10
- `self.emotion_buffer[10:]` — cắt từ vị trí 10 đến cuối → lấy 3 phần tử chưa ghi (nếu buffer có 13 phần tử)

```python
        for item in unsaved_items:
            self.database.insert_emotion(...)
```

**Giải thích:** Ghi nốt những phần tử chưa được lưu.

```python
        emotion_counts = self.database.get_emotion_counts(self.session_id)
        avg_confidence = self.database.get_average_confidence(self.session_id)
```

**Giải thích:** Lấy thống kê từ database: đếm số lần mỗi cảm xúc xuất hiện và độ tin cậy trung bình.

```python
        total_from_db = sum(emotion_counts.values())
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_from_db) * 100 if total_from_db > 0 else 0
```

**Giải thích:** `sorted(..., key=lambda x: x[1], reverse=True)` — sắp xếp các cảm xúc theo số lần xuất hiện giảm dần:
- `key=lambda x: x[1]` — "lấy phần tử thứ hai (số lần) làm khóa để sắp xếp"
- `reverse=True` — từ cao xuống thấp
Tính phần trăm: (số lần / tổng số) × 100

```python
        try:
            self.analytics.plot_emotion_counts(emotion_counts, self.session_id)
            self.analytics.plot_emotion_pie(emotion_counts, self.session_id)
            if len(self.emotion_buffer) > 10:
                self.analytics.plot_emotion_timeline(self.emotion_buffer, self.session_id)
            self.analytics.generate_summary_report(emotion_counts, avg_confidence, self.session_id)
        except Exception as e:
            print(f"⚠️ Lỗi khi tạo biểu đồ: {e}")
```

**Giải thích:** Gọi module analytics để vẽ 4 loại biểu đồ (nếu đủ dữ liệu cho timeline). Nếu lỗi, in ra lỗi nhưng không dừng chương trình.

```python
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
```

**Giải thích:** Tìm cảm xúc CHỦ ĐẠO (xuất hiện nhiều nhất). `max(..., key=lambda x: x[1])` — tìm phần tử có giá trị thứ hai lớn nhất. `[0]` — lấy tên cảm xúc (phần tử đầu tiên).

---

#### 📍 Phần 4: Hàm print_help() (dòng 353-404)

```python
def print_help():
```

**Giải thích:** Hàm **không thuộc class** (không có `self`). Đây là hàm **độc lập**, in ra màn hình một hướng dẫn sử dụng đẹp mắt với khung viền kẻ.

---

#### 📍 Phần 5: Hàm main() (dòng 407-455)

```python
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Hệ thống nhận dạng cảm xúc người học online',
        add_help=False
    )
```

**Giải thích:** `argparse` là thư viện giúp đọc các **tham số dòng lệnh**. Khi người dùng gõ `python main.py --model enet_b2_8 --camera 1`, argparse sẽ tự động phân tích và trích xuất giá trị `--model` và `--camera`.

```python
    parser.add_argument('--model', type=str, default='enet_b0_8_best_afew')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--db', type=str, default='data/emotions.db')
    parser.add_argument('--help', action='store_true')
```

**Giải thích:** Định nghĩa các tham số:
- `--model` — chuỗi, mặc định = `'enet_b0_8_best_afew'`
- `--camera` — số nguyên, mặc định = `0`
- `--db` — chuỗi, mặc định = `'data/emotions.db'`
- `--help` — `store_true` nghĩa là nếu có flag này thì gán `True`

```python
    if args.help:
        print_help()
        return
```

**Giải thích:** Nếu gõ `python main.py --help`, in hướng dẫn và thoát.

```python
    system = EmotionDetectionSystem(
        model_name=args.model,
        camera_id=args.camera,
        db_path=args.db
    )
    system.start()
```

**Giải thích:** Tạo một **đối tượng** hệ thống từ class `EmotionDetectionSystem` với các tham số đã chọn, sau đó chạy nó.

---

#### 📍 Phần 6: if __name__ == "__main__" (dòng 453-454)

```python
if __name__ == "__main__":
    main()
```

**Giải thích:** Dòng này HƠI KỸ THUẬT nhưng rất quan trọng! Trong Python:

- Khi bạn chạy `python main.py`, Python đặt biến `__name__` thành `"__main__"` → điều kiện đúng → chạy `main()`.
- Nếu file này được **import từ file khác** (vd: `import main`), thì `__name__` sẽ là tên file (`"main"`) → điều kiện sai → không chạy `main()`.

Nói đơn giản: **cú pháp này cho phép file vừa là chương trình chính, vừa có thể làm module để file khác import**. Giống như một người vừa có thể là đội trưởng, vừa có thể là thành viên trong đội khác.

---

## 4. FILE: src/capture.py — XỬ LÝ CAMERA

### 4.1 Mục đích

File này giống như một **người quay phim** — nó biết cách:
- Mở webcam 📷
- Đọc từng khung hình video 🎞️
- Vẽ chữ và hình lên màn hình ✏️
- Đóng camera khi xong 🔒

### 4.2 Cấu trúc

```
capture.py
│
├── class CameraCapture
│   ├── __init__()           → Khởi tạo
│   ├── open_camera()        → Mở webcam
│   ├── read_frame()         → Đọc 1 khung hình
│   ├── release_camera()     → Đóng camera
│   ├── draw_text()          → Vẽ chữ lên ảnh
│   ├── draw_emotion_box()   → Vẽ khung quanh mặt
│   ├── show_frame()         → Hiển thị ảnh
│   ├── wait_key()           → Chờ phím bấm
│   ├── destroy_all_windows()→ Đóng cửa sổ
│   ├── get_frame_size()     → Lấy kích thước ảnh
│   └── get_fps()            → Lấy tốc độ frame
│
└── if __name__ == "__main__" → Test code
```

### 4.3 Giải thích chi tiết

---

#### 📍 Import (dòng 10-12)

```python
import cv2                        # OpenCV - xử lý ảnh/video
import numpy as np                # NumPy - tính toán số học
from typing import Tuple, Optional # Gợi ý kiểu dữ liệu
```

**Giải thích:**
- `cv2` (OpenCV) — thư viện **xử lý ảnh và video** mạnh nhất trong Python. Nó có thể đọc webcam, phát hiện khuôn mặt, vẽ hình, đọc/ghi file ảnh,...
- `numpy as np` — thư viện **tính toán số học** trên mảng. Ảnh trong OpenCV thực chất là một mảng NumPy.
- `typing` — dùng để **gợi ý kiểu dữ liệu** cho hàm, giúp code dễ đọc hơn (nhưng Python không bắt buộc).

---

#### 📍 Class CameraCapture

```python
class CameraCapture:
    """
    Lớp quản lý camera và capture video
    """
```

---

##### Hàm __init__() (dòng 25-33)

```python
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False
```

**Giải thích:** `self.cap = None` — cap (viết tắt của **capture**) là đối tượng đại diện cho camera trong OpenCV. Ban đầu chưa kết nối nên là `None`. `self.is_opened = False` — chưa mở camera.

---

##### Hàm open_camera() (dòng 36-55)

```python
    def open_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ KHÔNG THỂ MỞ CAMERA ID: {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_opened = True
        return True
```

**Giải thích:**
- `cv2.VideoCapture(camera_id)` — tạo đối tượng kết nối đến camera. Nếu camera_id là đường dẫn file (vd: "video.mp4"), nó sẽ đọc file video.
- `.isOpened()` — kiểm tra xem camera có mở được không.
- `.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` — đặt độ phân giải ngang là 640 pixel.
- `.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)` — đặt độ phân giải dọc là 480 pixel.
- 640×480 là độ phân giải chuẩn VGA, vừa đủ rõ mà không quá nặng cho AI xử lý.

---

##### Hàm read_frame() (dòng 57-73)

```python
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        return True, frame
```

**Giải thích:** `.read()` trả về hai giá trị:
- `ret` (return) — `True` nếu đọc thành công, `False` nếu lỗi (hết video, mất kết nối,...)
- `frame` — **mảng NumPy 3 chiều** biểu diễn khung hình. Kích thước: (480, 640, 3) — 480 dòng, 640 cột, 3 kênh màu (B, G, R).

---

##### Hàm draw_text() (dòng 84-126)

```python
    def draw_text(self, frame: np.ndarray, text: str, ...) -> np.ndarray:
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
```

**Giải thích:** `cv2.getTextSize()` tính toán kích thước của dòng chữ (rộng bao nhiêu pixel, cao bao nhiêu) để vẽ khung nền đen vừa vặn.

```python
        cv2.rectangle(
            frame,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 5, position[1] + 5),
            (0, 0, 0), -1   # Màu đen, -1 = tô kín
        )
```

**Giải thích:** Vẽ một **hình chữ nhật màu đen** làm nền cho chữ dễ đọc. `-1` là **thickness** = -1 nghĩa là "tô đầy" (nét đậm = -1 → tô kín). Màu `(0, 0, 0)` là màu **đen** (B=0, G=0, R=0).

```python
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
```

**Giải thích:** `putText` (put text = đặt chữ) — vẽ chữ lên ảnh. `cv2.FONT_HERSHEY_SIMPLEX` là kiểu chữ đơn giản.

---

##### Hàm draw_emotion_box() (dòng 128-173)

```python
    def draw_emotion_box(self, frame, emotion, confidence, bbox):
        x, y, w, h = bbox
```

**Giải thích:** `bbox` (bounding box) là 4 số: `x` (vị trí trái), `y` (vị trí trên), `w` (chiều rộng), `h` (chiều cao). Câu lệnh này "giải nén" 4 số từ bộ dữ liệu. Kiểu: `bbox = (100, 50, 200, 250)` → `x=100, y=50, w=200, h=250`.

```python
        color_map = {
            'Happy': (0, 255, 0),      # Xanh lá
            'Sad': (255, 0, 0),        # Xanh dương
            'Surprise': (0, 165, 255), # Cam
            'Angry': (0, 0, 255),      # Đỏ
            'Neutral': (200, 200, 200),# Xám
            'Fear': (128, 0, 128),     # Tím
            'Disgust': (0, 128, 128)   # Nâu
        }
        color = color_map.get(emotion, (255, 255, 255))
```

**Giải thích:** Mỗi cảm xúc có một màu riêng:
- Vui → xanh lá (màu của hy vọng, tích cực)
- Buồn → xanh dương (màu của nước mắt, trầm lắng)
- Ngạc nhiên → cam (màu nổi bật, gây chú ý)
- Tức giận → đỏ (màu của lửa, giận dữ)
- Trung tính → xám (màu trung tính)
- Sợ hãi → tím (màu huyền bí, sợ hãi)
- Ghê tởm → nâu (màu không dễ chịu)

`.color_map.get(emotion, (255, 255, 255))` — nếu cảm xúc không có trong danh sách, dùng màu trắng làm mặc định.

```python
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
```

**Giải thích:** Vẽ **khung chữ nhật** quanh khuôn mặt. `(x, y)` là góc trên-bên-trái, `(x+w, y+h)` là góc dưới-bên-phải. `2` là độ dày nét vẽ.

```python
        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), ...)
```

**Giải thích:** Vẽ dòng chữ như "Happy: 0.95" phía trên khung mặt. `:.2f` — làm tròn confidence đến 2 chữ số thập phân.

---

##### Các hàm khác (dòng 175-230)

```python
    def show_frame(self, frame, window_name="Emotion Detection"):
        cv2.imshow(window_name, frame)
```
**Giải thích:** `imshow` (image show) — hiển thị ảnh lên cửa sổ có tên "Emotion Detection - HUST".

```python
    def wait_key(self, delay=1):
        return cv2.waitKey(delay)
```
**Giải thích:** `waitKey(delay)` — chờ delay mili-giây để nhận phím bấm. Trả về mã ASCII của phím, hoặc -1.

```python
    @staticmethod
    def destroy_all_windows():
        cv2.destroyAllWindows()
```
**Giải thích:** `@staticmethod` — một **trang trí** (decorator) nói rằng hàm này không cần `self`. Nó đóng tất cả cửa sổ OpenCV.

```python
    def get_frame_size(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
```
**Giải thích:** `cap.get(PROPERTY_ID)` — lấy một thuộc tính của camera. `cv2.CAP_PROP_FRAME_WIDTH` là ID của thuộc tính "độ rộng khung hình". Tương tự cho chiều cao.

---

#### 📍 Test code (dòng 233-287)

```python
if __name__ == "__main__":
```

**Giải thích:** Phần test này chỉ chạy khi bạn gõ `python src/capture.py` (chạy trực tiếp). Khi file này được import vào main.py, phần test sẽ KHÔNG chạy — nhờ vào cơ chế `__name__` đã giải thích ở trên.

Phần test:
1. Tạo camera và mở webcam
2. Hiển thị kích thước khung hình và FPS
3. Vòng lặp đọc frame, vẽ thông tin lên ảnh, hiển thị
4. Mỗi 30 frame, vẽ một box test giả lập (một khuôn mặt giả vui vẻ)
5. Phím 'q' để thoát, 's' để chụp ảnh

---

## 5. FILE: src/emotion_engine.py — "BỘ NÃO" NHẬN DIỆN

### 5.1 Mục đích

Đây là file **quan trọng nhất về mặt kỹ thuật** (bên cạnh main.py). Nó chứa AI thực sự!

File này giống như một **chuyên gia tâm lý học** — nhìn vào khuôn mặt và phán đoán cảm xúc. Nhưng nó còn thông minh hơn nhờ các kỹ thuật:
- **HSEmotion**: AI nhận diện cảm xúc từ ảnh mặt
- **Haar Cascade**: Phát hiện vị trí khuôn mặt
- **Temporal Smoothing**: Làm mịn kết quả theo thời gian
- **CLAHE**: Cân bằng ánh sáng
- **Adaptive Threshold**: Ngưỡng thông minh cho từng cảm xúc

### 5.2 Cấu trúc

```
emotion_engine.py
│
├── class EmotionEngine
│   ├── __init__()               → Khởi tạo AI
│   ├── _preprocess_face()       → Tiền xử lý ảnh mặt
│   ├── _get_face_key()          → Định danh khuôn mặt
│   ├── predict_emotion()        → Dự đoán cảm xúc
│   ├── detect_faces()           → Phát hiện khuôn mặt
│   ├── process_frame()          → Xử lý toàn bộ khung hình
│   └── get_emotion_distribution() → Phân bố xác suất
│
└── if __name__ == "__main__"    → Test code
```

### 5.3 Giải thích chi tiết

---

#### 📍 Import (dòng 10-14)

```python
import cv2                                           # Xử lý ảnh
import numpy as np                                   # Tính toán
from hsemotion.facial_emotions import HSEmotionRecognizer  # AI cảm xúc
from collections import deque                         # Hàng đợi (bộ đệm)
from typing import Tuple, List, Optional, Dict        # Gợi ý kiểu
```

**Giải thích:**
- `HSEmotionRecognizer` — Thư viện **HSEmotion** (EmotiEffLib) của Đại học Công nghệ Thông tin, Truyền thông và Kỹ thuật Điện Warszawa (Ba Lan). Nó dùng mạng neural **EfficientNet** để nhận diện 8 cảm xúc.
- `deque` — viết tắt của "**double-ended queue**" (hàng đợi hai đầu). Là một list thông minh: nếu bạn giới hạn nó là 10 phần tử, khi thêm phần tử thứ 11, phần tử cũ nhất sẽ tự động biến mất.

---

#### 📍 Hàm __init__() (dòng 27-60)

```python
    def __init__(self, model_name: str = 'enet_b0_8_best_afew'):
        self.model_name = model_name
        self.emotion_recognizer = HSEmotionRecognizer(model_name=model_name)
```

**Giải thích:** Tạo đối tượng nhận diện cảm xúc. `model_name` có thể là:
- `'enet_b0_8_best_afew'` — EfficientNet-B0, nhanh nhất (mặc định)
- `'enet_b0_8_best_vgaf'` — EfficientNet-B0, chính xác hơn
- `'enet_b2_8'` — EfficientNet-B2, chính xác nhất nhưng chậm

Các model này đều nhận diện **8 cảm xúc** (con số 8 trong tên model): Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.

```python
        self.face_cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_classifier = cv2.CascadeClassifier(self.face_cascade)
```

**Giải thích:** Tải **mô hình phát hiện khuôn mặt** Haar Cascade. Đây là một file XML chứa các đặc trưng của khuôn mặt người (mắt, mũi, miệng,...) được huấn luyện từ hàng ngàn bức ảnh. `cv2.data.haarcascades` là đường dẫn đến thư mục chứa các file XML này.

```python
        self.emotion_labels = [
            'Anger', 'Contempt', 'Disgust', 'Fear',
            'Happiness', 'Neutral', 'Sadness', 'Surprise'
        ]
```

**Giải thích:** Danh sách 8 cảm xúc theo đúng thứ tự mà mô hình HSEmotion xuất ra. Khi AI trả về một mảng 8 số xác suất (vd: [0.02, 0.01, 0.01, 0.03, 0.85, 0.05, 0.02, 0.01]), chúng ta biết số thứ 5 (index 4) là 'Happiness' với 85%.

```python
        # Temporal Smoothing: Mỗi khuôn mặt có buffer riêng
        self.face_buffers: Dict[tuple, deque] = {}
        self.BUFFER_SIZE = 10    # Lưu 10 frame gần nhất
        self.GRID_SIZE = 60      # Kích thước ô lưới 60px
```

**Giải thích:** **Ý tưởng THÔNG MINH** — Mỗi khuôn mặt có buffer riêng:
- `self.face_buffers` là một **dictionary** (từ điển). Key là vị trí khuôn mặt, value là buffer (deque) chứa 10 kết quả gần nhất.
- Tại sao phải riêng? Nếu có 2 người, một người vui và một người buồn, buffer chung có thể làm kết quả sai lẫn lộn.
- `GRID_SIZE = 60` — Chia màn hình thành các ô 60×60 pixel. Hai khuôn mặt có tâm trong cùng một ô được coi là CÙNG MỘT khuôn mặt (chịu được rung nhẹ).

```python
        # Adaptive Thresholds
        self.thresholds = {
            'Happiness': 0.50,   # Cao → tránh nhận nhầm khi nói cười
            'Sadness': 0.30,     # Thấp → nhạy với buồn nhẹ
            'Surprise': 0.35,    # Thấp → bắt nhanh ngạc nhiên
            'Anger': 0.40,
            'Neutral': 0.35
        }
```

**Giải thích:** Ngưỡng tin cậy (threshold) là "điểm đủ để kết luận". Khác với AI thông thường dùng 1 ngưỡng cố định (vd: 50%), ở đây mỗi cảm xúc có ngưỡng riêng:
- **Happiness (50%)**: Ngưỡng cao vì mặt người hay có nụ cười xã giao, cần chắc chắn mới kết luận "vui".
- **Sadness (30%)**: Ngưỡng thấp vì buồn thường rất khó thấy trên mặt, cần nhạy hơn.
- **Surprise (35%)**: Ngưỡng thấp vì ngạc nhiên thoáng qua rất nhanh.

Nếu độ tin cậy < ngưỡng → kết luận là **'Neutral'** (trung tính).

```python
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```

**Giải thích:** CLAHE = "**Contrast Limited Adaptive Histogram Equalization**" — nghe có vẻ phức tạp nhưng ý tưởng đơn giản: nó **cân bằng ánh sáng** cục bộ. 

Ví dụ: Nếu bạn ngồi dưới ánh đèn vàng, mặt bạn bị ngả vàng → AI khó nhận diện. CLAHE chia ảnh thành nhiều ô nhỏ (8×8 = 64 ô), cân bằng độ tương phản từng ô một, giúp làm rõ các **nếp nhăn**, **đặc trưng biểu cảm** trên mặt.

---

#### 📍 Hàm _preprocess_face() (dòng 62-77)

```python
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        if face_img.size == 0:
            return face_img
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)     # Bước 1: đổi sang đen trắng
        equalized = self.clahe.apply(gray)                      # Bước 2: cân bằng ánh sáng
        processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB) # Bước 3: đổi về màu
        return processed
```

**Giải thích:** Quá trình "tiền xử lý" giống như chụp ảnh selfie và dùng app chỉnh sửa:
1. **BGR → GRAY**: Ảnh màu (3 kênh) → đen trắng (1 kênh). Dễ xử lý hơn.
2. **CLAHE**: Cân bằng sáng/tối, làm rõ các đặc điểm trên mặt.
3. **GRAY → RGB**: Đưa về 3 kênh màu RGB (vì HSEmotion yêu cầu đầu vào 3 kênh).

Chú ý: `cv2.COLOR_BGR2GRAY` — OpenCV dùng BGR (xanh dương-lục-đỏ) nhưng HSEmotion dùng RGB (đỏ-lục-xanh dương). Đây là 2 thứ tự khác nhau!

---

#### 📍 Hàm _get_face_key() (dòng 79-86)

```python
    def _get_face_key(self, x: int, y: int, w: int, h: int) -> tuple:
        cx = ((x + w // 2) // self.GRID_SIZE) * self.GRID_SIZE
        cy = ((y + h // 2) // self.GRID_SIZE) * self.GRID_SIZE
        return (cx, cy)
```

**Giải thích:** Đây là "**số căn cước**" của mỗi khuôn mặt:
1. Tính tâm khuôn mặt: `cx = x + w//2`, `cy = y + h//2`
2. Lượng tử hóa: `(cx // 60) * 60` — làm tròn xuống bội số của 60 

Ví dụ: Nếu tâm khuôn mặt ở pixel (127, 93):
- cx = (127 // 60) * 60 = 2 * 60 = 120
- cy = (93 // 60) * 60 = 1 * 60 = 60
- Key = (120, 60)

Nếu frame sau, khuôn mặt rung nhẹ đến (133, 98), vẫn cho key là (120, 60) → vẫn được coi là cùng một khuôn mặt.

---

#### 📍 Hàm predict_emotion() — PHỨC TẠP NHẤT (dòng 88-135)

```python
    def predict_emotion(self, face_img: np.ndarray,
                        face_key: tuple = (0, 0)) -> Tuple[str, float, np.ndarray]:
```

**Giải thích:** Hàm này nhận vào ảnh khuôn mặt, trả về 3 thứ:
1. `str` — tên cảm xúc (vd: "Happy")
2. `float` — độ tin cậy (vd: 0.85)
3. `np.ndarray` — mảng xác suất của 8 cảm xúc

```python
        try:
            processed_face = self._preprocess_face(face_img)
            emotion_name_raw, scores = self.emotion_recognizer.predict_emotions(
                processed_face, logits=False
            )
```

**Giải thích:** `predict_emotions()` của HSEmotion trả về:
- `emotion_name_raw` — tên cảm xúc (nhưng không dùng, vì ta muốn dùng buffer smoothing)
- `scores` — mảng 8 số, mỗi số là xác suất (từ 0 đến 1) của 8 cảm xúc. `logits=False` nghĩa là "trả về xác suất đã chuẩn hóa (tổng = 1), đừng trả về logits thô".

```python
            if face_key not in self.face_buffers:
                self.face_buffers[face_key] = deque(maxlen=self.BUFFER_SIZE)
            buf = self.face_buffers[face_key]
            buf.append(scores)
```

**Giải thích:** Lấy (hoặc tạo mới) buffer riêng cho khuôn mặt này. `deque(maxlen=10)` là một hàng đợi chứa tối đa 10 phần tử — nếu thêm phần tử thứ 11, phần tử cũ nhất tự động bị đẩy ra.

```python
            avg_scores = np.mean(buf, axis=0)
```

**Giải thích:** **Temporal Smoothing** — tính TRUNG BÌNH của tất cả scores trong buffer. `np.mean()` tính giá trị trung bình. `axis=0` nghĩa là "tính trung bình theo cột" — tức là tính xác suất trung bình của từng cảm xúc trong 10 frame gần nhất.

Ví dụ:
```
Frame 1: [0.1, 0.7, 0.1, 0.1]  (Sad 70%)
Frame 2: [0.2, 0.5, 0.2, 0.1]  (Sad 50%)
Frame 3: [0.6, 0.2, 0.1, 0.1]  (Happy 60%) ← AI lắc lư!
─────────────────────────────────
Average: [0.3, 0.47, 0.13, 0.1] → Sad (47%)  ← Ổn định hơn!
```

Không có smoothing: AI sẽ nhảy từ Sad → Happy → Sad liên tục (giật). Có smoothing: kết quả mượt mà, chuyển đổi từ từ.

```python
            max_idx = np.argmax(avg_scores)
            max_conf = float(avg_scores[max_idx])
            emotion_candidate = self.emotion_labels[max_idx]
```

**Giải thích:** `argmax()` trả về **chỉ số** (index) của phần tử lớn nhất. Ví dụ: avg_scores = [0.05, 0.02, 0.01, 0.02, 0.80, 0.05, 0.03, 0.02] → argmax = 4 → emotion_candidate = 'Happiness' (vì self.emotion_labels[4] = 'Happiness').

```python
            threshold = self.thresholds.get(emotion_candidate, 0.40)
            
            if max_conf < threshold:
                final_emotion = 'Neutral'
            else:
                mapping = {
                    'Anger': 'Angry',
                    'Happiness': 'Happy',
                    'Sadness': 'Sad',
                    'Contempt': 'Disgust'
                }
                final_emotion = mapping.get(emotion_candidate, emotion_candidate)
```

**Giải thích:** Đây là bộ lọc cuối cùng:
1. Lấy ngưỡng riêng cho cảm xúc dự đoán (mặc định 0.4 nếu không có)
2. Nếu xác suất < ngưỡng → trả về 'Neutral' (không đủ tin tưởng để kết luận)
3. Nếu đủ ngưỡng → ánh xạ tên: HSEmotion gọi 'Happiness' nhưng ta muốn 'Happy'; 'Anger' → 'Angry'; 'Sadness' → 'Sad'; 'Contempt' → 'Disgust' (khinh bỉ và ghê tởm là tương tự)

```python
        except Exception as e:
            print(f"❌ Lỗi khi predict emotion: {e}")
            return "Neutral", 0.0, np.zeros(len(self.emotion_labels))
```

**Giải thích:** Nếu có lỗi (vd: ảnh quá nhỏ, không đọc được), trả về an toàn: Neutral với confidence = 0.

---

#### 📍 Hàm detect_faces() (dòng 137-143)

```python
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
```

**Giải thích:** `detectMultiScale()` — phát hiện vật thể ở nhiều kích cỡ khác nhau:
- `gray` — ảnh đen trắng (xử lý nhanh hơn ảnh màu)
- `scaleFactor=1.1` — thu nhỏ ảnh 10% mỗi lần (1.1 = 110%). Số càng gần 1, phát hiện càng kỹ nhưng chậm hơn.
- `minNeighbors=5` — một vùng phải có ít nhất 5 "hàng xóm" là khuôn mặt mới được xác nhận. Cao hơn → giảm phát hiện sai (false positive).
- `minSize=(30, 30)` — bỏ qua các vùng nhỏ hơn 30×30 pixel (nhiễu).

---

#### 📍 Hàm process_frame() (dòng 145-178)

```python
    def process_frame(self, frame):
        results = []
        faces = self.detect_faces(frame)
        
        # Dọn buffer cũ
        active_keys = set()
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.shape[0] >= 48 and face_img.shape[1] >= 48:
                active_keys.add(self._get_face_key(x, y, w, h))
        stale_keys = set(self.face_buffers.keys()) - active_keys
        for k in stale_keys:
            del self.face_buffers[k]
```

**Giải thích:** Quản lý bộ nhớ buffer — xóa buffer của các khuôn mặt cũ không còn xuất hiện. `set` là cấu trúc dữ liệu "tập hợp", hỗ trợ phép tính hiệu (`-`). Chỉ giữ buffer cho khuôn mặt đã xuất hiện ở frame này và có kích thước ≥ 48×48 pixel (đủ lớn để nhận diện).

```python
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                continue
            face_key = self._get_face_key(x, y, w, h)
            emotion, confidence, probs = self.predict_emotion(face_img, face_key)
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probs
            })
```

**Giải thích:** Với mỗi khuôn mặt đủ lớn, cắt ảnh mặt ra, gán key, dự đoán cảm xúc, và thêm kết quả vào danh sách.

---

#### 📍 Hàm get_emotion_distribution() (dòng 180-193)

```python
    def get_emotion_distribution(self, probabilities):
        return {
            emotion: float(prob)
            for emotion, prob in zip(self.emotion_labels, probabilities)
        }
```

**Giải thích:** Chuyển mảng 8 xác suất thành dictionary dễ đọc. `zip()` ghép hai list lại với nhau: `[Anger, Contempt,...]` với `[0.02, 0.01,...]` → `{Anger: 0.02, Contempt: 0.01, ...}`.

---

## 6. FILE: src/database.py — KHO LƯU TRỮ

### 6.1 Mục đích

File này quản lý **cơ sở dữ liệu SQLite** — một cơ sở dữ liệu siêu nhẹ, lưu trữ trong một file duy nhất (`emotions.db`). Giống như một **kế toán viên** ghi chép mọi giao dịch.

### 6.2 Cấu trúc

```
database.py
│
├── class EmotionDatabase
│   ├── __init__()                → Kết nối database
│   ├── _create_tables()          → Tạo bảng
│   ├── insert_emotion()          → Thêm bản ghi
│   ├── get_emotions_by_session() → Lấy dữ liệu theo session
│   ├── get_emotion_counts()      → Đếm số lượng
│   ├── get_average_confidence()  → Độ tin cậy trung bình
│   ├── get_all_sessions()        → Danh sách session
│   ├── delete_session()          → Xóa session
│   └── clear_all_data()          → Xóa toàn bộ
│
└── if __name__ == "__main__"    → Test code
```

### 6.3 Giải thích chi tiết

---

#### 📍 Import (dòng 10-13)

```python
import sqlite3     # Thư viện SQLite (có sẵn trong Python)
import os          # Thao tác file
from datetime import datetime  # Thời gian
from typing import List, Dict, Tuple  # Gợi ý kiểu
```

---

#### 📍 Hàm __init__() (dòng 26-39)

```python
    def __init__(self, db_path: str = "data/emotions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._create_tables()
```

**Giải thích:** Tạo thư mục `data/` nếu chưa tồn tại, rồi gọi `_create_tables()` để tạo bảng.

---

#### 📍 Hàm _create_tables() (dòng 41-77)

```python
    def _create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
```

**Giải thích:** 
- `sqlite3.connect(db_path)` — mở kết nối đến file database. Nếu file chưa có, nó sẽ tạo file mới.
- `cursor` — con trỏ, giống như "ngón tay trỏ" để thực hiện các câu lệnh SQL.

```python
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                session_id TEXT NOT NULL
            )
        """)
```

**Giải thích:** Đây là **câu lệnh SQL** tạo bảng `emotions` với 5 cột:
- **`id`** — số thứ tự tự động tăng (`AUTOINCREMENT`). Mỗi bản ghi mới tự động có id = id cũ + 1.
- **`timestamp`** — thời gian (dạng text, ISO 8601). `NOT NULL` = bắt buộc phải có.
- **`emotion`** — tên cảm xúc (text).
- **`confidence`** — độ tin cậy (số thực, `REAL`). VD: 0.95.
- **`session_id`** — ID phiên học (text). Giúp phân biệt dữ liệu giữa các lần chạy.

```python
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id
            ON emotions(session_id)
        """)
```

**Giải thích:** **INDEX** (chỉ mục) — giống như mục lục của quyển sách. Khi cần tìm dữ liệu theo `session_id`, thay vì phải đọc hết 100.000 bản ghi, SQLite chỉ cần tra mục lục. Nhanh hơn hàng trăm lần!

---

#### 📍 Hàm insert_emotion() (dòng 79-99)

```python
    def insert_emotion(self, emotion: str, confidence: float, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO emotions (timestamp, emotion, confidence, session_id)
            VALUES (?, ?, ?, ?)
        """, (timestamp, emotion, confidence, session_id))
        
        conn.commit()
        conn.close()
```

**Giải thích:** Câu lệnh `INSERT INTO` thêm một dòng mới vào bảng. Dấu `?` là **placeholder** (chỗ trống) — sau đó được thay bằng giá trị thực tế từ bộ `(timestamp, emotion, confidence, session_id)`. Tại sao không viết thẳng? Để **tránh tấn công SQL injection** (một kiểu hack).

- `conn.commit()` — **xác nhận** thay đổi. Giống như bạn nói "OK, ghi vào sổ đi".
- `conn.close()` — **đóng kết nối**. Luôn đóng sau khi dùng xong để tránh rò rỉ tài nguyên.

---

#### 📍 Các hàm truy vấn (dòng 101-226)

```python
    def get_emotion_counts(self, session_id):
        cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM emotions WHERE session_id = ?
            GROUP BY emotion ORDER BY count DESC
        """, (session_id,))
        results = cursor.fetchall()
        return {emotion: count for emotion, count in results}
```

**Giải thích:** `COUNT(*)` là hàm đếm của SQL. `GROUP BY emotion` nghĩa là "gom nhóm theo cảm xúc". `ORDER BY count DESC` — sắp xếp từ cao xuống thấp. Kết quả: `[('Happy', 45), ('Neutral', 30), ('Surprise', 15)]`.

```python
    def get_average_confidence(self, session_id):
        cursor.execute("""
            SELECT emotion, AVG(confidence) as avg_conf
            FROM emotions WHERE session_id = ?
            GROUP BY emotion
        """, (session_id,))
        results = cursor.fetchall()
        return {emotion: round(avg_conf, 3) for emotion, avg_conf in results}
```

**Giải thích:** `AVG(confidence)` là hàm tính TRUNG BÌNH của SQL. `round(avg_conf, 3)` làm tròn đến 3 chữ số thập phân.

```python
    def get_all_sessions(self):
        cursor.execute("""
            SELECT DISTINCT session_id
            FROM emotions ORDER BY session_id DESC
        """)
```

**Giải thích:** `SELECT DISTINCT` — lấy các giá trị KHÔNG TRÙNG nhau.

```python
    def clear_all_data(self):
        cursor.execute("DELETE FROM emotions")
```

**Giải thích:** `DELETE FROM emotions` — xóa TOÀN BỘ dữ liệu trong bảng. Cẩn thận! Không thể undo.

---

## 7. FILE: src/analytics.py — VẼ BIỂU ĐỒ

### 7.1 Mục đích

File này giống như một **họa sĩ vẽ tranh** — nó lấy dữ liệu cảm xúc và tạo ra các biểu đồ đẹp mắt:
- Biểu đồ cột 📊 (số lần mỗi cảm xúc)
- Biểu đồ tròn 🥧 (tỉ lệ phần trăm)
- Biểu đồ timeline 📈 (biến đổi theo thời gian)
- Báo cáo tổng hợp 🗂️ (kết hợp tất cả)

### 7.2 Cấu trúc

```
analytics.py
│
├── class EmotionAnalytics
│   ├── __init__()                    → Khởi tạo
│   ├── plot_emotion_counts()         → Biểu đồ cột
│   ├── plot_emotion_pie()            → Biểu đồ tròn
│   ├── plot_emotion_timeline()       → Biểu đồ thời gian
│   └── generate_summary_report()     → Báo cáo tổng hợp
│
└── if __name__ == "__main__"        → Test code
```

### 7.3 Giải thích chi tiết

---

#### 📍 Import (dòng 10-16)

```python
import matplotlib.pyplot as plt     # Vẽ biểu đồ
import seaborn as sns               # Làm đẹp biểu đồ
import pandas as pd                 # Xử lý dữ liệu bảng
import numpy as np                  # Tính toán số học
from typing import Dict, List       # Gợi ý kiểu
import os                           # Thao tác file
from datetime import datetime       # Thời gian
```

---

#### 📍 Hàm __init__() (dòng 29-53)

```python
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)
```

**Giải thích:** 
- `output_dir = "reports"` — tất cả biểu đồ sẽ được lưu trong thư mục `reports/`.
- `sns.set_theme(style="whitegrid")` — đặt style cho biểu đồ: nền trắng với lưới kẻ (dễ đọc).
- `plt.rcParams['font.size'] = 10` — cỡ chữ mặc định là 10.
- `plt.rcParams['figure.figsize'] = (12, 8)` — kích thước mặc định của biểu đồ: 12 inch × 8 inch.

```python
        self.emotion_colors = {
            'Happy': '#2ecc71',      # Xanh lá
            'Sad': '#3498db',        # Xanh dương
            'Surprise': '#f39c12',   # Cam
            'Angry': '#e74c3c',      # Đỏ
            'Neutral': '#95a5a6',    # Xám
            'Fear': '#9b59b6',       # Tím
            'Disgust': '#16a085'     # Xanh lục
        }
```

**Giải thích:** Mỗi cảm xúc có một **màu sắc** cố định, dùng mã màu HEX (thường dùng trong web design).

---

#### 📍 Hàm plot_emotion_counts() (dòng 55-113)

```python
    def plot_emotion_counts(self, emotion_counts, session_id, save_path=None):
        plt.figure(figsize=(10, 6))
        
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = [self.emotion_colors.get(e, '#7f8c8d') for e in emotions]
        
        bars = plt.bar(emotions, counts, color=colors, alpha=0.8, edgecolor='black')
```

**Giải thích:**
- `plt.bar(emotions, counts, ...)` — vẽ **biểu đồ cột** (bar chart). Trục X là tên cảm xúc, trục Y là số lần.
- `alpha=0.8` — độ trong suốt 80% (1.0 = đục hoàn toàn).
- `edgecolor='black'` — viền cột màu đen.

```python
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
```

**Giải thích:** Vòng lặp thêm **nhãn số** lên đỉnh mỗi cột:
- `bar.get_x()` + `bar.get_width()/2` — vị trí giữa cột
- `bar.get_height()` — chiều cao cột (giá trị)
- `ha='center'` — căn giữa theo chiều ngang
- `va='bottom'` — căn dưới theo chiều dọc

---

#### 📍 Hàm plot_emotion_pie() (dòng 115-179)

```python
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=emotions,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
```

**Giải thích:** `plt.pie()` vẽ **biểu đồ tròn**:
- `autopct='%1.1f%%'` — hiển thị phần trăm với 1 số thập phân. VD: "45.5%"
- `startangle=90` — bắt đầu vẽ từ góc 90 độ (trên cùng). Mặc định là 0 (bên phải).

```python
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
```

**Giải thích:** Làm đẹp text phần trăm: màu trắng, chữ đậm.

---

#### 📍 Hàm plot_emotion_timeline() (dòng 181-243)

```python
    def plot_emotion_timeline(self, emotion_data, session_id, save_path=None):
        df = pd.DataFrame(emotion_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
```

**Giải thích:** `pd.DataFrame()` chuyển list dictionary thành **bảng dữ liệu** (giống Excel). `pd.to_datetime()` chuyển chuỗi thời gian thành kiểu ngày giờ của pandas.

```python
        for emotion in df['emotion'].unique():
            emotion_df = df[df['emotion'] == emotion].copy()
            emotion_df = emotion_df.sort_values('timestamp')
            emotion_df['count'] = 1
            emotion_df['rolling_count'] = emotion_df['count'].rolling(
                window=window_size, min_periods=1
            ).sum()
```

**Giải thích:** Với mỗi cảm xúc, vẽ một đường riêng:
- `df['emotion'].unique()` — lấy danh sách các cảm xúc duy nhất
- `rolling(window=10).sum()` — **cửa sổ trượt** (rolling window). Mỗi điểm trên đồ thị là tổng của 10 mẫu gần nhất. Giúp đường biểu đồ mượt hơn, không bị gai nhọn.

Ví dụ rolling: Dữ liệu gốc: [1, 1, 0, 1, 1, 0], window=3 → [1, 2, 2, 2, 2, 1] (tổng 3 số gần nhau)

---

#### 📍 Hàm generate_summary_report() (dòng 245-341)

```python
    def generate_summary_report(self, emotion_counts, avg_confidence,
                                session_id, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
```

**Giải thích:** `plt.subplots(2, 2)` tạo một **lưới 2×2 biểu đồ** (4 biểu đồ con trên cùng một ảnh):
- `axes[0, 0]` — biểu đồ cột (góc trên trái)
- `axes[0, 1]` — biểu đồ tròn (góc trên phải)
- `axes[1, 0]` — biểu đồ độ tin cậy (góc dưới trái)
- `axes[1, 1]` — bảng thống kê (góc dưới phải)

```python
        axes[1, 0].barh(conf_emotions, conf_values, ...)
```

**Giải thích:** `barh()` — bar **h**orizontal = biểu đồ cột NẰM NGANG.

```python
        axes[1, 1].axis('off')
        table_data = []
        for emotion in emotions:
            table_data.append([emotion, count, f"{percentage:.1f}%", f"{conf:.2f}"])
        table = axes[1, 1].table(cellText=table_data, ...)
```

**Giải thích:** Tạo một **bảng dữ liệu** với 4 cột: Cảm xúc | Số lần | Tỉ lệ | Độ tin cậy. `axis('off')` — tắt trục tọa độ (chỉ hiện bảng).

```python
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
```

**Giải thích:** Lưu biểu đồ với **DPI** (dots per inch) = 300 — chất lượng in ấn. `bbox_inches='tight'` — cắt bỏ viền trắng thừa. `plt.close()` — đóng figure để giải phóng bộ nhớ.

---

## 8. FILE: requirements.txt

### 8.1 Mục đích

File này là **danh sách mua sắm** — nó liệt kê tất cả các thư viện Python cần cài đặt để dự án chạy được.

### 8.2 Giải thích từng dòng

```
opencv-python==4.9.0.80          # Xử lý camera, video, hình ảnh
opencv-contrib-python==4.9.0.80  # Các module bổ sung của OpenCV
hsemotion==0.2.0                 # EmotiEffLib - Nhận dạng cảm xúc (EfficientNet)
torch==2.2.0                     # PyTorch - Backend cho hsemotion
torchvision==0.17.0              # Xử lý ảnh cho PyTorch
timm==0.9.16                     # PyTorch Image Models (cần cho EfficientNet)
matplotlib==3.8.3                # Vẽ biểu đồ cơ bản
seaborn==0.13.2                  # Vẽ biểu đồ đẹp hơn
pandas==2.2.1                    # Xử lý bảng dữ liệu
numpy==1.26.4                    # Tính toán số học
pillow==10.2.0                   # Xử lý ảnh bổ sung
```

**Có gì khác thường?** Các dòng note `#` chỉ là chú thích, không có tác dụng khi cài đặt. `pip` chỉ đọc phần trước dấu `=` hoặc `==`.

**`==` có nghĩa là gì?** Trong requirements.txt:
- `opencv-python==4.9.0.80` — cài ĐÚNG phiên bản 4.9.0.80 (không khác)
- `opencv-python>=4.9.0.80` — cài phiên bản >= 4.9.0.80 (có thể cao hơn)
- `opencv-python` — cài phiên bản mới nhất

Dự án này dùng `==` để đảm bảo **tương thích** — các phiên bản khác có thể không chạy đúng.

---

## 9. FILE: README.md

### 9.1 Mục đích

`README.md` là **tờ giới thiệu** của dự án. Khi ai đó vào thư mục dự án, đây là file đầu tiên họ đọc. Nó viết bằng **Markdown** (`.md`) — một ngôn ngữ đánh dấu đơn giản để tạo văn bản đẹp.

### 9.2 Nội dung chính

File này bao gồm:
1. **Tên dự án** và **mô tả ngắn**
2. **Tính năng chính**: nhận dạng real-time, tối ưu hiệu năng, lưu trữ SQLite, báo cáo trực quan
3. **Yêu cầu hệ thống**: Python 3.8+, webcam
4. **Hướng dẫn cài đặt** chi tiết từng bước
5. **Hướng dẫn sử dụng**: các option dòng lệnh
6. **Cấu trúc thư mục**
7. **Đầu ra hệ thống**: 4 loại biểu đồ

---

## 10. LUỒNG HOẠT ĐỘNG TỔNG THỂ

### 10.1 Sơ đồ luồng

```
NGƯỜI DÙNG gõ: python main.py
        │
        ▼
    ┌──────────────────┐
    │    main()        │  ← Đọc tham số (--model, --camera, --db)
    │  argparse        │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  EmotionDetection│  ← TẠO hệ thống
    │  System.__init__ │     - Mở kết nối camera
    │                  │     - Nạp AI model
    │                  │     - Kết nối database
    │                  │     - Chuẩn bị vẽ biểu đồ
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   system.start() │  ← BẮT ĐẦU
    │   → mở webcam    │
    └────────┬─────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────┐
    │  VÒNG LẶP CHÍNH (_main_loop)                        │
    │                                                     │
    │  while is_running:                                  │
    │  1. Đọc frame từ camera (capture.py)                │
    │  2. Nếu frame bị SKIP:                             │
    │     → Vẽ lại kết quả cũ, hiển thị                 │
    │  3. Nếu frame được XỬ LÝ:                         │
    │     a. Phát hiện khuôn mặt (Haar Cascade)          │
    │     b. Với mỗi mặt: cắt ảnh, tiền xử lý (CLAHE)   │
    │     c. HSEmotion → 8 xác suất cảm xúc              │
    │     d. Temporal Smoothing (trung bình 10 frame)    │
    │     e. Adaptive Threshold → cảm xúc cuối cùng      │
    │     f. Lưu vào buffer (RAM)                        │
    │     g. Mỗi 5 mẫu → ghi 1 bản ghi vào SQLite        │
    │  4. Vẽ overlay (khung mặt, thông tin)              │
    │  5. Hiển thị lên màn hình                          │
    │  6. Kiểm tra phím bấm (q=thoát, s=chụp, r=reset)  │
    └──────────────────────┬──────────────────────────────┘
                           │  (người dùng nhấn q/ESC)
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │  KẾT THÚC (_cleanup)                                │
    │  1. Đóng camera                                     │
    │  2. Lưu nốt dữ liệu còn trong buffer                │
    │  3. Lấy thống kê từ database                        │
    │  4. Tạo 4 biểu đồ (cột, tròn, timeline, tổng hợp)   │
    │  5. In kết quả ra màn hình                          │
    └─────────────────────────────────────────────────────┘
```

### 10.2 Luồng dữ liệu

```
Webcam ──→ capture.py ──→ emotion_engine.py ──→ buffer (RAM)
                                      │                  │
                                      ▼                  ▼
                               Database (SQLite) ←─── mỗi 5 mẫu
                                      │
                                      ▼
                               analytics.py ──→ Biểu đồ PNG
```

### 10.3 Kiến trúc module

```
main.py (Điều khiển trung tâm)
   │
   ├── src/capture.py (Giao tiếp với webcam)
   │       └── Vẽ khung mặt, text lên ảnh
   │
   ├── src/emotion_engine.py (AI nhận diện)
   │       ├── detect_faces() → Haar Cascade
   │       ├── _preprocess_face() → CLAHE
   │       ├── predict_emotion() → HSEmotion + Temporal Smoothing
   │       └── process_frame() → Kết hợp tất cả
   │
   ├── src/database.py (Lưu trữ)
   │       ├── SQLite → bảng emotions
   │       └── Các hàm CRUD (Create, Read, ...)
   │
   └── src/analytics.py (Báo cáo)
           ├── Matplotlib → bar, pie, line charts
           ├── Seaborn → làm đẹp
           └── Pandas → xử lý dữ liệu
```

### 10.4 Các kỹ thuật nổi bật

| Kỹ thuật | File | Tác dụng |
|-----------|------|----------|
| **Temporal Smoothing** | emotion_engine.py | Làm mịn kết quả AI, tránh giật |
| **Per-Face Buffer** | emotion_engine.py | Mỗi khuôn mặt 1 buffer riêng |
| **CLAHE** | emotion_engine.py | Cân bằng ánh sáng, rõ đặc trưng |
| **Adaptive Threshold** | emotion_engine.py | Ngưỡng riêng cho từng cảm xúc |
| **Frame Skipping** | main.py | AI xử lý cách frame, tăng FPS |
| **Batch DB Write** | main.py | Ghi database theo batch, giảm chậm |
| **Kết quả Cache** | main.py | Tái sử dụng trên frame skip |
| **Grid Face Key** | emotion_engine.py | Chịu rung nhẹ, ổn định nhận diện |

---

## 🎯 TỔNG KẾT

Dự án **Nhận dạng cảm xúc người học online** là một hệ thống hoàn chỉnh bao gồm:

1. **Capture** (thu thập) — webcam ghi lại khuôn mặt
2. **AI Engine** (nhận diện) — HSEmotion + các kỹ thuật xử lý ảnh nâng cao
3. **Database** (lưu trữ) — SQLite ghi lại lịch sử cảm xúc
4. **Analytics** (báo cáo) — Matplotlib vẽ biểu đồ trực quan
5. **Main** (điều phối) — Kết nối tất cả lại với nhau

Từ góc nhìn của người dùng cuối: **bật webcam → chạy chương trình → xem cảm xúc real-time trên màn hình → kết thúc → nhận báo cáo biểu đồ đẹp mắt** 📊

---

> 📝 **Ghi chú:** File backup gốc được lưu tại thư mục `backups/` với đuôi `.backup`.  
> Nếu cần khôi phục: copy file `.backup` thành `.py` tương ứng.  
> 
> *Tài liệu được tạo vào ngày 16/06/2026 cho dự án cuối kỳ - Đại học Bách khoa Hà Nội*

# 🎭 Hệ thống Nhận dạng Cảm xúc Người học Online

> **Môn học:** Thị giác Máy tính — Đại học Bách khoa Hà Nội  
> **Mô hình:** EfficientNet-B0 (HSEmotion) + Fine-tuning + Multi-face Detection  
> **Ngôn ngữ:** Python 3.8+ | PyTorch | OpenCV

---

## 📋 Mục lục

1. [Giới thiệu tổng quan](#1-giới-thiệu-tổng-quan)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Sơ đồ hoạt động hệ thống](#3-sơ-đồ-hoạt-động-hệ-thống)
4. [Các chức năng chính](#4-các-chức-năng-chính)
5. [Giải thích thuật toán & code](#5-giải-thích-thuật-toán--code)
6. [Quy trình Fine-tuning](#6-quy-trình-fine-tuning)
7. [Đánh giá mô hình](#7-đánh-giá-mô-hình)
8. [Cài đặt và chạy](#8-cài-đặt-và-chạy)
9. [Tham số dòng lệnh](#9-tham-số-dòng-lệnh)

---

## 1. Giới thiệu tổng quan

### 1.1. Mục tiêu

Hệ thống nhận dạng **7 cảm xúc cơ bản** của người học theo thời gian thực từ webcam, video hoặc ảnh tĩnh, bao gồm:

| Cảm xúc | Tiếng Anh | Ký hiệu trong model |
|---------|-----------|---------------------|
| Tức giận | Angry | `Anger` |
| Ghê tởm | Disgust | `Disgust` |
| Sợ hãi | Fear | `Fear` |
| Vui vẻ | Happy | `Happiness` |
| Trung tính | Neutral | `Neutral` |
| Buồn bã | Sad | `Sadness` |
| Ngạc nhiên | Surprise | `Surprise` |

### 1.2. Kiến trúc tổng thể

Hệ thống kết hợp **hai giai đoạn** xử lý:

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE NHẬN DIỆN                       │
│                                                             │
│   Đầu vào         Giai đoạn 1          Giai đoạn 2          │
│  (Frame ảnh)    (Detect khuôn mặt)   (Nhận diện cảm xúc)   │
│                                                             │
│  [Camera]  ──►  [Haar Cascade]  ──►  [EfficientNet-B0]      │
│  [Video ]       (Classical CV)        (Deep Learning)       │
│  [Image ]                                                   │
└─────────────────────────────────────────────────────────────┘
```

**Tại sao dùng 2 giai đoạn?**  
- Haar Cascade rất nhanh (~1ms/frame) để định vị khuôn mặt trong ảnh lớn
- EfficientNet tốn tài nguyên hơn nhưng chỉ cần xử lý vùng khuôn mặt đã cắt (224×224px), không phải toàn bộ frame

### 1.3. Model cốt lõi — EfficientNet-B0

**EfficientNet** (Tan & Le, 2019) được thiết kế dựa trên kỹ thuật **Compound Scaling** — đồng thời tăng độ rộng (width), chiều sâu (depth) và độ phân giải (resolution) theo một hệ số chung φ:

```
depth    d = α^φ
width    w = β^φ  
resolution r = γ^φ

với α·β²·γ² ≈ 2 (ràng buộc tài nguyên)
```

EfficientNet-B0 chỉ có **5.3M tham số** nhưng đạt độ chính xác tương đương ResNet-50 (25M tham số) — rất phù hợp chạy thời gian thực trên CPU.

Thư viện **HSEmotion (EmotiEffLib)** cung cấp EfficientNet đã được pre-train trên tập **AffectNet** (~450,000 ảnh khuôn mặt), với 3 checkpoint:

| Model | Tập train | Tốc độ | Độ chính xác |
|-------|-----------|--------|-------------|
| `enet_b0_8_best_afew` | AffectNet + AFEW | ⚡ Nhanh nhất | Tốt |
| `enet_b0_8_best_vgaf` | AffectNet + VGAF | ⚡ Nhanh | Tốt hơn |
| `enet_b2_8` | AffectNet | 🐢 Chậm | Tốt nhất |

---

## 2. Cấu trúc thư mục

```
KT/
│
├── main.py                    # Entry point — điều phối 5 chế độ chạy
├── collect_data.py            # Script thu thập ảnh dataset từ webcam
├── requirements.txt           # Danh sách thư viện cần cài
│
├── src/                       # Các module xử lý nghiệp vụ
│   ├── capture.py             # Quản lý camera và hiển thị
│   ├── emotion_engine.py      # CORE: nhận diện cảm xúc (CLAHE + Smoothing)
│   ├── fine_tuner.py          # Fine-tuning EfficientNet trên dataset nhỏ
│   ├── image_predictor.py     # Nhận diện từ ảnh tĩnh + vẽ probability bar
│   ├── evaluator.py           # So sánh model gốc vs fine-tuned, vẽ biểu đồ
│   ├── analytics.py           # (Legacy) Thống kê và biểu đồ phiên học
│   └── database.py            # (Legacy) Lưu trữ SQLite
│
├── dataset/                   # Dữ liệu tự thu thập để fine-tune
│   ├── README.md              # Hướng dẫn thu thập dataset
│   ├── Anger/                 # ≥30 ảnh khuôn mặt tức giận
│   ├── Disgust/               # ≥30 ảnh
│   ├── Fear/                  # ≥30 ảnh
│   ├── Happiness/             # ≥30 ảnh
│   ├── Neutral/               # ≥30 ảnh
│   ├── Sadness/               # ≥30 ảnh
│   └── Surprise/              # ≥30 ảnh
│
├── models/                    # Trọng số mô hình
│   └── finetuned.pt           # (Tự tạo) Weights sau fine-tuning
│
├── data/                      # Dữ liệu runtime
│   └── screenshot_*.jpg       # Ảnh chụp màn hình trong phiên chạy
│
└── reports/                   # Kết quả đánh giá
    ├── confusion_matrix.png   # Ma trận nhầm lẫn
    └── training_curve.png     # Biểu đồ loss/accuracy theo epoch
```

---

## 3. Sơ đồ hoạt động hệ thống

### 3.1. Luồng xử lý chính (Webcam / Video)

```
╔══════════════════════════════════════════════════════════════╗
║                   VÒNG LẶP XỬ LÝ FRAME                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [Frame mới]                                                 ║
║       │                                                      ║
║       ▼                                                      ║
║  Frame số N?  ──── N % 2 ≠ 0 ────► [Dùng kết quả cũ]        ║
║       │                                  │                   ║
║       │ N % 2 == 0                       │                   ║
║       ▼                                  │                   ║
║  [Haar Cascade]                          │                   ║
║  Detect khuôn mặt                        │                   ║
║       │                                  │                   ║
║       ├── Không có mặt ──────────────────┤                   ║
║       │                                  │                   ║
║       ▼                                  │                   ║
║  Với mỗi khuôn mặt (x,y,w,h):           │                   ║
║  ┌─────────────────────────┐             │                   ║
║  │ 1. Cắt vùng mặt        │             │                   ║
║  │ 2. CLAHE preprocessing  │             │                   ║
║  │ 3. EfficientNet predict │             │                   ║
║  │ 4. Temporal Smoothing   │             │                   ║
║  │ 5. Adaptive Threshold   │             │                   ║
║  └─────────────────────────┘             │                   ║
║       │                                  │                   ║
║       ▼                                  ▼                   ║
║  [Vẽ Bounding Box + Label] ◄─────────────┘                   ║
║       │                                                      ║
║       ▼                                                      ║
║  [Hiển thị / Lưu Video]                                      ║
║       │                                                      ║
║       ▼                                                      ║
║  [Xử lý phím bấm] ──── q/ESC ──► [Kết thúc]                 ║
║       │                                                      ║
║       └──────────────── [Frame tiếp theo]                    ║
╚══════════════════════════════════════════════════════════════╝
```

### 3.2. Quy trình Fine-tuning & Đánh giá

```
  Thu thập ảnh          Fine-tuning            Đánh giá
  ──────────────        ─────────────          ──────────
  
  [collect_data.py]     [Dataset]              [Test Set 15%]
        │                   │                       │
        │ SPACE=chụp ảnh    │ 70% Train             │
        ▼                   ▼                       ▼
  [dataset/             [Freeze backbone]    [Model gốc] ──► acc_orig
   Anger/               [Unfreeze 2 blocks]  [Fine-tuned] ──► acc_ft
   Happy/               [Adam lr=1e-4]             │
   ...]                 [15 epoch]                 ▼
                               │             [Confusion Matrix]
                               ▼             [Classification Report]
                        [models/             [Training Curve]
                         finetuned.pt]
```

---

## 4. Các chức năng chính

### Chế độ 1: Webcam Real-time (`--mode webcam`)

Nhận diện cảm xúc liên tục từ webcam, áp dụng skip-frame để cân bằng FPS và độ trễ AI.

```bash
python main.py --mode webcam
python main.py --mode webcam --weights models/finetuned.pt   # dùng model đã fine-tune
python main.py --mode webcam --camera 1                      # camera ID 1
```

**Phím điều khiển:**
- `q` / `ESC` — Thoát
- `s` — Chụp ảnh màn hình lưu vào `data/`
- `r` — Reset temporal smoothing buffer

---

### Chế độ 2: Ảnh tĩnh (`--mode image`)

Nhận diện cảm xúc từ file ảnh, hiển thị bounding box và biểu đồ xác suất 7 cảm xúc.

```bash
python main.py --mode image --input face.jpg
python main.py --mode image --input face.jpg --output result.jpg
python main.py --mode image --input face.jpg --weights models/finetuned.pt
```

---

### Chế độ 3: Video (`--mode video`)

Xử lý từng frame của video, hiển thị nhận diện cảm xúc kèm thanh tiến trình.

```bash
python main.py --mode video --input clip.mp4
python main.py --mode video --input clip.mp4 --output result.mp4   # lưu video kết quả
```

**Phím điều khiển:**
- `q` / `ESC` — Dừng
- `SPACE` — Tạm dừng / Tiếp tục
- `s` — Chụp ảnh frame hiện tại

---

### Chế độ 4: Fine-tuning (`--mode finetune`)

Huấn luyện lại các lớp cuối của EfficientNet trên dataset tự thu thập.

```bash
python main.py --mode finetune --dataset dataset/ --epochs 15
python main.py --mode finetune --dataset dataset/ --epochs 20 --lr 5e-5
```

---

### Chế độ 5: Đánh giá (`--mode evaluate`)

So sánh hiệu năng model gốc vs fine-tuned, xuất confusion matrix và training curve.

```bash
python main.py --mode evaluate --dataset dataset/ --weights models/finetuned.pt
```

---

### Thu thập Dataset (`collect_data.py`)

Script chụp ảnh tự động từ webcam, phân loại vào đúng thư mục cảm xúc.

```bash
python collect_data.py
python collect_data.py --class Happy --count 40
```

**Phím điều khiển:**
- `1`–`7` — Chọn class (1=Anger, 2=Disgust, 3=Fear, 4=Happiness, 5=Neutral, 6=Sadness, 7=Surprise)
- `SPACE` — Chụp ảnh
- `n` / `p` — Class tiếp / trước
- `q` — Thoát và xem tổng kết

---

## 5. Giải thích thuật toán & code

### 5.1. Phát hiện khuôn mặt — Haar Cascade

**File:** [`src/emotion_engine.py`](src/emotion_engine.py) — hàm `detect_faces()`

Haar Cascade (Viola & Jones, 2001) là thuật toán phát hiện đối tượng cổ điển dựa trên:

**Haar-like features:** Các đặc trưng hình chữ nhật phản ánh sự chênh lệch độ sáng:
```
Vùng mắt thường tối hơn vùng má:
┌───────────────┐
│  Tối  │ Sáng  │   → Haar feature = sum(Sáng) - sum(Tối)
│ (mắt) │(gò má)│
└───────────────┘
```

**Integral Image:** Tính tổng pixel trong bất kỳ hình chữ nhật nào trong O(1):
```
I(x,y) = Σ p(x',y')  với x'≤x, y'≤y
```

**AdaBoost Cascade:** Xếp tầng các bộ phân loại yếu (weak classifiers). Vùng ảnh bị loại ngay từ giai đoạn đầu nếu không phải mặt → rất nhanh.

```python
# Code trong detect_faces():
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = self.face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,   # Thu nhỏ ảnh 10% mỗi bước để detect mặt nhiều kích thước
    minNeighbors=5,    # Cần 5 vùng lân cận xác nhận → giảm false positive
    minSize=(30, 30)   # Bỏ qua mặt nhỏ hơn 30x30px
)
```

**Ưu điểm:** Tốc độ cực nhanh (~1ms/frame trên CPU)  
**Nhược điểm:** Kém hơn với mặt nghiêng, ánh sáng xấu. Trong tương lai có thể thay bằng MTCNN hoặc RetinaFace.

---

### 5.2. Tiền xử lý CLAHE

**File:** [`src/emotion_engine.py`](src/emotion_engine.py) — hàm `_preprocess_face()`

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** cải thiện độ tương phản cục bộ để làm rõ các đặc trưng biểu cảm (nếp nhăn, đường viền miệng).

**Cách hoạt động:**
```
Ảnh gốc (có thể tối/sáng không đều)
        │
        ▼
Chuyển sang Grayscale (1 kênh)
        │
        ▼
Chia thành lưới 8×8 tile
        │
        ▼
Với mỗi tile: cân bằng histogram cục bộ
(clipLimit=2.0 → giới hạn khuếch đại nhiễu)
        │
        ▼
Nội suy bilinear giữa các tile → ảnh mượt
        │
        ▼
Chuyển lại RGB (HSEmotion yêu cầu 3 kênh)
```

```python
# Code thực tế:
self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
equalized = self.clahe.apply(gray)          # Cân bằng sáng cục bộ
processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
```

**Tại sao quan trọng?** Webcam trong phòng học thường có ánh sáng không đều (cửa sổ chiều, đèn từ phía sau). CLAHE giúp model nhận diện ổn định hơn trong các điều kiện ánh sáng khó.

---

### 5.3. Nhận diện cảm xúc — EfficientNet Inference

**File:** [`src/emotion_engine.py`](src/emotion_engine.py) — hàm `predict_emotion()`

```python
# Toàn bộ pipeline cho 1 khuôn mặt:

# Bước 1: Tiền xử lý CLAHE
processed_face = self._preprocess_face(face_img)

# Bước 2: Forward pass qua EfficientNet → 8 xác suất (softmax)
emotion_name_raw, scores = self.emotion_recognizer.predict_emotions(
    processed_face, logits=False   # logits=False → trả về xác suất [0,1]
)
# scores = [0.05, 0.02, 0.08, 0.03, 0.72, 0.05, 0.03, 0.02]
#           Anger Cont  Disg  Fear  Happ  Neut  Sad   Surp

# Bước 3: Temporal Smoothing
buf.append(scores)                    # Thêm vào buffer 10 frame
avg_scores = np.mean(buf, axis=0)    # Trung bình cộng → giảm nhiễu

# Bước 4: Tìm cảm xúc có xác suất cao nhất sau smoothing
max_idx = np.argmax(avg_scores)
max_conf = float(avg_scores[max_idx])
emotion_candidate = self.emotion_labels[max_idx]

# Bước 5: Adaptive Threshold
threshold = self.thresholds.get(emotion_candidate, 0.40)
if max_conf < threshold:
    final_emotion = 'Neutral'   # Nếu không đủ tự tin → mặc định Neutral
```

---

### 5.4. Per-Face Temporal Smoothing

**File:** [`src/emotion_engine.py`](src/emotion_engine.py)

**Vấn đề:** EfficientNet đôi khi dao động giữa các frame liền kề do noise ảnh, ánh sáng thay đổi nhẹ, hoặc biểu cảm chuyển tiếp. Điều này gây ra hiện tượng nhãn cảm xúc "nhấp nháy".

**Giải pháp:** Buffer riêng cho mỗi khuôn mặt, lấy trung bình 10 frame:

```
Frame 1: scores = [0.1, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0]  → Happy
Frame 2: scores = [0.2, 0.0, 0.0, 0.0, 0.6, 0.2, 0.0, 0.0]  → Happy (noise)
Frame 3: scores = [0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]  → Happy
...
avg_10  = [0.1, 0.0, 0.0, 0.0, 0.75, 0.15, 0.0, 0.0] → Happy (ổn định!)
```

**Định danh khuôn mặt theo lưới (Face Key):**

```python
def _get_face_key(self, x, y, w, h):
    cx = ((x + w//2) // 60) * 60   # Làm tròn vào lưới 60px
    cy = ((y + h//2) // 60) * 60
    return (cx, cy)
```

Ví dụ: Khuôn mặt ở vị trí (125, 83) → key = (120, 60)  
Nếu face dịch chuyển nhẹ sang (135, 91) → key = (120, 60) (**vẫn cùng key**)  
→ Buffer không bị reset khi đầu người hơi nhúc nhích

**Multi-face:** Với 2 người trong frame:
```
Face A (center ~120,80): key=(120,60) → buffer_A = [scores_A1, scores_A2, ...]
Face B (center ~400,90): key=(360,60) → buffer_B = [scores_B1, scores_B2, ...]
```
Hai khuôn mặt có buffer độc lập, không trộn lẫn dữ liệu.

---

### 5.5. Adaptive Threshold

**File:** [`src/emotion_engine.py`](src/emotion_engine.py)

Thay vì dùng ngưỡng cố định 0.5 cho tất cả cảm xúc, hệ thống dùng ngưỡng riêng:

```python
self.thresholds = {
    'Happiness': 0.50,  # Cao — tránh nhận nhầm khi cười trong lúc nói chuyện
    'Sadness':   0.30,  # Thấp — buồn biểu hiện kín đáo, cần nhạy hơn
    'Surprise':  0.35,  # Thấp — khoảnh khắc ngạc nhiên thoáng qua rất nhanh
    'Anger':     0.40,
    'Neutral':   0.35
}
```

**Logic:** Nếu xác suất cao nhất < ngưỡng → kết quả là `Neutral` (hệ thống không chắc).  
Điều này giảm "false positive" — tránh nhận sai cảm xúc với độ tin cậy thấp.

---

### 5.6. Skip-Frame Strategy

**File:** [`main.py`](main.py) — hàm `run_webcam()` và `run_video()`

```python
frame_count += 1

if frame_count % skip_frames != 0:
    # Frame bị skip: VẼ LẠI kết quả cũ (last_results)
    display = _draw_webcam_overlay(frame, last_results, ...)
else:
    # Frame xử lý AI
    last_results = engine.process_frame(frame)
    display = _draw_webcam_overlay(frame, last_results, ...)
```

**Tại sao quan trọng?** EfficientNet mất ~50–100ms/lần predict. Nếu mỗi frame đều chạy AI → FPS chỉ đạt 10–20. Với skip_frames=2, AI chạy mỗi 2 frame nhưng UI vẫn hiển thị 30+ FPS (frame bị skip dùng kết quả cũ → không nhấp nháy).

---

## 6. Quy trình Fine-tuning

**File:** [`src/fine_tuner.py`](src/fine_tuner.py)

### 6.1. Chiến lược Freeze/Unfreeze

```
EfficientNet-B0 Architecture:
┌──────────────────────────────────────────────┐
│  Stem Conv (3→32 filters)           [FROZEN] │
│  MBConv Block 1 (32→16)             [FROZEN] │
│  MBConv Block 2 (16→24)             [FROZEN] │
│  MBConv Block 3 (24→40)             [FROZEN] │
│  MBConv Block 4 (40→80)             [FROZEN] │
│  MBConv Block 5 (80→112)            [FROZEN] │
│  MBConv Block 6 (112→192)        [TRAINABLE] │ ← Unfreeze
│  MBConv Block 7 (192→320)        [TRAINABLE] │ ← Unfreeze
│  Head + Classifier (320→8)       [TRAINABLE] │ ← Unfreeze
└──────────────────────────────────────────────┘

Tham số frozen:   ~4.8M  (80%)  — giữ kiến thức từ 450k ảnh
Tham số trainable: ~0.5M  (20%)  — học thêm từ dataset nhỏ
```

**Tại sao freeze phần lớn?**
- Các lớp đầu học được đặc trưng tổng quát (cạnh, texture, hình dạng)
- Với dataset nhỏ (~200 ảnh), train toàn bộ → **overfitting**
- Chỉ fine-tune lớp cuối: model giữ kiến thức cũ + học thêm đặc điểm riêng của dataset nhỏ

### 6.2. Dataset Split

```python
# 70% train / 15% val / 15% test — seed cố định để tái lặp
train_ds, val_ds, test_ds = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
```

**Test set** được tách ra ngay từ đầu và **không bao giờ** được dùng trong quá trình train hay chọn hyperparameter → đảm bảo đánh giá công bằng.

### 6.3. Training Loop

```
Mỗi epoch:
  ┌─ TRAIN PHASE ──────────────────────────────────┐
  │  for batch in train_loader:                     │
  │    output = model(images)      # forward pass   │
  │    loss = CrossEntropy(output, labels)          │
  │    loss.backward()             # backprop       │
  │    optimizer.step()            # update weights │
  └─────────────────────────────────────────────────┘
  
  ┌─ VALIDATION PHASE ─────────────────────────────┐
  │  with torch.no_grad():                          │
  │    output = model(val_images)                   │
  │    val_loss, val_acc = compute_metrics(...)     │
  └─────────────────────────────────────────────────┘
  
  if val_acc > best_val_acc:
      save checkpoint → models/finetuned.pt
  
  scheduler.step()   # giảm lr 50% mỗi 5 epoch
```

**Optimizer:** Adam với `lr=1e-4`  
**Loss:** CrossEntropyLoss (tương đương Softmax + NLLLoss)  
**Scheduler:** StepLR (step=5, gamma=0.5) → lr: 1e-4 → 5e-5 → 2.5e-5 ...

---

## 7. Đánh giá mô hình

**File:** [`src/evaluator.py`](src/evaluator.py)

### 7.1. Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **Accuracy** | TP+TN / Total | Tỉ lệ dự đoán đúng tổng thể |
| **Precision** | TP / (TP+FP) | Trong các dự đoán là X, bao nhiêu thực sự là X |
| **Recall** | TP / (TP+FN) | Trong tất cả ảnh X thật, model nhận ra được bao nhiêu |
| **F1-score** | 2·P·R / (P+R) | Trung bình điều hòa Precision & Recall |

### 7.2. Confusion Matrix

Ma trận N×N (N = số class) hiển thị chi tiết sự nhầm lẫn:

```
                  DỰ ĐOÁN
              Ang  Dis  Fea  Hap  Neu  Sad  Sur
         Ang [  8    0    1    0    0    1    0 ]
         Dis [  0    9    0    0    1    0    0 ]
THỰC     Fea [  0    0    6    0    0    0    4 ] ← Fear hay nhầm Surprise
TẾ       Hap [  0    0    0   10    0    0    0 ]
         Neu [  0    1    0    1    8    0    0 ]
         Sad [  1    0    0    0    2    7    0 ]
         Sur [  0    0    2    0    0    0    8 ]

Đường chéo = đoán đúng  |  Ngoài đường chéo = nhầm lẫn
```

**Cách đọc:** Hàng = nhãn thật, Cột = nhãn dự đoán. Ô `[Fea, Sur]=4` có nghĩa 4 ảnh sợ hãi bị nhận nhầm là ngạc nhiên (hợp lý vì 2 cảm xúc đều có mắt mở to, miệng há).

### 7.3. Training Curve

Biểu đồ loss và accuracy theo từng epoch:
- **Train Loss giảm** → model đang học
- **Val Loss giảm** → model tổng quát hóa tốt
- **Val Loss tăng trong khi Train Loss giảm** → overfitting (dừng sớm)

---

## 8. Cài đặt và chạy

### 8.1. Yêu cầu hệ thống

- Python 3.8+
- Webcam (cho mode webcam)
- RAM tối thiểu 4GB
- CPU (không cần GPU — EfficientNet-B0 chạy tốt trên CPU)

### 8.2. Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/ncthangtt/khaithac.git
cd khaithac

# 2. Tạo môi trường ảo
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Linux/macOS

# 3. Cài thư viện
pip install -r requirements.txt
```

> Lần đầu chạy, hệ thống tự tải model (~25MB) từ internet về `C:\Users\USERNAME\.hsemotion\`

### 8.3. Luồng làm việc đề xuất

```bash
# Bước 1: Thu thập ảnh dataset
python collect_data.py --count 40

# Bước 2: Fine-tune model (15–30 phút trên CPU)
python main.py --mode finetune --dataset dataset/ --epochs 15

# Bước 3: Đánh giá kết quả
python main.py --mode evaluate --dataset dataset/ --weights models/finetuned.pt

# Bước 4: Demo
python main.py --mode webcam --weights models/finetuned.pt
python main.py --mode image  --input face.jpg --weights models/finetuned.pt
python main.py --mode video  --input lecture.mp4 --output result.mp4
```

---

## 9. Tham số dòng lệnh

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `--mode` | `webcam` | Chế độ: `webcam` / `image` / `video` / `finetune` / `evaluate` |
| `--model` | `enet_b0_8_best_afew` | Model HSEmotion base |
| `--weights` | `None` | File `.pt` fine-tuned weights |
| `--camera` | `0` | ID webcam |
| `--input` | `None` | Đường dẫn ảnh hoặc video đầu vào |
| `--output` | `None` | Lưu kết quả ra file |
| `--dataset` | `None` | Thư mục dataset (finetune & evaluate) |
| `--epochs` | `15` | Số epoch fine-tune |
| `--lr` | `1e-4` | Learning rate |
| `--batch-size` | `16` | Batch size |
| `--save-weights` | `models/finetuned.pt` | Đường dẫn lưu model fine-tuned |

---

*Đại học Bách khoa Hà Nội — Môn Thị giác Máy tính*

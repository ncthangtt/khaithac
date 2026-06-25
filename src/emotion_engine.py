"""
==================================================
MODULE: emotion_engine.py
Chức năng: Nhận dạng cảm xúc bằng HSEmotion
Mô tả: Sử dụng EfficientNet từ thư viện HSEmotion.
        Hỗ trợ load fine-tuned weights tùy chỉnh.
Đại học Bách khoa Hà Nội
==================================================
"""

import cv2
import numpy as np
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer
from collections import deque
from typing import Tuple, List, Optional, Dict


class EmotionEngine:
    """
    Lớp nhận dạng cảm xúc nâng cao sử dụng HSEmotion (EfficientNet)
    
    Tích hợp các kỹ thuật:
    - Temporal Smoothing: Làm mịn kết quả theo thời gian (trung bình 10 frame).
    - CLAHE Preprocessing: Cân bằng ánh sáng cục bộ để làm rõ đặc trưng cơ mặt.
    - Adaptive Thresholding: Ngưỡng tin cậy riêng cho từng loại cảm xúc.
    """

    def __init__(self, model_name: str = 'enet_b0_8_best_afew',
                 weights_path: str = None):
        """
        Khởi tạo Emotion Engine

        Args:
            model_name:   Tên model HSEmotion làm base
            weights_path: Đường dẫn file .pt fine-tuned (None = dùng model gốc)
        """
        self.model_name = model_name
        self.emotion_recognizer = HSEmotionRecognizer(model_name=model_name)
        self.face_cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_classifier = cv2.CascadeClassifier(self.face_cascade)

        # Danh sách các cảm xúc (theo thứ tự output của mô hình 8 lớp)
        self.emotion_labels = [
            'Anger', 'Contempt', 'Disgust', 'Fear',
            'Happiness', 'Neutral', 'Sadness', 'Surprise'
        ]

        # Load fine-tuned weights nếu được cung cấp
        if weights_path is not None:
            self._load_finetuned_weights(weights_path)

        # 1. Temporal Smoothing: Mỗi khuôn mặt có buffer riêng để tránh trộn lẫn dữ liệu
        #    Key = (cx, cy) — tâm khuôn mặt được lượng tử hóa vào lưới 60px
        self.face_buffers: Dict[tuple, deque] = {}
        self.BUFFER_SIZE = 10   # Lưu 10 frame gần nhất cho mỗi khuôn mặt
        self.GRID_SIZE   = 60   # Kích thước ô lưới (px) — chịu được rung nhỏ < 60px

        # 2. Adaptive Thresholds: Ngưỡng riêng cho từng cảm xúc để tăng độ nhạy
        self.thresholds = {
            'Happiness': 0.50,
            'Sadness': 0.30,
            'Surprise': 0.35,
            'Anger': 0.40,
            'Neutral': 0.35
        }

        # 3. CLAHE: Bộ cân bằng ánh sáng thích nghi
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        mode = f"Fine-tuned ({weights_path})" if weights_path else "Pretrained gốc"
        print(f"✓ EmotionEngine sẵn sàng | Model: {model_name} | Weights: {mode}")

    def _load_finetuned_weights(self, weights_path: str):
        """
        Load fine-tuned weights vào model hiện tại.

        Args:
            weights_path: Đường dẫn file .pt được lưu bởi fine_tuner.py
        """
        import os
        if not os.path.isfile(weights_path):
            print(f"⚠️  Không tìm thấy file weights: {weights_path}")
            print("   Tiếp tục với model gốc (pretrained).")
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(weights_path, map_location=device)
        self.emotion_recognizer.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        self.emotion_recognizer.model.to(device)
        val_acc = checkpoint.get('val_acc', 0)
        epoch = checkpoint.get('epoch', '?')
        print(f"✓ Đã load fine-tuned weights: {weights_path}")
        print(f"  (Epoch {epoch} | Val accuracy: {val_acc:.1f}%)")

    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý nâng cao: Chuyển sang ảnh xám, áp dụng CLAHE và quay lại RGB
        """
        if face_img.size == 0:
            return face_img
            
        # Chuyển sang Grayscale để xử lý độ tương phản
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng CLAHE làm rõ các nếp nhăn và đặc trưng biểu cảm
        equalized = self.clahe.apply(gray)
        
        # HSEmotion yêu cầu ảnh RGB 3 kênh
        processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        return processed

    def _get_face_key(self, x: int, y: int, w: int, h: int) -> tuple:
        """
        Tạo key nhận diện khuôn mặt dựa trên tâm, lượng tử hóa vào lưới
        để chịu được dao động vị trí nhỏ (< GRID_SIZE px).
        """
        cx = ((x + w // 2) // self.GRID_SIZE) * self.GRID_SIZE
        cy = ((y + h // 2) // self.GRID_SIZE) * self.GRID_SIZE
        return (cx, cy)

    def predict_emotion(self, face_img: np.ndarray,
                        face_key: tuple = (0, 0)) -> Tuple[str, float, np.ndarray]:
        """
        Nhận dạng cảm xúc với xử lý làm mịn và ngưỡng thích nghi.
        Mỗi khuôn mặt (face_key) dùng buffer riêng để tránh trộn lẫn.
        """
        try:
            # Bước 1: Tiền xử lý ảnh mặt
            processed_face = self._preprocess_face(face_img)

            # Bước 2: Dự đoán lấy Raw Probabilities (logits=False)
            emotion_name_raw, scores = self.emotion_recognizer.predict_emotions(
                processed_face, logits=False
            )

            # Bước 3: Lấy (hoặc tạo) buffer riêng cho khuôn mặt này
            if face_key not in self.face_buffers:
                self.face_buffers[face_key] = deque(maxlen=self.BUFFER_SIZE)
            buf = self.face_buffers[face_key]
            buf.append(scores)

            # Bước 4: Temporal Smoothing (Trung bình cộng trong buffer)
            avg_scores = np.mean(buf, axis=0)

            # Bước 5: Tìm cảm xúc có xác suất trung bình cao nhất
            max_idx = np.argmax(avg_scores)
            max_conf = float(avg_scores[max_idx])
            emotion_candidate = self.emotion_labels[max_idx]

            # Bước 6: Áp dụng Adaptive Threshold & Default Neutral logic
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

            return final_emotion, max_conf, avg_scores

        except Exception as e:
            print(f"❌ Lỗi khi predict emotion: {e}")
            return "Neutral", 0.0, np.zeros(len(self.emotion_labels))

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Phát hiện khuôn mặt bằng Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Xử lý toàn bộ frame: Phát hiện mặt + Nhận diện cảm xúc.
        Mỗi khuôn mặt được theo dõi bằng face_key độc lập."""
        results = []
        faces = self.detect_faces(frame)

        # Dọn buffer của khuôn mặt không còn xuất hiện (giữ bộ nhớ sạch)
        active_keys = set()
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.shape[0] >= 48 and face_img.shape[1] >= 48:
                active_keys.add(self._get_face_key(x, y, w, h))
        stale_keys = set(self.face_buffers.keys()) - active_keys
        for k in stale_keys:
            del self.face_buffers[k]

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



        return results

    def get_emotion_distribution(self, probabilities: np.ndarray) -> Dict[str, float]:
        """
        Chuyển probabilities thành dictionary {emotion: probability}

        Args:
            probabilities: Mảng xác suất từ model

        Returns:
            Dictionary {emotion_name: probability}
        """
        return {
            emotion: float(prob)
            for emotion, prob in zip(self.emotion_labels, probabilities)
        }


# Test code (chỉ chạy khi file này được chạy trực tiếp)
if __name__ == "__main__":
    print("=== TEST EMOTION ENGINE MODULE ===\n")

    # Tạo emotion engine
    engine = EmotionEngine(model_name='enet_b0_8_best_afew')

    print("\n✓ Engine đã sẵn sàng")
    print(f"✓ Danh sách cảm xúc: {engine.emotion_labels}")

    # Test với webcam
    print("\n⏳ Mở webcam để test...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không thể mở webcam. Kết thúc test.")
        exit(1)

    print("✓ Webcam đã mở")
    print("\nNhấn 'q' để thoát\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Xử lý mỗi 3 frame (tăng tốc độ)
        if frame_count % 3 != 0:
            cv2.imshow("Emotion Test", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # Nhận dạng cảm xúc
        results = engine.process_frame(frame)

        # Vẽ kết quả lên khung hình
        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']

            # Vẽ box
            color = (0, 255, 0) if emotion == 'Happy' else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Vẽ text
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hiển thị số lượng khuôn mặt
        cv2.putText(frame, f"Faces: {len(results)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Hiển thị
        cv2.imshow("Emotion Test", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Giải phóng
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✓ Đã xử lý {frame_count} khung hình")
    print("=== TEST HOÀN TẤT ===")

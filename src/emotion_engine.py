"""
==================================================
MODULE: emotion_engine.py
Chức năng: Nhận dạng cảm xúc bằng HSEmotion
Mô tả: Sử dụng EfficientNet từ thư viện HSEmotion
Đại học Bách khoa Hà Nội
==================================================
"""

import cv2
import numpy as np
from hsemotion.facial_emotions import HSEmotionRecognizer
from typing import Tuple, List, Optional, Dict


class EmotionEngine:
    """
    Lớp nhận dạng cảm xúc sử dụng HSEmotion (EfficientNet)

    Chức năng chính:
    - Load mô hình pre-trained (EfficientNet-B0 hoặc B2)
    - Phát hiện khuôn mặt
    - Nhận dạng cảm xúc real-time
    """

    def __init__(self, model_name: str = 'enet_b0_8_best_afew'):
        """
        Khởi tạo Emotion Engine

        Args:
            model_name: Tên mô hình HSEmotion
                - 'enet_b0_8_best_afew': EfficientNet-B0 (nhanh, nhẹ)
                - 'enet_b0_8_best_vgaf': EfficientNet-B0 (chính xác hơn)
                - 'enet_b2_8': EfficientNet-B2 (chính xác nhất, chậm hơn)
        """
        self.model_name = model_name
        self.emotion_recognizer = None
        self.face_cascade = None

        # Danh sách các cảm xúc (theo thứ tự output của mô hình 8 lớp)
        self.emotion_labels = [
            'Anger',      # Tức giận
            'Contempt',   # Khinh bỉ
            'Disgust',    # Ghê tởm
            'Fear',       # Sợ hãi
            'Happiness',  # Vui vẻ
            'Neutral',    # Trung tính
            'Sadness',    # Buồn
            'Surprise'    # Ngạc nhiên
        ]

        # Load models
        self._load_models()

    def _load_models(self):
        """
        Load mô hình nhận dạng cảm xúc và phát hiện khuôn mặt
        """
        try:
            # Load HSEmotion model
            print(f"⏳ Đang tải mô hình HSEmotion: {self.model_name}...")
            self.emotion_recognizer = HSEmotionRecognizer(model_name=self.model_name)
            print(f"✓ Đã tải mô hình HSEmotion thành công!")

            # Load Haar Cascade cho phát hiện khuôn mặt
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                raise Exception("Không thể load Haar Cascade")

            print("✓ Đã tải Haar Cascade thành công!")

        except Exception as e:
            print(f"❌ Lỗi khi tải models: {e}")
            raise

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt trong khung hình

        Args:
            frame: Khung hình BGR từ OpenCV

        Returns:
            List[(x, y, w, h)] - Danh sách bounding boxes
        """
        # Chuyển sang grayscale để tăng tốc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return faces

    def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Nhận dạng cảm xúc từ ảnh khuôn mặt

        Args:
            face_img: Ảnh khuôn mặt (BGR)

        Returns:
            (emotion_name, confidence, probabilities_array)
        """
        try:
            # HSEmotion cần ảnh RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # Predict
            # LƯU Ý: Thư viện HSEmotion trả về (emotion_name, scores)
            # emotion_name là một STRING (ví dụ: 'Happiness'), không phải INDEX
            emotion_name, scores = self.emotion_recognizer.predict_emotions(
                face_rgb,
                logits=False  # Trả về probabilities thay vì logits
            )

            # Tính độ tin cậy (confidence) là giá trị lớn nhất trong mảng scores
            confidence = float(np.max(scores))

            # Chuyển đổi tên cảm xúc sang định dạng chuẩn của project nếu cần
            # (Ví dụ: 'Happiness' -> 'Happy', 'Anger' -> 'Angry')
            mapping = {
                'Anger': 'Angry',
                'Happiness': 'Happy',
                'Sadness': 'Sad',
                'Contempt': 'Disgust'
            }
            emotion_name = mapping.get(emotion_name, emotion_name)

            return emotion_name, confidence, scores

        except Exception as e:
            print(f"❌ Lỗi khi predict emotion: {e}")
            return "Unknown", 0.0, np.zeros(len(self.emotion_labels))

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Xử lý một khung hình: phát hiện khuôn mặt + nhận dạng cảm xúc

        Args:
            frame: Khung hình BGR từ OpenCV

        Returns:
            List[{
                'bbox': (x, y, w, h),
                'emotion': str,
                'confidence': float,
                'probabilities': np.ndarray
            }]
        """
        results = []

        # Phát hiện khuôn mặt
        faces = self.detect_faces(frame)

        # Nhận dạng cảm xúc cho từng khuôn mặt
        for (x, y, w, h) in faces:
            # Crop khuôn mặt
            face_img = frame[y:y+h, x:x+w]

            # Bỏ qua nếu khuôn mặt quá nhỏ
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                continue

            # Predict emotion
            emotion, confidence, probs = self.predict_emotion(face_img)

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

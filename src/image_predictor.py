"""
==================================================
MODULE: image_predictor.py
Chức năng: Nhận diện cảm xúc từ ảnh tĩnh
Mô tả: Load ảnh → detect face → predict emotion → hiển thị kết quả
Đại học Bách khoa Hà Nội
==================================================
"""

import cv2
import numpy as np
import os
from src.emotion_engine import EmotionEngine


class ImagePredictor:
    """
    Nhận diện cảm xúc từ file ảnh tĩnh (.jpg, .png, ...).
    Hỗ trợ cả model gốc và model fine-tuned.
    """

    COLOR_MAP = {
        'Happy':   (0, 255, 0),
        'Sad':     (255, 80, 80),
        'Angry':   (0, 0, 255),
        'Surprise':(0, 165, 255),
        'Neutral': (180, 180, 180),
        'Fear':    (180, 0, 180),
        'Disgust': (0, 160, 160),
    }

    def __init__(self, model_name: str = 'enet_b0_8_best_afew',
                 weights_path: str = None):
        """
        Args:
            model_name:   Tên model HSEmotion
            weights_path: Đường dẫn file .pt fine-tuned (None = dùng model gốc)
        """
        self.engine = EmotionEngine(
            model_name=model_name,
            weights_path=weights_path
        )

    def predict(self, image_path: str,
                output_path: str = None,
                show: bool = True) -> list:
        """
        Nhận diện cảm xúc trong ảnh.

        Args:
            image_path:  Đường dẫn ảnh đầu vào
            output_path: Lưu ảnh kết quả (None = không lưu)
            show:        Hiển thị cửa sổ xem (True/False)

        Returns:
            List kết quả nhận diện [{emotion, confidence, bbox, probabilities}]
        """
        if not os.path.isfile(image_path):
            print(f"❌ Không tìm thấy ảnh: {image_path}")
            return []

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Không thể đọc ảnh: {image_path}")
            return []

        results = self.engine.process_frame(frame)

        if not results:
            print("⚠️  Không phát hiện khuôn mặt trong ảnh.")
            if show:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            print(f"✓ Phát hiện {len(results)} khuôn mặt:")
            for i, result in enumerate(results):
                frame = self._draw_result(frame, result, i)
                emotion = result['emotion']
                conf = result['confidence']
                print(f"  Khuôn mặt #{i+1}: {emotion} ({conf:.1%})")

        # Vẽ thanh xác suất (dùng kết quả đầu tiên nếu có)
        if results:
            frame = self._draw_prob_bars(frame, results[0])

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            cv2.imwrite(output_path, frame)
            print(f"✓ Đã lưu ảnh kết quả: {output_path}")

        if show:
            window_name = "Emotion Detection - Image Mode"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            h, w = frame.shape[:2]
            # Resize cửa sổ cho vừa màn hình
            cv2.resizeWindow(window_name, min(w, 1024), min(h, 720))
            cv2.imshow(window_name, frame)
            print("📌 Nhấn phím bất kỳ để thoát...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results

    # ──────────────────────────────────────────────
    # Drawing helpers
    # ──────────────────────────────────────────────

    def _draw_result(self, frame: np.ndarray, result: dict, idx: int) -> np.ndarray:
        """Vẽ bounding box và label lên frame."""
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        confidence = result['confidence']
        color = self.COLOR_MAP.get(emotion, (255, 255, 255))

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        label = f"{emotion}  {confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - lh - 12), (x + lw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame

    def _draw_prob_bars(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Vẽ thanh xác suất top-7 ở góc phải."""
        probs = result['probabilities']
        labels = self.engine.emotion_labels  # ['Anger', 'Contempt', ...]

        # Hiển thị map sang tên thân thiện
        display_map = {
            'Anger': 'Angry', 'Happiness': 'Happy',
            'Sadness': 'Sad', 'Contempt': 'Disgust'
        }

        h, w = frame.shape[:2]
        bar_x = w - 220
        bar_max_w = 190
        bar_h = 22
        padding = 6

        # Nền mờ
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x - 8, 10),
                      (w - 5, len(labels) * (bar_h + padding) + 20),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Sắp xếp theo xác suất giảm dần để highlight
        sorted_idx = np.argsort(probs)[::-1]

        for rank, orig_i in enumerate(sorted_idx):
            name = display_map.get(labels[orig_i], labels[orig_i])
            prob = float(probs[orig_i])
            y_pos = 15 + rank * (bar_h + padding)

            # Màu bar: highlight cảm xúc cao nhất
            bar_color = self.COLOR_MAP.get(name, (100, 180, 100))
            if rank == 0:
                bar_color = tuple(min(255, c + 60) for c in bar_color)

            bar_len = int(prob * bar_max_w)
            cv2.rectangle(frame, (bar_x, y_pos),
                          (bar_x + bar_len, y_pos + bar_h),
                          bar_color, -1)
            cv2.putText(frame,
                        f"{name[:8]}: {prob:.1%}",
                        (bar_x + 4, y_pos + bar_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)

        return frame

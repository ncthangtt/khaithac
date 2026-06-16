"""
==================================================
MODULE: capture.py
Chức năng: Xử lý camera và capture video
Mô tả: Mở webcam, đọc khung hình real-time
Đại học Bách khoa Hà Nội
==================================================
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CameraCapture:
    """
    Lớp quản lý camera và capture video

    Chức năng chính:
    - Mở/đóng webcam
    - Đọc khung hình real-time
    - Hiển thị video với overlay text
    """

    def __init__(self, camera_id: int = 0):
        """
        Khởi tạo camera capture

        Args:
            camera_id: ID của camera (mặc định: 0 - webcam mặc định)
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False

    def open_camera(self) -> bool:
        """
        Mở kết nối với camera

        Returns:
            True nếu mở thành công, False nếu thất bại
        """
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"❌ KHÔNG THỂ MỞ CAMERA ID: {self.camera_id}")
            return False

        # Thiết lập độ phân giải (640x480 cho hiệu suất tốt)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_opened = True
        print(f"✓ Đã mở camera ID: {self.camera_id}")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Đọc một khung hình từ camera

        Returns:
            (success, frame) - success=True nếu đọc thành công
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            print("❌ Không thể đọc khung hình từ camera")
            return False, None

        return True, frame

    def release_camera(self):
        """
        Đóng kết nối camera và giải phóng tài nguyên
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("✓ Đã đóng camera")

    def draw_text(self, frame: np.ndarray, text: str,
                  position: Tuple[int, int] = (10, 30),
                  font_scale: float = 0.7,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        Vẽ text lên khung hình

        Args:
            frame: Khung hình gốc
            text: Nội dung text
            position: Vị trí (x, y) của text
            font_scale: Kích thước font
            color: Màu sắc (B, G, R)
            thickness: Độ dày nét chữ

        Returns:
            Khung hình đã có text
        """
        # Vẽ background đen cho text (dễ đọc hơn)
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            frame,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 5, position[1] + 5),
            (0, 0, 0),
            -1
        )

        # Vẽ text
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness
        )

        return frame

    def draw_emotion_box(self, frame: np.ndarray,
                         emotion: str,
                         confidence: float,
                         bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Vẽ khung bao quanh khuôn mặt với thông tin cảm xúc

        Args:
            frame: Khung hình gốc
            emotion: Tên cảm xúc
            confidence: Độ tin cậy (0.0 - 1.0)
            bbox: Bounding box (x, y, w, h)

        Returns:
            Khung hình đã vẽ box
        """
        x, y, w, h = bbox

        # Chọn màu dựa trên cảm xúc
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

        # Vẽ khung chữ nhật
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Vẽ text cảm xúc và confidence
        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        return frame

    def show_frame(self, frame: np.ndarray, window_name: str = "Emotion Detection"):
        """
        Hiển thị khung hình trong cửa sổ

        Args:
            frame: Khung hình cần hiển thị
            window_name: Tên cửa sổ
        """
        cv2.imshow(window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        """
        Chờ phím bấm

        Args:
            delay: Thời gian chờ (ms). 1ms = real-time, 0 = chờ vô hạn

        Returns:
            Mã phím được bấm (-1 nếu không có phím nào)
        """
        return cv2.waitKey(delay)

    @staticmethod
    def destroy_all_windows():
        """
        Đóng tất cả cửa sổ OpenCV
        """
        cv2.destroyAllWindows()

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Lấy kích thước khung hình hiện tại

        Returns:
            (width, height)
        """
        if not self.is_opened or self.cap is None:
            return (0, 0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return (width, height)

    def get_fps(self) -> float:
        """
        Lấy FPS của camera

        Returns:
            FPS (frames per second)
        """
        if not self.is_opened or self.cap is None:
            return 0.0

        return self.cap.get(cv2.CAP_PROP_FPS)


# Test code (chỉ chạy khi file này được chạy trực tiếp)
if __name__ == "__main__":
    print("=== TEST CAMERA CAPTURE MODULE ===\n")

    # Tạo camera capture object
    camera = CameraCapture(camera_id=0)

    # Mở camera
    if not camera.open_camera():
        print("❌ Không thể mở camera. Kết thúc test.")
        exit(1)

    print(f"✓ Kích thước khung hình: {camera.get_frame_size()}")
    print(f"✓ FPS: {camera.get_fps()}")
    print("\nNhấn 'q' để thoát, 's' để chụp ảnh test\n")

    frame_count = 0

    while True:
        # Đọc khung hình
        ret, frame = camera.read_frame()

        if not ret:
            break

        frame_count += 1

        # Vẽ thông tin lên khung hình
        frame = camera.draw_text(frame, f"Frame: {frame_count}", (10, 30))
        frame = camera.draw_text(frame, "Nhan 'q' de thoat", (10, 60), color=(0, 255, 255))

        # Vẽ box test (giả lập phát hiện khuôn mặt)
        if frame_count % 30 == 0:  # Mỗi 30 frame vẽ box test
            h, w = frame.shape[:2]
            test_bbox = (w//4, h//4, w//2, h//2)
            frame = camera.draw_emotion_box(frame, "Happy", 0.95, test_bbox)

        # Hiển thị
        camera.show_frame(frame)

        # Kiểm tra phím bấm
        key = camera.wait_key(1)

        if key == ord('q'):
            print("\n✓ Người dùng nhấn 'q'. Thoát chương trình.")
            break
        elif key == ord('s'):
            cv2.imwrite("data/test_screenshot.jpg", frame)
            print(f"✓ Đã lưu ảnh test: data/test_screenshot.jpg")

    # Giải phóng tài nguyên
    camera.release_camera()
    camera.destroy_all_windows()

    print(f"\n✓ Đã xử lý {frame_count} khung hình")
    print("=== TEST HOÀN TẤT ===")

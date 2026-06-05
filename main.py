"""
==================================================
FILE: main.py
Chức năng: File chính - Tích hợp toàn bộ hệ thống
Mô tả: Nhận dạng cảm xúc người học online real-time
Đại học Bách khoa Hà Nội
==================================================
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.capture import CameraCapture
from src.emotion_engine import EmotionEngine
from src.database import EmotionDatabase
from src.analytics import EmotionAnalytics
from datetime import datetime
import time


class EmotionDetectionSystem:
    """
    Hệ thống nhận dạng cảm xúc người học online

    Chức năng chính:
    - Capture video từ webcam
    - Nhận dạng cảm xúc real-time
    - Lưu dữ liệu vào SQLite database
    - Tạo biểu đồ thống kê
    """

    def __init__(self, model_name: str = 'enet_b0_8_best_afew',
                 camera_id: int = 0,
                 db_path: str = "data/emotions.db"):
        """
        Khởi tạo hệ thống

        Args:
            model_name: Tên mô hình HSEmotion
            camera_id: ID camera (0 = webcam mặc định)
            db_path: Đường dẫn database
        """
        print("="*60)
        print("HỆ THỐNG NHẬN DẠNG CẢM XÚC NGƯỜI HỌC ONLINE")
        print("Đại học Bách khoa Hà Nội")
        print("="*60)

        # Tạo session ID (timestamp)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n📝 Session ID: {self.session_id}")

        # Khởi tạo các modules
        print("\n⏳ Đang khởi tạo hệ thống...\n")

        print("1️⃣ Khởi tạo Camera...")
        self.camera = CameraCapture(camera_id=camera_id)

        print("\n2️⃣ Khởi tạo Emotion Engine...")
        self.engine = EmotionEngine(model_name=model_name)

        print("\n3️⃣ Khởi tạo Database...")
        self.database = EmotionDatabase(db_path=db_path)

        print("\n4️⃣ Khởi tạo Analytics...")
        self.analytics = EmotionAnalytics(output_dir="reports")

        print("\n✅ Hệ thống đã sẵn sàng!\n")

        # Biến theo dõi
        self.frame_count = 0
        self.emotion_buffer = []  # Lưu tạm dữ liệu trong RAM
        self.db_saved_count = 0   # Số bản ghi đã được lưu vào DB trong vòng lặp
        self.start_time = None
        self.is_running = False

        # Cache kết quả nhận diện để tái sử dụng trên frame bị skip
        self.last_results = []  # Kết quả nhận diện của frame trước
        self.last_frame_drawn = None  # Frame đã vẽ overlay của lần xử lý trước

    def start(self):
        """
        Bắt đầu chương trình chính
        """
        # Mở camera
        if not self.camera.open_camera():
            print("❌ Không thể mở camera. Kết thúc chương trình.")
            return

        print("\n" + "="*60)
        print("BẮT ĐẦU NHẬN DẠNG CẢM XÚC")
        print("="*60)
        print("\n📹 Webcam đã mở")
        print("\n⌨️  PHÍM ĐIỀU KHIỂN:")
        print("  • Nhấn 'q' hoặc 'ESC' để thoát")
        print("  • Nhấn 's' để chụp ảnh màn hình")
        print("  • Nhấn 'r' để reset dữ liệu session hiện tại")
        print("\n⏱️  Đang xử lý...\n")

        self.start_time = time.time()
        self.is_running = True

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

    def _draw_overlays(self, frame, results):
        """
        Vẽ toàn bộ overlay (emotion box + info text) lên frame.
        Tách riêng để tái sử dụng trên cả frame xử lý lẫn frame bị skip.

        Args:
            frame: Khung hình gốc
            results: Danh sách kết quả nhận diện từ engine

        Returns:
            Khung hình đã vẽ đầy đủ overlay
        """
        # Vẽ bounding box và emotion label cho từng khuôn mặt
        for result in results:
            frame = self.camera.draw_emotion_box(
                frame,
                result['emotion'],
                result['confidence'],
                result['bbox']
            )

        # Tính FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        # Vẽ thông tin hệ thống góc trên trái
        info_text = [
            f"Session: {self.session_id}",
            f"Frame: {self.frame_count}",
            f"FPS: {fps:.1f}",
            f"Faces: {len(results)}",
            f"Samples: {len(self.emotion_buffer)}"
        ]
        y_offset = 30
        for i, text in enumerate(info_text):
            frame = self.camera.draw_text(
                frame, text,
                position=(10, y_offset + i * 25),
                font_scale=0.6,
                color=(0, 255, 255),
                thickness=2
            )

        # Vẽ hướng dẫn góc dưới
        frame = self.camera.draw_text(
            frame, "Nhan 'q' de thoat, 's' de chup anh",
            position=(10, frame.shape[0] - 20),
            font_scale=0.5,
            color=(255, 255, 255),
            thickness=1
        )

        return frame

    def _main_loop(self):
        """
        Vòng lặp chính xử lý video
        """
        skip_frames = 2  # Xử lý AI mỗi 2 frame để tăng tốc

        while self.is_running:
            # Đọc frame
            ret, frame = self.camera.read_frame()
            if not ret:
                print("\n❌ Không thể đọc frame. Kết thúc.")
                break

            self.frame_count += 1

            # ── Frame bị SKIP: vẽ lại kết quả nhận diện của frame TRƯỚC ──
            # Giúp hiển thị mượt mà, không giật, không bị nhấp nháy box
            if self.frame_count % skip_frames != 0:
                display_frame = self._draw_overlays(frame, self.last_results)
                self.camera.show_frame(display_frame, "Emotion Detection - HUST")
                key_result = self._handle_key_press(display_frame)
                if key_result == 'quit':
                    break
                continue

            # ── Frame được XỬ LÝ: chạy AI nhận diện cảm xúc ──
            results = self.engine.process_frame(frame)

            # Cập nhật cache kết quả mới nhất
            self.last_results = results

            # Lưu dữ liệu vào buffer và database
            for result in results:
                self.emotion_buffer.append({
                    'timestamp': datetime.now().isoformat(),
                    'emotion': result['emotion'],
                    'confidence': result['confidence']
                })

                # Lưu vào database theo batch (mỗi 5 detection)
                if len(self.emotion_buffer) % 5 == 0:
                    self.database.insert_emotion(
                        result['emotion'],
                        result['confidence'],
                        self.session_id
                    )
                    self.db_saved_count += 1

            # Vẽ overlay và hiển thị
            display_frame = self._draw_overlays(frame, results)
            self.camera.show_frame(display_frame, "Emotion Detection - HUST")

            # Xử lý phím bấm
            if self._handle_key_press(display_frame) == 'quit':
                break

        self.is_running = False

    def _handle_key_press(self, current_frame=None):
        """
        Xử lý phím bấm

        Args:
            current_frame: Frame đang hiển thị (dùng để lưu screenshot)

        Returns:
            'quit' nếu cần thoát, None nếu tiếp tục
        """
        key = self.camera.wait_key(1)

        # Phím 'q' hoặc ESC để thoát
        if key == ord('q') or key == 27:
            print("\n✓ Người dùng yêu cầu thoát.")
            return 'quit'

        # Phím 's' để screenshot — lưu đúng frame đang hiển thị (có overlay)
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/screenshot_{timestamp}.jpg"
            import cv2
            save_frame = current_frame if current_frame is not None else None
            if save_frame is not None:
                cv2.imwrite(filename, save_frame)
                print(f"✓ Đã lưu ảnh: {filename}")
            else:
                print("⚠️ Không có frame để lưu")

        # Phím 'r' để reset
        elif key == ord('r'):
            print("\n⚠️ Reset dữ liệu session...")
            self.emotion_buffer.clear()
            self.last_results = []  # Xóa cache kết quả
            print("✓ Đã xóa buffer")

        return None

    def _cleanup(self):
        """
        Dọn dẹp và tạo báo cáo khi kết thúc
        """
        print("\n\n" + "="*60)
        print("KẾT THÚC PHIÊN HỌC")
        print("="*60)

        # Đóng camera
        print("\n🔒 Đang đóng camera...")
        self.camera.release_camera()
        self.camera.destroy_all_windows()

        # Thống kê tổng quan
        elapsed_time = time.time() - self.start_time
        total_samples = len(self.emotion_buffer)

        print(f"\n📊 THỐNG KÊ:")
        print(f"  • Thời gian chạy: {elapsed_time:.1f} giây")
        print(f"  • Tổng số frame: {self.frame_count}")
        print(f"  • Tổng số mẫu cảm xúc: {total_samples}")

        if total_samples == 0:
            print("\n⚠️ Không có dữ liệu để tạo báo cáo.")
            return

        # Chỉ lưu phần dữ liệu CHƯA được lưu trong vòng lặp
        # (tránh ghi trùng toàn bộ buffer)
        already_saved = self.db_saved_count  # mỗi lần ghi = 1 bản ghi (ghi từng cái)
        # Tính số bận ghi thực tế đã vào DB: mỗi lần emotion_buffer % 5 == 0 thì gọi insert 1 lần
        # = tổng số lần buffer chia hết cho 5 = len(buffer) // 5
        db_written_in_loop = (len(self.emotion_buffer) // 5)
        unsaved_items = self.emotion_buffer[db_written_in_loop * 5:]

        print("\n💾 Đang lưu dữ liệu còn lại vào database...")
        saved_now = 0
        for item in unsaved_items:
            self.database.insert_emotion(
                item['emotion'],
                item['confidence'],
                self.session_id
            )
            saved_now += 1
        print(f"✓ Đã lưu thêm {saved_now} bản ghi (tổng: {db_written_in_loop + saved_now}/{total_samples})")

        # Lấy thống kê từ database
        print("\n📈 Đang tạo báo cáo thống kê...")
        emotion_counts = self.database.get_emotion_counts(self.session_id)
        avg_confidence = self.database.get_average_confidence(self.session_id)

        # In kết quả ra console
        total_from_db = sum(emotion_counts.values())  # Dùng tổng từ DB để tính %
        print("\n🎭 PHÂN TÍCH CẢM XÚC:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_from_db) * 100 if total_from_db > 0 else 0
            conf = avg_confidence.get(emotion, 0.0)
            print(f"  • {emotion:12s}: {count:4d} lần ({percentage:5.1f}%) - Độ tin cậy: {conf:.2f}")

        # Tạo biểu đồ
        print("\n📊 Đang tạo biểu đồ...")
        try:
            # Biểu đồ số lượng
            self.analytics.plot_emotion_counts(emotion_counts, self.session_id)

            # Biểu đồ tròn
            self.analytics.plot_emotion_pie(emotion_counts, self.session_id)

            # Biểu đồ timeline (nếu có đủ dữ liệu)
            if len(self.emotion_buffer) > 10:
                self.analytics.plot_emotion_timeline(self.emotion_buffer, self.session_id)

            # Báo cáo tổng hợp
            self.analytics.generate_summary_report(emotion_counts, avg_confidence, self.session_id)

            print("✓ Đã tạo tất cả biểu đồ thành công!")
            print(f"✓ Kiểm tra thư mục 'reports/' để xem kết quả")

        except Exception as e:
            print(f"⚠️ Lỗi khi tạo biểu đồ: {e}")

        # Cảm xúc chủ đạo
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        print(f"\n🏆 Cảm xúc chủ đạo: {dominant_emotion}")

        print("\n" + "="*60)
        print("CẢM ƠN BẠN ĐÃ SỬ DỤNG HỆ THỐNG!")
        print("="*60 + "\n")


def print_help():
    """
    In hướng dẫn sử dụng
    """
    help_text = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  HỆ THỐNG NHẬN DẠNG CẢM XÚC NGƯỜI HỌC ONLINE                ║
    ║  Đại học Bách khoa Hà Nội                                    ║
    ╚══════════════════════════════════════════════════════════════╝

    CÁCH CHẠY CHƯƠNG TRÌNH:
    ----------------------
    python main.py [options]

    OPTIONS:
    --------
    --model <name>     : Chọn mô hình HSEmotion
                         • enet_b0_8_best_afew (mặc định - nhanh)
                         • enet_b0_8_best_vgaf (chính xác hơn)
                         • enet_b2_8 (chính xác nhất - chậm)

    --camera <id>      : Chọn camera ID (mặc định: 0)

    --db <path>        : Đường dẫn database (mặc định: data/emotions.db)

    --help             : Hiển thị hướng dẫn này

    VÍ DỤ:
    ------
    python main.py
    python main.py --model enet_b2_8
    python main.py --camera 1
    python main.py --model enet_b0_8_best_vgaf --camera 0

    PHÍM TẮT KHI CHẠY:
    ------------------
    • 'q' hoặc ESC : Thoát chương trình
    • 's'          : Chụp ảnh màn hình
    • 'r'          : Reset dữ liệu session

    YÊU CẦU HỆ THỐNG:
    -----------------
    • Python 3.8+
    • Webcam
    • Các thư viện: opencv-python, hsemotion, torch, matplotlib, seaborn

    CÀI ĐẶT THƯ VIỆN:
    -----------------
    py -m pip install -r requirements.txt

    """
    print(help_text)


def main():
    """
    Hàm main - Entry point của chương trình
    """
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Hệ thống nhận dạng cảm xúc người học online',
        add_help=False
    )
    parser.add_argument('--model', type=str, default='enet_b0_8_best_afew',
                       help='Tên mô hình HSEmotion')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (0 = mặc định)')
    parser.add_argument('--db', type=str, default='data/emotions.db',
                       help='Đường dẫn database')
    parser.add_argument('--help', action='store_true',
                       help='Hiển thị hướng dẫn')

    args = parser.parse_args()

    # Hiển thị help nếu được yêu cầu
    if args.help:
        print_help()
        return

    # Tạo hệ thống
    try:
        system = EmotionDetectionSystem(
            model_name=args.model,
            camera_id=args.camera,
            db_path=args.db
        )

        # Chạy hệ thống
        system.start()

    except KeyboardInterrupt:
        print("\n\n⚠️ Chương trình bị ngắt bởi người dùng")
    except Exception as e:
        print(f"\n\n❌ Lỗi nghiêm trọng: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
==================================================
SCRIPT: collect_data.py
Chức năng: Thu thập ảnh khuôn mặt từ webcam để tạo dataset
Mô tả: Bấm phím để chụp ảnh vào đúng thư mục class
Đại học Bách khoa Hà Nội
==================================================

CÁCH DÙNG:
    python collect_data.py
    python collect_data.py --class Happy --count 30

PHÍM ĐIỀU KHIỂN khi đang chạy:
    SPACE    : Chụp ảnh (tự động lưu vào thư mục class đang chọn)
    1-7      : Chuyển nhanh sang class tương ứng
    n / p    : Chuyển sang class Tiếp theo / Trước đó
    q / ESC  : Thoát
"""

import cv2
import os
import sys
import argparse
from datetime import datetime

# Danh sách 7 class (không có Contempt vì rất ít gặp)
CLASSES = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
DATASET_DIR = 'dataset'


def ensure_dirs():
    """Tạo thư mục cho từng class nếu chưa có."""
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)


def count_images():
    """Đếm số ảnh hiện có trong mỗi class."""
    counts = {}
    for cls in CLASSES:
        path = os.path.join(DATASET_DIR, cls)
        if os.path.isdir(path):
            counts[cls] = len([f for f in os.listdir(path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            counts[cls] = 0
    return counts


def draw_hud(frame, current_class, class_idx, counts, session_count, target):
    """Vẽ HUD (heads-up display) lên frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Nền cho panel trái
    cv2.rectangle(overlay, (0, 0), (260, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Tiêu đề
    cv2.putText(frame, "DATA COLLECTOR", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.line(frame, (8, 35), (252, 35), (60, 60, 60), 1)

    # Danh sách class
    for i, cls in enumerate(CLASSES):
        y = 58 + i * 32
        cnt = counts.get(cls, 0)
        is_current = (i == class_idx)

        if is_current:
            cv2.rectangle(frame, (5, y - 18), (255, y + 10), (0, 120, 80), -1)
            color = (255, 255, 255)
            weight = cv2.FONT_HERSHEY_SIMPLEX
        else:
            color = (160, 160, 160)
            weight = cv2.FONT_HERSHEY_SIMPLEX

        label = f"[{i+1}] {cls:10s} {cnt:3d} anh"
        cv2.putText(frame, label, (10, y), weight, 0.5, color, 1)

    # Thông tin dưới
    sep_y = h - 110
    cv2.line(frame, (8, sep_y), (252, sep_y), (60, 60, 60), 1)
    cv2.putText(frame, f"Phien nay: {session_count} anh", (8, sep_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Muc tieu:  {target} anh/class", (8, sep_y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Hướng dẫn
    cv2.putText(frame, "SPACE: Chup anh", (8, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1)
    cv2.putText(frame, "n/p: Class tiep/truoc", (8, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 200, 255), 1)
    cv2.putText(frame, "q/ESC: Thoat", (8, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 100, 100), 1)

    # Nhãn class hiện tại ở góc phải (to, dễ nhìn)
    txt = current_class.upper()
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
    tx = w - tw - 15
    ty = 45
    cv2.putText(frame, txt, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 150), 2)

    return frame


def flash_effect(frame, color=(0, 255, 0)):
    """Hiệu ứng flash khi chụp ảnh."""
    overlay = np.full_like(frame, color[::-1] + (0,))  # BGR
    # Simple: just return a slightly brightened frame
    return cv2.addWeighted(frame, 0.6, np.full_like(frame, 255), 0.4, 0)


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description='Thu thập ảnh dataset cảm xúc')
    parser.add_argument('--class', dest='start_class', default=None,
                        help=f'Class bắt đầu ({", ".join(CLASSES)})')
    parser.add_argument('--count', type=int, default=30,
                        help='Mục tiêu số ảnh mỗi class (mặc định: 30)')
    args = parser.parse_args()

    ensure_dirs()

    # Xác định class bắt đầu
    class_idx = 0
    if args.start_class:
        if args.start_class in CLASSES:
            class_idx = CLASSES.index(args.start_class)
        else:
            print(f"⚠️  Class '{args.start_class}' không hợp lệ. Bắt đầu từ {CLASSES[0]}.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    session_count = 0
    flash_frames = 0
    print(f"\n✓ Webcam đã mở. Bắt đầu thu thập dataset...")
    print(f"  Thư mục: {os.path.abspath(DATASET_DIR)}")
    print(f"  Mục tiêu: {args.count} ảnh/class\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counts = count_images()
        current_class = CLASSES[class_idx]

        # Flash effect sau khi chụp
        display = frame.copy()
        if flash_frames > 0:
            alpha = flash_frames / 5
            white = np.full_like(display, 255)
            display = cv2.addWeighted(display, 1 - alpha, white, alpha, 0)
            flash_frames -= 1

        display = draw_hud(display, current_class, class_idx,
                           counts, session_count, args.count)

        cv2.imshow("Dataset Collector - HUST", display)
        key = cv2.waitKey(1) & 0xFF

        # ── Xử lý phím ──
        if key in (ord('q'), 27):  # q hoặc ESC
            break

        elif key == ord(' '):  # SPACE = chụp ảnh
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"{current_class}_{ts}.jpg"
            save_path = os.path.join(DATASET_DIR, current_class, filename)
            cv2.imwrite(save_path, frame)
            session_count += 1
            flash_frames = 5
            cnt = counts.get(current_class, 0) + 1
            print(f"  📸 [{current_class}] {cnt}/{args.count}: {filename}")

            # Tự động chuyển class khi đủ target
            if cnt >= args.count:
                print(f"  ✅ Đã đủ {args.count} ảnh cho [{current_class}]!")
                next_idx = class_idx + 1
                if next_idx < len(CLASSES):
                    class_idx = next_idx
                    print(f"  ➡️  Chuyển sang: {CLASSES[class_idx]}\n")

        elif key == ord('n'):  # next class
            class_idx = (class_idx + 1) % len(CLASSES)
            print(f"  ➡️  Class: {CLASSES[class_idx]}")

        elif key == ord('p'):  # previous class
            class_idx = (class_idx - 1) % len(CLASSES)
            print(f"  ⬅️  Class: {CLASSES[class_idx]}")

        elif ord('1') <= key <= ord('7'):  # phím số 1-7
            idx = key - ord('1')
            if idx < len(CLASSES):
                class_idx = idx
                print(f"  🔢 Class: {CLASSES[class_idx]}")

    cap.release()
    cv2.destroyAllWindows()

    # Tổng kết
    final_counts = count_images()
    total = sum(final_counts.values())
    print(f"\n{'='*40}")
    print(f"  KẾT QUẢ THU THẬP DATASET")
    print(f"{'='*40}")
    for cls in CLASSES:
        cnt = final_counts.get(cls, 0)
        bar = '█' * min(cnt, 30) + '░' * max(0, 30 - cnt)
        status = '✓' if cnt >= args.count else '!'
        print(f"  {status} {cls:10s}: {cnt:3d} ảnh  |{bar}|")
    print(f"{'='*40}")
    print(f"  Tổng cộng: {total} ảnh | Phiên này: {session_count} ảnh")
    print(f"{'='*40}\n")
    print(f"  Bước tiếp theo:")
    print(f"  python main.py --mode finetune --dataset {DATASET_DIR}/")


if __name__ == '__main__':
    main()

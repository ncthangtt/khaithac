"""
==================================================
FILE: main.py
Chức năng: Entry point — 4 chế độ chạy
Mô tả: Nhận dạng cảm xúc người học online
Đại học Bách khoa Hà Nội
==================================================

CÁCH DÙNG:
  python main.py --mode webcam
  python main.py --mode image   --input face.jpg
  python main.py --mode finetune --dataset dataset/
  python main.py --mode evaluate --dataset dataset/

  Thêm --weights models/finetuned.pt để dùng model fine-tuned (webcam & image)
  Thêm --help để xem đầy đủ tùy chọn
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import argparse
import time
from datetime import datetime


# ══════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════

def print_header(mode: str):
    print("=" * 60)
    print("  HỆ THỐNG NHẬN DẠNG CẢM XÚC NGƯỜI HỌC ONLINE")
    print("  Đại học Bách khoa Hà Nội")
    print("=" * 60)
    print(f"  Chế độ : {mode.upper()}")
    print(f"  Thời điểm: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════
#  MODE 1: WEBCAM — nhận diện real-time
# ══════════════════════════════════════════════════

def run_webcam(args):
    """Nhận diện cảm xúc real-time từ webcam."""
    from src.capture import CameraCapture
    from src.emotion_engine import EmotionEngine

    print_header("WEBCAM")

    camera = CameraCapture(camera_id=args.camera)
    engine = EmotionEngine(model_name=args.model, weights_path=args.weights)

    if not camera.open_camera():
        print("❌ Không thể mở camera. Kết thúc.")
        return

    print("\n⌨️  Phím điều khiển:")
    print("  • q / ESC  : Thoát")
    print("  • s        : Chụp ảnh màn hình")
    print("  • r        : Reset temporal smoothing buffer")
    print("\n⏳ Đang xử lý...\n")

    frame_count = 0
    skip_frames = 2
    last_results = []
    start_time = time.time()

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                break

            frame_count += 1

            # Frame bị skip: vẽ lại kết quả cũ để tránh giật
            if frame_count % skip_frames != 0:
                display = _draw_webcam_overlay(frame, last_results, frame_count, start_time)
                camera.show_frame(display, "Emotion Detection — HUST")
                if _handle_key(camera, display, engine) == 'quit':
                    break
                continue

            # Frame xử lý AI
            results = engine.process_frame(frame)
            last_results = results

            display = _draw_webcam_overlay(frame, results, frame_count, start_time)
            camera.show_frame(display, "Emotion Detection — HUST")

            if _handle_key(camera, display, engine) == 'quit':
                break

    except KeyboardInterrupt:
        print("\n⚠️  Ngắt bởi người dùng (Ctrl+C)")
    finally:
        camera.release_camera()
        camera.destroy_all_windows()
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n✓ Kết thúc | {frame_count} frames | {elapsed:.1f}s | {fps:.1f} FPS")


def _draw_webcam_overlay(frame, results, frame_count, start_time):
    """Vẽ emotion box + info text lên frame."""
    import cv2
    import time

    display = frame.copy()

    COLOR_MAP = {
        'Happy': (0,255,0), 'Sad': (255,80,80), 'Angry': (0,0,255),
        'Surprise': (0,165,255), 'Neutral': (180,180,180),
        'Fear': (180,0,180), 'Disgust': (0,160,160),
    }

    for result in results:
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        confidence = result['confidence']
        color = COLOR_MAP.get(emotion, (255,255,255))

        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion}: {confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display, (x, y-lh-10), (x+lw+6, y), color, -1)
        cv2.putText(display, label, (x+3, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

    # Info overlay góc trên trái
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    info = [
        f"Frame: {frame_count}",
        f"FPS:   {fps:.1f}",
        f"Faces: {len(results)}",
    ]
    for i, txt in enumerate(info):
        cv2.putText(display, txt, (10, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Hướng dẫn góc dưới
    cv2.putText(display, "q: thoat | s: chup anh | r: reset",
                (10, display.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    return display


def _handle_key(camera, current_frame, engine):
    """Xử lý phím bấm. Trả về 'quit' nếu cần thoát."""
    import cv2
    key = camera.wait_key(1)
    if key in (ord('q'), 27):
        print("\n✓ Người dùng yêu cầu thoát.")
        return 'quit'
    elif key == ord('s'):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"data/screenshot_{ts}.jpg"
        os.makedirs('data', exist_ok=True)
        cv2.imwrite(path, current_frame)
        print(f"✓ Đã lưu ảnh: {path}")
    elif key == ord('r'):
        engine.face_buffers.clear()
        print("✓ Đã reset temporal smoothing buffer.")
    return None


# ══════════════════════════════════════════════════
#  MODE 2: IMAGE — nhận diện từ ảnh tĩnh
# ══════════════════════════════════════════════════

def run_image(args):
    """Nhận diện cảm xúc từ file ảnh."""
    from src.image_predictor import ImagePredictor

    print_header("IMAGE")

    if not args.input:
        print("❌ Cần cung cấp đường dẫn ảnh: --input <path>")
        return

    predictor = ImagePredictor(
        model_name=args.model,
        weights_path=args.weights
    )
    predictor.predict(
        image_path=args.input,
        output_path=args.output,
        show=True
    )


# ══════════════════════════════════════════════════
#  MODE 3: FINETUNE — fine-tune trên dataset
# ══════════════════════════════════════════════════

def run_finetune(args):
    """Fine-tune model trên dataset tự thu thập."""
    from src.fine_tuner import EmotionFineTuner

    print_header("FINE-TUNE")

    if not args.dataset:
        print("❌ Cần cung cấp đường dẫn dataset: --dataset <path>")
        return

    tuner = EmotionFineTuner(model_name=args.model)
    tuner.load_dataset(
        dataset_dir=args.dataset,
        batch_size=args.batch_size
    )
    history = tuner.train(
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_weights
    )

    print(f"\n💡 Bước tiếp theo:")
    print(f"   Đánh giá: python main.py --mode evaluate --dataset {args.dataset}")
    print(f"   Demo:     python main.py --mode webcam --weights {args.save_weights}")


# ══════════════════════════════════════════════════
#  MODE 4: EVALUATE — đánh giá & so sánh model
# ══════════════════════════════════════════════════

def run_evaluate(args):
    """Đánh giá và so sánh model gốc vs fine-tuned trên test set."""
    from src.fine_tuner import EmotionFineTuner
    from src.evaluator import ModelEvaluator
    from hsemotion.facial_emotions import HSEmotionRecognizer
    import torch

    print_header("EVALUATE")

    if not args.dataset:
        print("❌ Cần cung cấp đường dẫn dataset: --dataset <path>")
        return

    weights_path = args.weights or args.save_weights
    if not os.path.isfile(weights_path):
        print(f"❌ Không tìm thấy file fine-tuned weights: {weights_path}")
        print(f"   Chạy fine-tune trước: python main.py --mode finetune --dataset {args.dataset}")
        return

    # Load dataset (chỉ dùng test set)
    print("⏳ Đang load dataset...")
    tuner = EmotionFineTuner(model_name=args.model)
    tuner.load_dataset(dataset_dir=args.dataset, batch_size=args.batch_size)
    test_loader, class_names = tuner.test_data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model gốc
    print("\n⏳ Đang load model gốc...")
    original_recognizer = HSEmotionRecognizer(model_name=args.model)
    original_model = original_recognizer.model.to(device)

    # Model fine-tuned
    print("⏳ Đang load model fine-tuned...")
    checkpoint = torch.load(weights_path, map_location=device)
    finetuned_model = HSEmotionRecognizer(model_name=args.model).model.to(device)
    finetuned_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Đánh giá
    evaluator = ModelEvaluator(class_names=class_names, output_dir='reports')

    # Đọc history nếu tồn tại (được lưu từ fine-tune)
    history_path = weights_path.replace('.pt', '_history.npy')
    history = None
    if os.path.isfile(history_path):
        import numpy as np
        history = np.load(history_path, allow_pickle=True).item()

    evaluator.evaluate_and_compare(
        original_model=original_model,
        finetuned_model=finetuned_model,
        test_loader=test_loader,
        history=history
    )

    print(f"\n✓ Kết quả đã lưu vào thư mục: reports/")


# ══════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Hệ thống nhận dạng cảm xúc người học online — HUST',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  python main.py --mode webcam
  python main.py --mode image --input face.jpg
  python main.py --mode finetune --dataset dataset/ --epochs 15
  python main.py --mode evaluate --dataset dataset/
  python main.py --mode webcam --weights models/finetuned.pt
        """
    )

    parser.add_argument('--mode', type=str, default='webcam',
                        choices=['webcam', 'image', 'finetune', 'evaluate'],
                        help='Chế độ chạy (mặc định: webcam)')

    # Model
    parser.add_argument('--model', type=str, default='enet_b0_8_best_afew',
                        choices=['enet_b0_8_best_afew', 'enet_b0_8_best_vgaf', 'enet_b2_8'],
                        help='Tên model HSEmotion base')
    parser.add_argument('--weights', type=str, default=None,
                        help='Đường dẫn fine-tuned weights .pt (webcam & image)')

    # Webcam
    parser.add_argument('--camera', type=int, default=0,
                        help='ID camera (mặc định: 0)')

    # Image mode
    parser.add_argument('--input', type=str, default=None,
                        help='Đường dẫn ảnh đầu vào (mode image)')
    parser.add_argument('--output', type=str, default=None,
                        help='Lưu ảnh kết quả vào file này (mode image)')

    # Fine-tune / Evaluate
    parser.add_argument('--dataset', type=str, default=None,
                        help='Thư mục dataset (mode finetune & evaluate)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Số epoch fine-tune (mặc định: 15)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (mặc định: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=16, dest='batch_size',
                        help='Batch size (mặc định: 16)')
    parser.add_argument('--save-weights', type=str, default='models/finetuned.pt',
                        dest='save_weights',
                        help='Đường dẫn lưu model fine-tuned (mặc định: models/finetuned.pt)')

    return parser


# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        'webcam':   run_webcam,
        'image':    run_image,
        'finetune': run_finetune,
        'evaluate': run_evaluate,
    }

    try:
        dispatch[args.mode](args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Chương trình bị ngắt bởi người dùng.")
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

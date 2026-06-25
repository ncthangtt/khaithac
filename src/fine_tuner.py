"""
==================================================
MODULE: fine_tuner.py
Chức năng: Fine-tune mô hình HSEmotion trên dataset nhỏ
Mô tả: Đóng băng backbone, fine-tune lớp cuối + classifier
Đại học Bách khoa Hà Nội
==================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from hsemotion.facial_emotions import HSEmotionRecognizer
import numpy as np

# Thứ tự class theo model HSEmotion (8 lớp)
MODEL_CLASSES = [
    'Anger', 'Contempt', 'Disgust', 'Fear',
    'Happiness', 'Neutral', 'Sadness', 'Surprise'
]


class EmotionFineTuner:
    """
    Fine-tune mô hình EfficientNet từ HSEmotion trên dataset tự thu thập.

    Chiến lược:
    - Đóng băng (freeze) toàn bộ backbone — giữ đặc trưng học từ 450k ảnh
    - Mở đóng băng (unfreeze) 2 blocks cuối + classifier → học thêm từ data nhóm
    - Split tự động 70/15/15 (train/val/test)
    """

    def __init__(self, model_name: str = 'enet_b0_8_best_afew'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚙️  Thiết bị: {self.device.upper()}")

        # Load model gốc từ HSEmotion
        print(f"⏳ Đang load model gốc: {model_name} ...")
        recognizer = HSEmotionRecognizer(model_name=model_name)
        self.model = recognizer.model.to(self.device)
        self.transform = recognizer.transform

        # Bước 1: Đóng băng TOÀN BỘ tham số
        for param in self.model.parameters():
            param.requires_grad = False

        # Bước 2: Mở đóng băng 2 blocks cuối + classifier
        for block in list(self.model.blocks)[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Tham số có thể train: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = []
        self.label_map = {}

    def load_dataset(self, dataset_dir: str, val_split: float = 0.15,
                     test_split: float = 0.15, batch_size: int = 16):
        """
        Đọc dataset từ cấu trúc thư mục: dataset_dir/ClassName/image.jpg

        Tên thư mục phải khớp với MODEL_CLASSES:
        Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise

        Args:
            dataset_dir: Đường dẫn thư mục dataset
            val_split:   Tỉ lệ validation (mặc định 15%)
            test_split:  Tỉ lệ test (mặc định 15%)
            batch_size:  Số ảnh mỗi batch
        """
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_dir}")

        # Đọc dataset thô để lấy danh sách class
        raw = datasets.ImageFolder(dataset_dir)
        folder_classes = raw.classes  # ImageFolder sắp xếp theo alphabet

        # Xây dựng mapping: index ImageFolder → index model
        self.label_map = {}
        self.class_names = []
        skipped = []
        for i, cls in enumerate(folder_classes):
            if cls in MODEL_CLASSES:
                self.label_map[i] = MODEL_CLASSES.index(cls)
                self.class_names.append(cls)
            else:
                skipped.append(cls)

        if skipped:
            print(f"⚠️  Bỏ qua class không hợp lệ: {skipped}")
            print(f"   Class hợp lệ: {MODEL_CLASSES}")

        if not self.class_names:
            raise ValueError("Không có class hợp lệ nào trong dataset!")

        # Tạo dataset với transform + label remapping
        full_dataset = datasets.ImageFolder(
            dataset_dir,
            transform=self.transform,
            target_transform=lambda y: self.label_map.get(y, y)
        )

        n = len(full_dataset)
        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        n_train = n - n_val - n_test

        if n_train <= 0:
            raise ValueError(f"Dataset quá nhỏ ({n} ảnh). Cần ít nhất 10 ảnh/class.")

        train_ds, val_ds, test_ds = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f"\n✓ Dataset: {n} ảnh | {len(self.class_names)} classes: {self.class_names}")
        print(f"  Train: {n_train} ảnh | Val: {n_val} ảnh | Test: {n_test} ảnh")

        return self.train_loader, self.val_loader, self.test_loader

    def train(self, epochs: int = 15, lr: float = 1e-4,
              save_path: str = 'models/finetuned.pt') -> dict:
        """
        Fine-tune mô hình và lưu weights tốt nhất.

        Args:
            epochs:    Số epoch huấn luyện
            lr:        Learning rate (Adam)
            save_path: Đường dẫn lưu model

        Returns:
            history: dict chứa loss/accuracy từng epoch
        """
        if self.train_loader is None:
            raise RuntimeError("Chưa load dataset! Gọi load_dataset() trước.")

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        # Giảm lr 50% mỗi 5 epoch để hội tụ ổn định
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_val_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        print(f"\n{'='*55}")
        print(f"  BẮT ĐẦU FINE-TUNE — {epochs} EPOCH")
        print(f"{'='*55}")
        print(f"  LR={lr} | Device={self.device.upper()} | Batch={self.train_loader.batch_size}")
        print(f"{'='*55}\n")

        for epoch in range(epochs):
            # ── Train ──
            self.model.train()
            t_loss = t_correct = t_total = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                t_loss += loss.item()
                _, predicted = outputs.max(1)
                t_correct += predicted.eq(labels).sum().item()
                t_total += labels.size(0)

            # ── Validation ──
            self.model.eval()
            v_loss = v_correct = v_total = 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    v_loss += loss.item()
                    _, predicted = outputs.max(1)
                    v_correct += predicted.eq(labels).sum().item()
                    v_total += labels.size(0)

            train_acc = 100.0 * t_correct / t_total
            val_acc = 100.0 * v_correct / v_total
            avg_t_loss = t_loss / len(self.train_loader)
            avg_v_loss = v_loss / len(self.val_loader)

            history['train_loss'].append(avg_t_loss)
            history['val_loss'].append(avg_v_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            saved = ''
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'class_names': self.class_names,
                    'model_name': self.model_name,
                    'val_acc': val_acc,
                    'epoch': epoch + 1,
                }, save_path)
                saved = '  ✓ Saved!'

            print(f"Epoch [{epoch+1:2d}/{epochs}] "
                  f"Train Loss: {avg_t_loss:.4f} Acc: {train_acc:5.1f}% | "
                  f"Val Loss: {avg_v_loss:.4f} Acc: {val_acc:5.1f}%{saved}")

            scheduler.step()

        print(f"\n{'='*55}")
        print(f"  ✅ Fine-tune hoàn tất!")
        print(f"  Best val accuracy: {best_val_acc:.1f}%")
        print(f"  Model đã lưu tại: {save_path}")
        print(f"{'='*55}\n")

        return history

    @property
    def test_data(self):
        return self.test_loader, self.class_names

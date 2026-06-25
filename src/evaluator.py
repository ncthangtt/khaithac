"""
==================================================
MODULE: evaluator.py
Chức năng: Đánh giá và so sánh model gốc vs fine-tuned
Mô tả: Confusion matrix, classification report, training curve
Đại học Bách khoa Hà Nội
==================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from hsemotion.facial_emotions import HSEmotionRecognizer

# Thứ tự class theo model HSEmotion
MODEL_CLASSES = [
    'Anger', 'Contempt', 'Disgust', 'Fear',
    'Happiness', 'Neutral', 'Sadness', 'Surprise'
]


class ModelEvaluator:
    """
    Đánh giá hiệu năng mô hình trên test set.
    So sánh model gốc (pretrained) vs model fine-tuned.
    Xuất: confusion matrix, classification report, training curve.
    """

    def __init__(self, class_names: list, output_dir: str = 'reports'):
        """
        Args:
            class_names: Danh sách tên class có trong dataset
            output_dir:  Thư mục lưu ảnh biểu đồ
        """
        self.class_names = class_names
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # Inference helpers
    # ──────────────────────────────────────────────

    def _predict(self, model, dataloader) -> tuple:
        """Chạy model trên dataloader, trả về (y_true, y_pred)."""
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = model(images)
                # Lấy chỉ các class có trong dataset
                _, predicted = outputs.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        return np.array(all_labels), np.array(all_preds)

    # ──────────────────────────────────────────────
    # Main evaluation
    # ──────────────────────────────────────────────

    def evaluate_and_compare(self, original_model, finetuned_model,
                             test_loader, history: dict = None) -> tuple:
        """
        So sánh model gốc và fine-tuned trên test set.

        Args:
            original_model:  PyTorch model gốc (HSEmotionRecognizer.model)
            finetuned_model: PyTorch model đã fine-tune
            test_loader:     DataLoader của test set
            history:         dict loss/acc từng epoch (từ fine_tuner.train())

        Returns:
            (acc_original, acc_finetuned)
        """
        print("\n" + "=" * 52)
        print("     ĐÁNH GIÁ MÔ HÌNH EMOTION DETECTION")
        print("=" * 52)

        y_true, y_pred_orig = self._predict(original_model, test_loader)
        _, y_pred_ft = self._predict(finetuned_model, test_loader)

        acc_orig = accuracy_score(y_true, y_pred_orig) * 100
        acc_ft = accuracy_score(y_true, y_pred_ft) * 100
        delta = acc_ft - acc_orig

        print(f"\n📂 Test set: {len(y_true)} ảnh | {len(self.class_names)} classes")
        print(f"\n  Model gốc  (HSEmotion pretrained): {acc_orig:.1f}%")
        print(f"  Model fine-tuned:                  {acc_ft:.1f}%  ({delta:+.1f}%)")

        # Lấy tên class thực sự xuất hiện trong test set
        present_indices = sorted(set(y_true.tolist()))
        present_names = [MODEL_CLASSES[i] for i in present_indices if i < len(MODEL_CLASSES)]

        print(f"\n📋 Classification Report (Model Fine-tuned):")
        print(classification_report(
            y_true, y_pred_ft,
            labels=present_indices,
            target_names=present_names,
            zero_division=0
        ))

        # Lưu biểu đồ
        self._plot_confusion_matrix(y_true, y_pred_ft, present_indices, present_names)
        if history:
            self._plot_training_curve(history)

        return acc_orig, acc_ft

    # ──────────────────────────────────────────────
    # Plot helpers
    # ──────────────────────────────────────────────

    def _plot_confusion_matrix(self, y_true, y_pred,
                               present_indices, present_names):
        """Vẽ và lưu confusion matrix dạng heatmap."""
        cm = confusion_matrix(y_true, y_pred, labels=present_indices)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=present_names, yticklabels=present_names,
            linewidths=0.5
        )
        plt.title('Confusion Matrix — Fine-tuned Model',
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: {path}")

    def _plot_training_curve(self, history: dict):
        """Vẽ và lưu biểu đồ loss + accuracy theo epoch."""
        epochs = range(1, len(history['train_loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Fine-tuning Training Curves', fontsize=14, fontweight='bold')

        # Loss
        ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
        ax1.set_title('Loss theo Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-o', markersize=4, label='Val Acc')
        ax2.set_title('Accuracy theo Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'training_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: {path}")

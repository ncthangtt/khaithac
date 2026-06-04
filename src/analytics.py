"""
==================================================
MODULE: analytics.py
Chức năng: Thống kê và vẽ biểu đồ cảm xúc
Mô tả: Sử dụng Matplotlib và Seaborn để visualization
Đại học Bách khoa Hà Nội
==================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from datetime import datetime


class EmotionAnalytics:
    """
    Lớp phân tích và vẽ biểu đồ thống kê cảm xúc

    Chức năng chính:
    - Tạo biểu đồ tần suất cảm xúc
    - Tạo biểu đồ phân bố theo thời gian
    - Xuất báo cáo thống kê
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Khởi tạo Analytics

        Args:
            output_dir: Thư mục lưu biểu đồ (mặc định: reports/)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Thiết lập style cho biểu đồ đẹp hơn
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)

        # Màu sắc cho mỗi cảm xúc
        self.emotion_colors = {
            'Happy': '#2ecc71',      # Xanh lá
            'Sad': '#3498db',        # Xanh dương
            'Surprise': '#f39c12',   # Cam
            'Angry': '#e74c3c',      # Đỏ
            'Neutral': '#95a5a6',    # Xám
            'Fear': '#9b59b6',       # Tím
            'Disgust': '#16a085'     # Xanh lục
        }

    def plot_emotion_counts(self, emotion_counts: Dict[str, int],
                           session_id: str,
                           save_path: str = None) -> str:
        """
        Vẽ biểu đồ cột (bar chart) thống kê số lượng cảm xúc

        Args:
            emotion_counts: Dictionary {emotion_name: count}
            session_id: ID phiên học
            save_path: Đường dẫn lưu file (None = tự động)

        Returns:
            Đường dẫn file đã lưu
        """
        if not emotion_counts:
            print("⚠️ Không có dữ liệu để vẽ biểu đồ")
            return ""

        # Tạo figure
        plt.figure(figsize=(10, 6))

        # Dữ liệu
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = [self.emotion_colors.get(e, '#7f8c8d') for e in emotions]

        # Vẽ biểu đồ cột
        bars = plt.bar(emotions, counts, color=colors, alpha=0.8, edgecolor='black')

        # Thêm giá trị trên mỗi cột
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Tổng số
        total = sum(counts)

        # Tiêu đề và nhãn
        plt.title(f'Thống kê cảm xúc - Phiên học: {session_id}\n'
                 f'Tổng số mẫu: {total}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Loại cảm xúc', fontsize=12, fontweight='bold')
        plt.ylabel('Số lần xuất hiện', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Lưu file
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir,
                                    f"emotion_counts_{session_id}_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Đã lưu biểu đồ số lượng: {save_path}")
        return save_path

    def plot_emotion_pie(self, emotion_counts: Dict[str, int],
                        session_id: str,
                        save_path: str = None) -> str:
        """
        Vẽ biểu đồ tròn (pie chart) phân bố cảm xúc

        Args:
            emotion_counts: Dictionary {emotion_name: count}
            session_id: ID phiên học
            save_path: Đường dẫn lưu file

        Returns:
            Đường dẫn file đã lưu
        """
        if not emotion_counts:
            print("⚠️ Không có dữ liệu để vẽ biểu đồ")
            return ""

        # Tạo figure
        plt.figure(figsize=(10, 8))

        # Dữ liệu
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = [self.emotion_colors.get(e, '#7f8c8d') for e in emotions]

        # Tổng số
        total = sum(counts)

        # Tính phần trăm
        percentages = [(c/total)*100 for c in counts]

        # Vẽ pie chart
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=emotions,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )

        # Làm đẹp text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Tiêu đề
        plt.title(f'Phân bố cảm xúc - Phiên học: {session_id}\n'
                 f'Tổng số mẫu: {total}',
                 fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Lưu file
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir,
                                    f"emotion_pie_{session_id}_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Đã lưu biểu đồ tròn: {save_path}")
        return save_path

    def plot_emotion_timeline(self, emotion_data: List[Dict],
                             session_id: str,
                             save_path: str = None) -> str:
        """
        Vẽ biểu đồ timeline cảm xúc theo thời gian

        Args:
            emotion_data: List[{'timestamp': str, 'emotion': str, 'confidence': float}]
            session_id: ID phiên học
            save_path: Đường dẫn lưu file

        Returns:
            Đường dẫn file đã lưu
        """
        if not emotion_data:
            print("⚠️ Không có dữ liệu để vẽ biểu đồ")
            return ""

        # Chuyển sang DataFrame
        df = pd.DataFrame(emotion_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Tạo figure
        plt.figure(figsize=(14, 6))

        # Đếm số lượng mỗi cảm xúc theo thời gian (rolling window)
        window_size = 10  # Mỗi 10 mẫu

        for emotion in df['emotion'].unique():
            emotion_df = df[df['emotion'] == emotion].copy()
            emotion_df = emotion_df.sort_values('timestamp')

            # Đếm số lượng trong cửa sổ trượt
            emotion_df['count'] = 1
            emotion_df['rolling_count'] = emotion_df['count'].rolling(
                window=window_size, min_periods=1
            ).sum()

            plt.plot(emotion_df['timestamp'], emotion_df['rolling_count'],
                    label=emotion, color=self.emotion_colors.get(emotion, '#7f8c8d'),
                    linewidth=2, marker='o', markersize=3, alpha=0.8)

        # Tiêu đề và nhãn
        plt.title(f'Biến động cảm xúc theo thời gian - Phiên học: {session_id}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Thời gian', fontsize=12, fontweight='bold')
        plt.ylabel(f'Số lần xuất hiện (cửa sổ {window_size} mẫu)',
                  fontsize=12, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Lưu file
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir,
                                    f"emotion_timeline_{session_id}_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Đã lưu biểu đồ timeline: {save_path}")
        return save_path

    def generate_summary_report(self, emotion_counts: Dict[str, int],
                               avg_confidence: Dict[str, float],
                               session_id: str,
                               save_path: str = None) -> str:
        """
        Tạo báo cáo tổng hợp với nhiều biểu đồ

        Args:
            emotion_counts: Dictionary {emotion_name: count}
            avg_confidence: Dictionary {emotion_name: avg_confidence}
            session_id: ID phiên học
            save_path: Đường dẫn lưu file

        Returns:
            Đường dẫn file đã lưu
        """
        if not emotion_counts:
            print("⚠️ Không có dữ liệu để tạo báo cáo")
            return ""

        # Tạo figure với 3 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'BÁO CÁO TỔNG HỢP CẢM XÚC - Phiên học: {session_id}',
                    fontsize=16, fontweight='bold', y=0.995)

        # 1. Biểu đồ cột số lượng
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = [self.emotion_colors.get(e, '#7f8c8d') for e in emotions]

        bars = axes[0, 0].bar(emotions, counts, color=colors, alpha=0.8, edgecolor='black')
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(count)}', ha='center', va='bottom', fontweight='bold')

        axes[0, 0].set_title('Số lượng cảm xúc', fontweight='bold')
        axes[0, 0].set_xlabel('Cảm xúc')
        axes[0, 0].set_ylabel('Số lần')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Biểu đồ tròn phân bố
        total = sum(counts)
        axes[0, 1].pie(counts, labels=emotions, colors=colors,
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title(f'Phân bố (Tổng: {total})', fontweight='bold')

        # 3. Biểu đồ độ tin cậy
        if avg_confidence:
            conf_emotions = list(avg_confidence.keys())
            conf_values = list(avg_confidence.values())
            conf_colors = [self.emotion_colors.get(e, '#7f8c8d') for e in conf_emotions]

            bars = axes[1, 0].barh(conf_emotions, conf_values, color=conf_colors, alpha=0.8)
            for bar, val in zip(bars, conf_values):
                width = bar.get_width()
                axes[1, 0].text(width, bar.get_y() + bar.get_height()/2.,
                              f'{val:.2f}', ha='left', va='center', fontweight='bold')

            axes[1, 0].set_title('Độ tin cậy trung bình', fontweight='bold')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_xlim(0, 1.0)

        # 4. Bảng thống kê
        axes[1, 1].axis('off')
        table_data = []
        for emotion in emotions:
            count = emotion_counts[emotion]
            percentage = (count/total)*100
            conf = avg_confidence.get(emotion, 0.0)
            table_data.append([emotion, count, f"{percentage:.1f}%", f"{conf:.2f}"])

        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Cảm xúc', 'Số lần', 'Tỉ lệ', 'Độ tin cậy'],
            cellLoc='center',
            loc='center',
            colWidths=[0.3, 0.2, 0.2, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Bảng thống kê chi tiết', fontweight='bold', pad=20)

        plt.tight_layout()

        # Lưu file
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir,
                                    f"summary_report_{session_id}_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Đã lưu báo cáo tổng hợp: {save_path}")
        return save_path


# Test code (chỉ chạy khi file này được chạy trực tiếp)
if __name__ == "__main__":
    print("=== TEST ANALYTICS MODULE ===\n")

    # Tạo analytics object
    analytics = EmotionAnalytics(output_dir="reports")

    # Dữ liệu test
    test_emotion_counts = {
        'Happy': 45,
        'Neutral': 30,
        'Surprise': 15,
        'Sad': 8,
        'Angry': 2
    }

    test_avg_confidence = {
        'Happy': 0.92,
        'Neutral': 0.85,
        'Surprise': 0.88,
        'Sad': 0.79,
        'Angry': 0.81
    }

    test_session_id = "test_session_001"

    print("✓ Đang vẽ biểu đồ số lượng...")
    analytics.plot_emotion_counts(test_emotion_counts, test_session_id)

    print("✓ Đang vẽ biểu đồ tròn...")
    analytics.plot_emotion_pie(test_emotion_counts, test_session_id)

    print("✓ Đang tạo báo cáo tổng hợp...")
    analytics.generate_summary_report(test_emotion_counts, test_avg_confidence, test_session_id)

    print("\n=== TEST HOÀN TẤT ===")
    print(f"✓ Kiểm tra thư mục 'reports/' để xem các biểu đồ")

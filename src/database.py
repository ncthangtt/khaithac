"""
==================================================
MODULE: database.py
Chức năng: Quản lý cơ sở dữ liệu SQLite
Mô tả: Lưu trữ dữ liệu cảm xúc của người học
Đại học Bách khoa Hà Nội
==================================================
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Tuple


class EmotionDatabase:
    """
    Lớp quản lý cơ sở dữ liệu SQLite cho dự án nhận dạng cảm xúc

    Chức năng chính:
    - Tạo và quản lý database
    - Lưu trữ dữ liệu cảm xúc theo thời gian
    - Truy vấn thống kê
    """

    def __init__(self, db_path: str = "data/emotions.db"):
        """
        Khởi tạo database

        Args:
            db_path: Đường dẫn file database (mặc định: data/emotions.db)
        """
        self.db_path = db_path

        # Tạo thư mục data nếu chưa có
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Kết nối và tạo bảng
        self._create_tables()

    def _create_tables(self):
        """
        Tạo các bảng trong database nếu chưa tồn tại

        Bảng emotions:
        - id: Khóa chính (tự động tăng)
        - timestamp: Thời điểm ghi nhận (ISO 8601)
        - emotion: Loại cảm xúc (VD: Happy, Sad, Surprise, etc.)
        - confidence: Độ tin cậy (0.0 - 1.0)
        - session_id: ID phiên học (để phân biệt các lần chạy)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                session_id TEXT NOT NULL
            )
        """)

        # Tạo index để tăng tốc truy vấn
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id
            ON emotions(session_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON emotions(timestamp)
        """)

        conn.commit()
        conn.close()

    def insert_emotion(self, emotion: str, confidence: float, session_id: str):
        """
        Thêm một bản ghi cảm xúc vào database

        Args:
            emotion: Tên cảm xúc (VD: "Happy", "Sad")
            confidence: Độ tin cậy (0.0 - 1.0)
            session_id: ID phiên học
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO emotions (timestamp, emotion, confidence, session_id)
            VALUES (?, ?, ?, ?)
        """, (timestamp, emotion, confidence, session_id))

        conn.commit()
        conn.close()

    def get_emotions_by_session(self, session_id: str) -> List[Tuple]:
        """
        Lấy tất cả cảm xúc của một phiên học

        Args:
            session_id: ID phiên học

        Returns:
            List[(id, timestamp, emotion, confidence, session_id)]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM emotions
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))

        results = cursor.fetchall()
        conn.close()

        return results

    def get_emotion_counts(self, session_id: str) -> Dict[str, int]:
        """
        Đếm số lượng mỗi loại cảm xúc trong một phiên học

        Args:
            session_id: ID phiên học

        Returns:
            Dictionary {emotion_name: count}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM emotions
            WHERE session_id = ?
            GROUP BY emotion
            ORDER BY count DESC
        """, (session_id,))

        results = cursor.fetchall()
        conn.close()

        # Chuyển thành dictionary
        return {emotion: count for emotion, count in results}

    def get_average_confidence(self, session_id: str) -> Dict[str, float]:
        """
        Tính độ tin cậy trung bình của mỗi cảm xúc

        Args:
            session_id: ID phiên học

        Returns:
            Dictionary {emotion_name: avg_confidence}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT emotion, AVG(confidence) as avg_conf
            FROM emotions
            WHERE session_id = ?
            GROUP BY emotion
        """, (session_id,))

        results = cursor.fetchall()
        conn.close()

        return {emotion: round(avg_conf, 3) for emotion, avg_conf in results}

    def get_all_sessions(self) -> List[str]:
        """
        Lấy danh sách tất cả session_id trong database

        Returns:
            List[session_id]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT session_id
            FROM emotions
            ORDER BY session_id DESC
        """)

        results = cursor.fetchall()
        conn.close()

        return [row[0] for row in results]

    def delete_session(self, session_id: str):
        """
        Xóa tất cả dữ liệu của một phiên học

        Args:
            session_id: ID phiên học cần xóa
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM emotions
            WHERE session_id = ?
        """, (session_id,))

        conn.commit()
        conn.close()

    def clear_all_data(self):
        """
        Xóa toàn bộ dữ liệu trong database (cẩn thận!)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM emotions")

        conn.commit()
        conn.close()


# Test code (chỉ chạy khi file này được chạy trực tiếp)
if __name__ == "__main__":
    print("=== TEST DATABASE MODULE ===\n")

    # Tạo database test
    db = EmotionDatabase("data/test_emotions.db")

    # Thêm dữ liệu mẫu
    test_session = "test_session_001"
    db.insert_emotion("Happy", 0.95, test_session)
    db.insert_emotion("Happy", 0.92, test_session)
    db.insert_emotion("Sad", 0.88, test_session)
    db.insert_emotion("Surprise", 0.91, test_session)
    db.insert_emotion("Happy", 0.89, test_session)

    print("✓ Đã thêm 5 bản ghi mẫu")

    # Kiểm tra đếm số lượng
    counts = db.get_emotion_counts(test_session)
    print(f"\n✓ Số lượng cảm xúc: {counts}")

    # Kiểm tra độ tin cậy trung bình
    avg_conf = db.get_average_confidence(test_session)
    print(f"\n✓ Độ tin cậy trung bình: {avg_conf}")

    # Lấy tất cả sessions
    sessions = db.get_all_sessions()
    print(f"\n✓ Danh sách sessions: {sessions}")

    print("\n=== TEST HOÀN TẤT ===")

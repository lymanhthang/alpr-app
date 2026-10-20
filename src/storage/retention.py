"""
Module quản lý chính sách lưu trữ (Retention Policy).
Tự động dọn dẹp các bản ghi và file ảnh/clip cũ khi vượt quá số ngày lưu hoặc dung lượng ổ đĩa tối đa.
"""

from datetime import datetime
import os
from pathlib import Path
from typing import List
from src.storage.db import DatabaseManager


def get_dir_size_bytes(directory: Path) -> int:
    """Tính tổng dung lượng (bytes) của một thư mục."""
    total = 0
    if not directory.exists():
        return 0
    try:
        for entry in os.scandir(directory):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size_bytes(Path(entry.path))
    except Exception:
        pass
    return total


def delete_sighting_files(file_paths: List[str]):
    """Xóa các file ảnh/clip liên quan trên đĩa."""
    for p_str in file_paths:
        if not p_str:
            continue
        try:
            p = Path(p_str)
            if p.is_file():
                p.unlink(missing_ok=True)
        except Exception:
            pass


def enforce_retention_policy(
    db: DatabaseManager,
    data_dir: str = "data",
    retention_days: int = 30,
    max_disk_gb: float = 20.0,
):
    """
    Thực thi chính sách dọn dẹp dữ liệu cũ:
    1. Xóa các sighting có tuổi thọ vượt quá retention_days.
    2. Nếu dung lượng thư mục data/ vượt quá max_disk_gb, tiếp tục xóa các lượt cũ nhất.
    """
    now_ts = datetime.now().timestamp()
    cutoff_ts = now_ts - (retention_days * 86400.0)

    # 1. Dọn dẹp theo thời gian (retention_days)
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, best_crop_path, context_path, clip_path FROM sightings WHERE t_start < ?",
            (cutoff_ts,),
        )
        old_rows = cursor.fetchall()
        if old_rows:
            ids_to_del = [row["id"] for row in old_rows]
            files_to_del = []
            for r in old_rows:
                files_to_del.extend([r["best_crop_path"], r["context_path"], r["clip_path"]])

            delete_sighting_files(files_to_del)
            cursor.execute(
                f"DELETE FROM sightings WHERE id IN ({','.join(['?'] * len(ids_to_del))})",
                ids_to_del,
            )

    # 2. Dọn dẹp theo dung lượng ổ đĩa (max_disk_gb)
    data_path = Path(data_dir)
    current_size_gb = get_dir_size_bytes(data_path) / (1024.0 ** 3)
    target_size_gb = max_disk_gb * 0.9  # Hạ xuống dưới 90% ngưỡng tối đa

    if current_size_gb > max_disk_gb:
        while current_size_gb > target_size_gb:
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, best_crop_path, context_path, clip_path FROM sightings ORDER BY t_start ASC LIMIT 50"
                )
                batch = cursor.fetchall()
                if not batch:
                    break

                ids_to_del = [r["id"] for r in batch]
                files_to_del = []
                for r in batch:
                    files_to_del.extend([r["best_crop_path"], r["context_path"], r["clip_path"]])

                delete_sighting_files(files_to_del)
                cursor.execute(
                    f"DELETE FROM sightings WHERE id IN ({','.join(['?'] * len(ids_to_del))})",
                    ids_to_del,
                )

            current_size_gb = get_dir_size_bytes(data_path) / (1024.0 ** 3)

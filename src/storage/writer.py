"""
Module ghi cơ sở dữ liệu và lưu ảnh bất đồng bộ qua hàng đợi (SightingWriter).
Hoạt động trên một thread riêng biệt để tuyệt đối không làm suy giảm FPS của pipeline chính.
"""

from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
from typing import Optional
import cv2

from src.config import StorageConfig
from src.reliability.track_state import FinalizedSighting
from src.storage.db import DatabaseManager
from src.storage.retention import enforce_retention_policy


class SightingWriter:
    """Thread ghi dữ liệu lượt xe và ảnh xuống đĩa."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.enabled = config.enabled
        self.dropped_count: int = 0
        self.written_count: int = 0

        if not self.enabled:
            return

        self.db = DatabaseManager(config.db_path)
        self.crops_dir = Path(config.crops_dir)
        self.contexts_dir = Path(config.contexts_dir)
        self.clips_dir = Path(config.clips_dir)

        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.contexts_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self.queue: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="ALPR_Storage_Writer", daemon=True)
        self._thread.start()

        # Thực thi dọn dẹp ban đầu khi khởi động
        try:
            enforce_retention_policy(
                self.db,
                data_dir=str(self.crops_dir.parent),
                retention_days=config.retention_days,
                max_disk_gb=config.max_disk_gb,
            )
        except Exception as e:
            print(f"[Storage Warning] Lỗi thực thi retention policy ban đầu: {e}")

    def submit(self, sighting: FinalizedSighting, source_id: int):
        """Đưa sighting vào hàng đợi để thread ghi xử lý không đồng bộ."""
        if not self.enabled:
            return

        try:
            self.queue.put_nowait((sighting, source_id))
        except queue.Full:
            self.dropped_count += 1
            print(
                f"[Storage Warning] Hàng đợi ghi CSDL đã đầy ({self.queue.maxsize})! "
                f"Bỏ qua lưu sighting Track #{sighting.track_id} (Tổng rơi: {self.dropped_count})."
            )

    def _worker(self):
        """Vòng lặp chạy ngầm lấy dữ liệu từ hàng đợi, lưu ảnh và ghi vào SQLite."""
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                self.queue.task_done()
                break

            sighting, source_id = item
            try:
                self._process_single_sighting(sighting, source_id)
                self.written_count += 1
            except Exception as e:
                print(f"[Storage Error] Lỗi ghi sighting Track #{sighting.track_id}: {e}")
            finally:
                self.queue.task_done()

    def _process_single_sighting(self, sighting: FinalizedSighting, source_id: int):
        """Lưu file ảnh và tạo bản ghi trong CSDL."""
        now = datetime.fromtimestamp(sighting.t_start)
        date_folder = now.strftime("%Y%m%d")

        crop_sub = self.crops_dir / date_folder
        crop_sub.mkdir(parents=True, exist_ok=True)
        context_sub = self.contexts_dir / date_folder
        context_sub.mkdir(parents=True, exist_ok=True)

        clean_plate = sighting.plate if sighting.plate else "UNKNOWN"
        ts_str = now.strftime("%H%M%S_%f")[:10]
        base_name = f"{ts_str}_t{sighting.track_id}_{clean_plate}"

        # 1. Lưu ảnh crop rõ nhất
        crop_filename = f"{base_name}_crop.jpg"
        crop_path = crop_sub / crop_filename
        if sighting.best_crop is not None and sighting.best_crop.size > 0:
            cv2.imwrite(str(crop_path), sighting.best_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            best_crop_str = str(crop_path.as_posix())
        else:
            best_crop_str = ""

        # 2. Lưu ảnh toàn cảnh (context frame) nếu có
        if sighting.context_frame is not None and sighting.context_frame.size > 0:
            context_filename = f"{base_name}_ctx.jpg"
            context_path = context_sub / context_filename
            cv2.imwrite(str(context_path), sighting.context_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            context_str = str(context_path.as_posix())
        else:
            context_str = ""

        # 3. Chèn vào CSDL SQLite
        db_payload = {
            "source_id": source_id,
            "track_id": sighting.track_id,
            "plate": sighting.plate,
            "plate_raw": sighting.plate_raw,
            "is_corrected": 1 if sighting.is_corrected else 0,
            "corrections_detail": sighting.corrections_detail,
            "plate_conf": float(sighting.plate_conf),
            "is_valid": 1 if sighting.is_valid else 0,
            "t_start": float(sighting.t_start),
            "t_end": float(sighting.t_end),
            "video_pos_s": float(sighting.video_pos_s),
            "frame_start": int(sighting.frame_start),
            "frame_end": int(sighting.frame_end),
            "n_ocr": int(sighting.n_ocr),
            "lock_reason": sighting.lock_reason,
            "lock_latency_s": float(sighting.lock_latency_s),
            "best_crop_path": best_crop_str,
            "context_path": context_str,
            "clip_path": "",
            "frame_w": int(sighting.frame_w),
            "frame_h": int(sighting.frame_h),
            "trajectory": json.dumps(sighting.trajectory),
            "created_at": time.time(),
        }

        self.db.insert_sighting(db_payload)

    def close(self, timeout: float = 5.0):
        """Dừng worker thread và chờ hàng đợi xử lý nốt các bản ghi còn lại."""
        if not self.enabled:
            return

        self._stop_event.set()
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            pass

        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
            print(
                f"[Storage] Đã đóng Writer Thread an toàn. "
                f"Đã ghi: {self.written_count} lượt, Bị rơi: {self.dropped_count} lượt."
            )

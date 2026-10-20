"""
Module Ring Buffer lưu trữ khung hình quay vòng phục vụ trích xuất clip ngắn (tuỳ chọn).
Tự động thu nhỏ khung hình để tiết kiệm RAM tối đa trên Raspberry Pi 5.
"""

from collections import deque
from pathlib import Path
import time
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np

from src.config import ClipConfig


class ClipRingBuffer:
    """Bộ đệm khung hình tuần hoàn phục vụ ghi clip ngắn sự kiện."""

    def __init__(self, config: ClipConfig, fps: float = 25.0):
        self.config = config
        self.enabled = config.enabled
        self.fps = fps or 25.0

        if not self.enabled:
            return

        # Tính tổng số frame tối đa trong buffer (VD: pre_seconds = 3.0s -> ~75 frames)
        max_seconds = config.pre_seconds + config.post_seconds + 5.0
        max_frames = int(max_seconds * self.fps)
        self.buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_frames)

    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Thêm frame vào ring buffer sau khi đã thu nhỏ theo scale config."""
        if not self.enabled or frame is None or frame.size == 0:
            return

        scale = self.config.scale
        if scale < 1.0:
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scaled = frame

        self.buffer.append((timestamp, scaled))

    def export_clip(
        self,
        t_start: float,
        t_end: float,
        output_path: Path,
    ) -> bool:
        """Trích xuất danh sách frames trong khoảng [t_start - pre, t_end + post] và lưu thành file mp4."""
        if not self.enabled or len(self.buffer) == 0:
            return False

        t_min = t_start - self.config.pre_seconds
        t_max = t_end + self.config.post_seconds

        clip_frames = [f for t, f in self.buffer if t_min <= t <= t_max]
        if not clip_frames:
            return False

        h, w = clip_frames[0].shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))

        for f in clip_frames:
            writer.write(f)
        writer.release()
        return True

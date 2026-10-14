"""
Module phát hiện biển số sử dụng YOLO định dạng NCNN qua thư viện ultralytics.
"""

import time
from pathlib import Path
from typing import Tuple
import numpy as np
from ultralytics import YOLO

from src.config import DetectorConfig


class PlateDetector:
    """Wrapper cho mô hình phát hiện biển số xe (YOLO NCNN)."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình biển số tại: {model_path.resolve()}"
            )

        print(f"[Detector] Đang nạp mô hình YOLO NCNN từ: {model_path}...")
        t0 = time.perf_counter()
        self.model = YOLO(str(model_path), task="detect")
        self.load_time_ms = (time.perf_counter() - t0) * 1000
        print(f"[Detector] Nạp mô hình biển số thành công trong {self.load_time_ms:.2f} ms.")

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dự đoán vị trí biển số trên frame ảnh.

        Args:
            frame: Ảnh đầu vào dạng numpy.ndarray (BGR).

        Returns:
            Tuple gồm:
                - xyxy: np.ndarray shape (N, 4) - Tọa độ [x1, y1, x2, y2]
                - confidences: np.ndarray shape (N,) - Độ tin cậy
                - class_ids: np.ndarray shape (N,) - ID lớp
        """
        results = self.model(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            verbose=False,
        )

        if not results or len(results) == 0 or len(results[0].boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int),
            )

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes.cls is not None
            else np.zeros(len(xyxy), dtype=int)
        )

        return xyxy, confidences, class_ids

    def warmup(self, imgsz: int = 640):
        """Khởi động (warm-up) mô hình YOLO NCNN với dummy input để nạp bộ nhớ đệm."""
        try:
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            self.model(dummy, imgsz=imgsz, verbose=False)
        except Exception as e:
            print(f"[Detector Warning] Warm-up gặp lỗi nhỏ: {e}")


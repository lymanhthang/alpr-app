"""
Module phát hiện và theo dõi biển số xe sử dụng YOLO định dạng NCNN qua thư viện ultralytics.
Hỗ trợ:
- detect(): Nhận diện trên ảnh tĩnh (không duy trì trạng thái tracking).
- track(): Theo dõi đối tượng trên video / stream sử dụng model.track() của Ultralytics YOLO.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from src.config import DetectorConfig


class PlateDetector:
    """Wrapper cho mô hình phát hiện và theo dõi biển số xe (YOLO NCNN)."""

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

        # Bộ nhớ lưu vết track phục vụ video/stream và skip-frame
        self.last_active_tracks: List[Tuple[int, np.ndarray, float]] = []
        self._next_fallback_id: int = 1

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dự đoán vị trí biển số trên ảnh tĩnh (Static Image Detection).
        Không gán hay duy trì Track ID qua các frame.

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
        confidences = (
            boxes.conf.cpu().numpy()
            if boxes.conf is not None
            else np.ones(len(xyxy), dtype=np.float32)
        )
        class_ids = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes.cls is not None
            else np.zeros(len(xyxy), dtype=int)
        )

        return xyxy, confidences, class_ids

    def track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        tracker: Optional[str] = None,
    ) -> List[Tuple[int, np.ndarray, float]]:
        """
        Theo dõi biển số xe trên Video / Camera Stream sử dụng phương thức model.track() của Ultralytics YOLO.

        Args:
            frame: Khung hình video/stream đầu vào (BGR).
            persist: Duy trì ID bám vết qua các frame liên tiếp (mặc định True).
            tracker: File cấu hình tracker ('bytetrack.yaml' hoặc 'botsort.yaml').

        Returns:
            Danh sách active tracks: List[Tuple[track_id, xyxy_bbox, confidence]]
        """
        tracker_name = tracker or getattr(self.config, "tracker_type", "bytetrack.yaml")

        try:
            results = self.model.track(
                frame,
                persist=persist,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                tracker=tracker_name,
                verbose=False,
            )
        except Exception as e:
            # Dự phòng an toàn nếu model.track gặp lỗi cụ thể của backend
            print(f"[Detector Track Warning] model.track lỗi ({e}), chuyển sang fallback detect.")
            xyxy, confidences, _ = self.detect(frame)
            active_tracks = [
                (self._next_fallback_id + i, box, float(conf))
                for i, (box, conf) in enumerate(zip(xyxy, confidences))
            ]
            self._next_fallback_id += len(xyxy)
            self.last_active_tracks = active_tracks
            return active_tracks

        active_tracks: List[Tuple[int, np.ndarray, float]] = []

        if not results or len(results) == 0 or len(results[0].boxes) == 0:
            self.last_active_tracks = []
            return active_tracks

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = (
            boxes.conf.cpu().numpy()
            if boxes.conf is not None
            else np.ones(len(xyxy), dtype=np.float32)
        )

        # Lấy track ID do Ultralytics gán
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            # Fallback nếu track chưa gán id ở frame đầu
            track_ids = [self._next_fallback_id + i for i in range(len(xyxy))]
            self._next_fallback_id += len(xyxy)

        for tid, box, conf in zip(track_ids, xyxy, confidences):
            active_tracks.append((int(tid), box, float(conf)))

        self.last_active_tracks = active_tracks
        return active_tracks

    def get_tracks_for_skipped_frame(self) -> List[Tuple[int, np.ndarray, float]]:
        """
        Trả về danh sách track ở frame bị skip (giữ nguyên vị trí track gần nhất).
        Hỗ trợ kỹ thuật skip-frame không làm gián đoạn bám vết.
        """
        return list(self.last_active_tracks)

    def reset(self):
        """Đặt lại bộ theo dõi cho luồng video/stream mới."""
        self.last_active_tracks = []
        self._next_fallback_id = 1
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = None

    def update(
        self,
        xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
    ) -> List[Tuple[int, np.ndarray, float]]:
        """Phương thức hỗ trợ tương thích ngược với interface PlateTracker cũ."""
        active_tracks: List[Tuple[int, np.ndarray, float]] = []
        for i, (box, conf) in enumerate(zip(xyxy, confidences)):
            active_tracks.append((i + 1, box, float(conf)))
        self.last_active_tracks = active_tracks
        return active_tracks

    def warmup(self, imgsz: int = 640):
        """Khởi động (warm-up) mô hình YOLO NCNN với dummy input để nạp bộ nhớ đệm."""
        try:
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            self.model(dummy, imgsz=imgsz, verbose=False)
        except Exception as e:
            print(f"[Detector Warning] Warm-up gặp lỗi nhỏ: {e}")


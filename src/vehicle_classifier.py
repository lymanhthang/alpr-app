"""
Module phát hiện và phân loại 14 loại xe 2 bánh Việt Nam sử dụng YOLO định dạng NCNN.
Bao gồm VehicleClassifier (YOLO NCNN) và VehicleTracker (ByteTrack với Skip-Frame).
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import supervision as sv
from ultralytics import YOLO
import yaml

from src.config import TrackerConfig, VehicleClassifierConfig

# 15 nhãn mặc định được huấn luyện cho mô hình 14 loại xe 2 bánh phổ biến ở VN
DEFAULT_VEHICLE_CLASSES: Dict[int, str] = {
    0: "honda-air-blade",
    1: "honda-sh-mode",
    2: "honda-wave-alpha",
    3: "honda-sh",
    4: "honda-wave-rsx",
    5: "honda-vision",
    6: "yamaha-exciter",
    7: "honda-blade",
    8: "honda-future",
    9: "honda-winner-x",
    10: "honda-vario",
    11: "honda-lead",
    12: "yamaha-sirius",
    13: "yamaha-jupiter",
    14: "yamaha-pg1",
}


class VehicleClassifier:
    """Wrapper cho mô hình phát hiện & phân loại xe 2 bánh (YOLO NCNN)."""

    def __init__(self, config: VehicleClassifierConfig):
        self.config = config
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình phân loại xe tại: {model_path.resolve()}"
            )

        print(f"[VehicleClassifier] Đang nạp mô hình YOLO NCNN từ: {model_path}...")
        t0 = time.perf_counter()
        self.model = YOLO(str(model_path), task="detect")
        self.load_time_ms = (time.perf_counter() - t0) * 1000

        # Nạp tên nhãn từ metadata.yaml nếu có hoặc từ model.names
        self.class_names: Dict[int, str] = dict(DEFAULT_VEHICLE_CLASSES)
        metadata_file = model_path / "metadata.yaml"
        if metadata_file.is_file():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                    names = meta.get("names", {})
                    if isinstance(names, dict):
                        self.class_names.update({int(k): str(v) for k, v in names.items()})
                    elif isinstance(names, list):
                        self.class_names.update({i: str(v) for i, v in enumerate(names)})
            except Exception as e:
                print(f"[VehicleClassifier Warning] Lỗi đọc metadata.yaml: {e}")

        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            for k, v in self.model.names.items():
                self.class_names[int(k)] = str(v)

        print(
            f"[VehicleClassifier] Nạp mô hình phân loại xe thành công ({self.load_time_ms:.2f} ms), "
            f"hỗ trợ {len(self.class_names)} lớp xe."
        )

        # Bộ nhớ lưu vết track phục vụ video/stream và skip-frame
        self.last_active_tracks: List[Tuple[int, np.ndarray, float, int, str]] = []
        self._next_fallback_id: int = 1

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dự đoán vị trí và loại xe trên frame ảnh.

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

    def track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        tracker: Optional[str] = None,
    ) -> List[Tuple[int, np.ndarray, float, int, str]]:
        """
        Theo dõi vị trí và nhận dạng loại xe trên Video / Camera Stream sử dụng model.track() của Ultralytics YOLO.

        Returns:
            Danh sách các tuple: (track_id, xyxy_bbox, confidence, class_id, class_name)
        """
        tracker_name = tracker or "bytetrack.yaml"

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
            print(f"[VehicleClassifier Track Warning] model.track lỗi ({e}), chuyển sang fallback detect.")
            xyxy, confidences, class_ids = self.detect(frame)
            active_tracks = [
                (self._next_fallback_id + i, box, float(conf), int(cls_id), self.get_class_name(cls_id))
                for i, (box, conf, cls_id) in enumerate(zip(xyxy, confidences, class_ids))
            ]
            self._next_fallback_id += len(xyxy)
            self.last_active_tracks = active_tracks
            return active_tracks

        active_tracks: List[Tuple[int, np.ndarray, float, int, str]] = []
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
        class_ids = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes.cls is not None
            else np.zeros(len(xyxy), dtype=int)
        )

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            track_ids = [self._next_fallback_id + i for i in range(len(xyxy))]
            self._next_fallback_id += len(xyxy)

        for tid, box, conf, cls_id in zip(track_ids, xyxy, confidences, class_ids):
            cls_name = self.get_class_name(cls_id)
            active_tracks.append((int(tid), box, float(conf), int(cls_id), cls_name))

        self.last_active_tracks = active_tracks
        return active_tracks

    def get_tracks_for_skipped_frame(self) -> List[Tuple[int, np.ndarray, float, int, str]]:
        """Trả về danh sách track xe ở frame bị skip (giữ nguyên vị trí track gần nhất)."""
        return list(self.last_active_tracks)

    def reset(self):
        """Đặt lại bộ theo dõi xe cho luồng video/stream mới."""
        self.last_active_tracks = []
        self._next_fallback_id = 1
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = None

    def get_class_name(self, class_id: int) -> str:
        """Lấy tên định danh lớp xe từ class_id."""
        return self.class_names.get(int(class_id), f"vehicle_{class_id}")

    def warmup(self, imgsz: Tuple[int, int] = (384, 640)):
        """Khởi động mô hình NCNN với dummy input."""
        try:
            h, w = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz, imgsz)
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            self.model(dummy, imgsz=imgsz, verbose=False)
        except Exception as e:
            print(f"[VehicleClassifier Warning] Warm-up gặp lỗi nhỏ: {e}")


class VehicleTracker:
    """Theo dõi vị trí và ID các xe 2 bánh qua từng khung hình (ByteTrack)."""

    def __init__(self, tracker_config: TrackerConfig):
        self.tracker_config = tracker_config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=tracker_config.track_activation_threshold,
            lost_track_buffer=tracker_config.lost_track_buffer,
            minimum_matching_threshold=tracker_config.minimum_matching_threshold,
            frame_rate=30,
        )
        # Lưu vết các track xe đang active: List[Tuple[track_id, box, conf, class_id, class_name]]
        self.last_active_tracks: List[Tuple[int, np.ndarray, float, int, str]] = []

    def update(
        self,
        xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        class_name_fn=None,
    ) -> List[Tuple[int, np.ndarray, float, int, str]]:
        """
        Cập nhật tracker với các bounding box xe từ YOLO.

        Returns:
            Danh sách các tuple: (track_id, xyxy_bbox, confidence, class_id, class_name)
        """
        if len(xyxy) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidences,
                class_id=class_ids,
            )

        tracked_detections = self.tracker.update_with_detections(detections)

        active_tracks: List[Tuple[int, np.ndarray, float, int, str]] = []
        if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                box = tracked_detections.xyxy[i]
                conf = (
                    float(tracked_detections.confidence[i])
                    if tracked_detections.confidence is not None
                    else 1.0
                )
                cls_id = (
                    int(tracked_detections.class_id[i])
                    if tracked_detections.class_id is not None
                    else 0
                )
                cls_name = class_name_fn(cls_id) if class_name_fn else f"vehicle_{cls_id}"
                active_tracks.append((int(track_id), box, conf, cls_id, cls_name))

        self.last_active_tracks = active_tracks
        return active_tracks

    def get_tracks_for_skipped_frame(self) -> List[Tuple[int, np.ndarray, float, int, str]]:
        """Trả về danh sách track ở frame bị skip để giữ nguyên vị trí, tránh chớp nháy."""
        return self.last_active_tracks

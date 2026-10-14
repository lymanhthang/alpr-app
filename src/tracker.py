"""
Module theo dõi biển số qua các frame bằng ByteTrack (thư viện supervision).
"""

from typing import List, Tuple
import warnings
import numpy as np
import supervision as sv

# Tắt cảnh báo deprecation của ByteTrack trong các phiên bản supervision mới
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

from src.config import TrackerConfig


class PlateTracker:
    """Wrapper cho ByteTrack để theo dõi biển số và gán Track ID ổn định."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.tracker = sv.ByteTrack(
                track_activation_threshold=config.track_activation_threshold,
                lost_track_buffer=config.lost_track_buffer,
                minimum_matching_threshold=config.minimum_matching_threshold,
                frame_rate=30,
            )
        # Lưu vết các track đang active ở frame gần nhất để phục vụ skip-frame
        self.last_active_tracks: List[Tuple[int, np.ndarray, float]] = []

    def update(
        self,
        xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
    ) -> List[Tuple[int, np.ndarray, float]]:
        """
        Cập nhật tracker với các bounding box phát hiện được từ YOLO.

        Returns:
            Danh sách các tuple: (track_id, xyxy_bbox, confidence)
        """
        if len(xyxy) == 0:
            # Tạo Detections rỗng
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidences,
                class_id=class_ids,
            )

        tracked_detections = self.tracker.update_with_detections(detections)

        active_tracks: List[Tuple[int, np.ndarray, float]] = []
        if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                box = tracked_detections.xyxy[i]
                conf = (
                    float(tracked_detections.confidence[i])
                    if tracked_detections.confidence is not None
                    else 1.0
                )
                active_tracks.append((int(track_id), box, conf))

        self.last_active_tracks = active_tracks
        return active_tracks

    def get_tracks_for_skipped_frame(self) -> List[Tuple[int, np.ndarray, float]]:
        """
        Trả về danh sách track ở frame bị skip (giữ nguyên vị trí track gần nhất để không bị chớp giật).
        """
        return self.last_active_tracks

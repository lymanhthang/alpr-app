"""
Module quản lý vòng đời và trạng thái của từng track xe (TrackStateManager).
Theo dõi quỹ đạo, lưu trữ ảnh crop đẹp nhất, khung cảnh (context frame), và chốt kết quả sighting.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from src.config import OCRConfig, TrackerConfig
from src.reliability.correction import correct_vn_plate, CorrectionDetail
from src.reliability.quality import calculate_crop_quality, QualityResult
from src.reliability.voting import PlateVoter, VoteResult, Observation


@dataclass
class FinalizedSighting:
    """Đối tượng lượt xe đã hoàn tất (sighting) chuẩn bị ghi vào cơ sở dữ liệu."""

    track_id: int
    plate: str
    plate_raw: str
    is_corrected: bool
    corrections_detail: str  # JSON string
    plate_conf: float
    is_valid: bool
    t_start: float
    t_end: float
    video_pos_s: float
    frame_start: int
    frame_end: int
    n_ocr: int
    lock_reason: str
    lock_latency_s: float
    best_crop: np.ndarray
    context_frame: Optional[np.ndarray]
    frame_w: int
    frame_h: int
    trajectory: List[Tuple[float, float, float, float, float]]  # [t_rel, norm_cx, norm_cy, norm_w, norm_h]
    clip_bytes: Optional[bytes] = None


class TrackState:
    """Lưu giữ toàn bộ thông tin trạng thái theo vòng đời của 1 Track ID."""

    def __init__(
        self,
        track_id: int,
        start_frame: int,
        start_time: float,
        video_pos_s: float,
        frame_shape: Tuple[int, int],
    ):
        self.track_id = track_id
        self.frame_start = start_frame
        self.frame_end = start_frame
        self.t_start = start_time
        self.t_end = start_time
        self.video_pos_s = video_pos_s
        self.frame_h, self.frame_w = frame_shape[:2]

        # Ảnh crop và ảnh ngữ cảnh đẹp nhất
        self.best_crop: Optional[np.ndarray] = None
        self.best_quality: float = -1.0
        self.best_frame_idx: int = start_frame
        self.context_frame: Optional[np.ndarray] = None

        # Quỹ đạo: lưu danh sách các điểm [t_offset, cx, cy, w, h] chuẩn hóa 0.0 - 1.0
        self.trajectory: List[Tuple[float, float, float, float, float]] = []

        # Danh sách quan sát OCR
        self.observations: List[Observation] = []
        self.ocr_count: int = 0
        self.last_ocr_frame: int = -999

        # Trạng thái chốt (Lock)
        self.is_locked: bool = False
        self.locked_plate: str = ""
        self.locked_plate_raw: str = ""
        self.locked_is_corrected: bool = False
        self.locked_corrections: List[Dict[str, Any]] = []
        self.locked_conf: float = 0.0
        self.locked_is_valid: bool = False
        self.locked_at_frame: int = -1
        self.lock_reason: str = "pending"

        # Kết quả biểu quyết mới nhất (provisional)
        self.latest_plate: str = ""
        self.latest_is_valid: bool = False

    def add_trajectory_point(
        self,
        bbox: np.ndarray,
        frame_idx: int,
        curr_time: float,
    ):
        """Thêm điểm tọa độ vào quỹ đạo di chuyển (chỉ gọi ở detector frame)."""
        self.frame_end = frame_idx
        self.t_end = curr_time

        x1, y1, x2, y2 = bbox[:4]
        cx = float((x1 + x2) / 2.0) / self.frame_w
        cy = float((y1 + y2) / 2.0) / self.frame_h
        bw = float(x2 - x1) / self.frame_w
        bh = float(y2 - y1) / self.frame_h
        t_offset = round(curr_time - self.t_start, 3)

        self.trajectory.append((
            t_offset,
            round(cx, 4),
            round(cy, 4),
            round(bw, 4),
            round(bh, 4),
        ))

    def update_best_crop(
        self,
        crop: np.ndarray,
        quality: float,
        frame_idx: int,
        full_frame: Optional[np.ndarray] = None,
    ):
        """Cập nhật ảnh crop có điểm chất lượng cao nhất."""
        if quality > self.best_quality and crop is not None and crop.size > 0:
            self.best_quality = quality
            self.best_crop = crop.copy()
            self.best_frame_idx = frame_idx

            # Lưu ảnh context frame (thu nhỏ về chiều rộng 640px để tiết kiệm bộ nhớ)
            if full_frame is not None and full_frame.size > 0:
                fh, fw = full_frame.shape[:2]
                cw = 640
                ch = int(fh * (cw / fw))
                self.context_frame = cv2.resize(full_frame, (cw, ch), interpolation=cv2.INTER_AREA)

    def add_ocr_result(
        self,
        raw_text: str,
        conf: float,
        quality: float,
        frame_idx: int,
        use_char_correction: bool = True,
        use_quality_selection: bool = True,
        early_lock_votes: int = 3,
        early_lock_conf: float = 0.85,
    ) -> VoteResult:
        """Thêm kết quả OCR vào bộ bỏ phiếu và đánh giá early-lock."""
        self.last_ocr_frame = frame_idx
        self.ocr_count += 1

        # 1. Sửa ký tự theo chuẩn biển số Việt Nam nếu bật
        if use_char_correction:
            corr = correct_vn_plate(raw_text)
            text_corrected = corr.plate_corrected
            is_corrected = corr.is_corrected
            corrections = corr.corrections
        else:
            text_corrected = raw_text
            is_corrected = False
            corrections = []

        obs = Observation(
            text_raw=raw_text,
            text_corrected=text_corrected,
            is_corrected=is_corrected,
            corrections=corrections,
            conf=conf,
            quality=quality,
            frame_idx=frame_idx,
        )
        self.observations.append(obs)

        # 2. Chạy bỏ phiếu bầu chọn
        vote_res = PlateVoter.vote(
            observations=self.observations,
            use_quality=use_quality_selection,
            early_lock_votes=early_lock_votes,
            early_lock_conf=early_lock_conf,
        )

        self.latest_plate = vote_res.plate
        self.latest_is_valid = vote_res.is_valid

        # 3. Kiểm tra khóa sớm
        if vote_res.should_early_lock and not self.is_locked:
            self.is_locked = True
            self.locked_plate = vote_res.plate
            self.locked_plate_raw = vote_res.plate_raw
            self.locked_is_corrected = vote_res.is_corrected
            self.locked_corrections = vote_res.corrections_detail
            self.locked_conf = vote_res.confidence
            self.locked_is_valid = vote_res.is_valid
            self.locked_at_frame = frame_idx
            self.lock_reason = vote_res.lock_reason

        return vote_res

    def finalize(
        self,
        use_quality_selection: bool = True,
    ) -> FinalizedSighting:
        """Chốt thông tin lượt xe khi track kết thúc hoặc kết thúc luồng."""
        # Nếu chưa được early lock, thực hiện bỏ phiếu lần cuối từ toàn bộ observations
        if not self.is_locked:
            vote_res = PlateVoter.vote(
                observations=self.observations,
                use_quality=use_quality_selection,
            )
            final_plate = vote_res.plate
            final_raw = vote_res.plate_raw
            final_is_corr = vote_res.is_corrected
            final_corr_detail = vote_res.corrections_detail
            final_conf = vote_res.confidence
            final_valid = vote_res.is_valid
            lock_reason = "track_end_vote"
        else:
            final_plate = self.locked_plate
            final_raw = self.locked_plate_raw
            final_is_corr = self.locked_is_corrected
            final_corr_detail = self.locked_corrections
            final_conf = self.locked_conf
            final_valid = self.locked_is_valid
            lock_reason = self.lock_reason

        lock_latency_s = (
            max(0.0, self.t_end - self.t_start)
            if self.locked_at_frame < 0
            else max(0.0, ((self.locked_at_frame - self.frame_start) / 30.0))
        )

        return FinalizedSighting(
            track_id=self.track_id,
            plate=final_plate,
            plate_raw=final_raw,
            is_corrected=final_is_corr,
            corrections_detail=json.dumps(final_corr_detail, ensure_ascii=False),
            plate_conf=final_conf,
            is_valid=final_valid,
            t_start=self.t_start,
            t_end=self.t_end,
            video_pos_s=self.video_pos_s,
            frame_start=self.frame_start,
            frame_end=self.frame_end,
            n_ocr=self.ocr_count,
            lock_reason=lock_reason,
            lock_latency_s=round(lock_latency_s, 2),
            best_crop=self.best_crop if self.best_crop is not None else np.zeros((32, 64, 3), dtype=np.uint8),
            context_frame=self.context_frame,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            trajectory=self.trajectory,
        )


class TrackStateManager:
    """Quản lý tập hợp tất cả các track đang hoạt động trong pipeline."""

    def __init__(self, tracker_config: TrackerConfig, ocr_config: OCRConfig):
        self.tracker_config = tracker_config
        self.ocr_config = ocr_config
        self.tracks: Dict[int, TrackState] = {}
        self.lost_track_buffer = tracker_config.lost_track_buffer

    def get_or_create(
        self,
        track_id: int,
        frame_idx: int,
        curr_time: float,
        video_pos_s: float,
        frame_shape: Tuple[int, int],
    ) -> TrackState:
        """Lấy hoặc tạo mới đối tượng TrackState cho track_id."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                start_frame=frame_idx,
                start_time=curr_time,
                video_pos_s=video_pos_s,
                frame_shape=frame_shape,
            )
        return self.tracks[track_id]

    def should_ocr(
        self,
        track: TrackState,
        frame_idx: int,
        crop_quality: float,
    ) -> bool:
        """
        Xác định xem có nên chạy OCR cho track tại frame hiện tại không:
        1. Nếu track đã bị khóa (early lock), không chạy OCR nữa.
        2. Nếu đã vượt quá max_ocr_per_track, không chạy nữa.
        3. Kiểm tra cooldown (retry_interval_frames).
        4. Kiểm tra ngưỡng điểm chất lượng crop (quality_min_score).
        """
        if track.is_locked:
            return False

        if track.ocr_count >= self.ocr_config.max_ocr_per_track:
            return False

        # Kiểm tra khoảng cách frame giữa 2 lần gọi OCR
        if (frame_idx - track.last_ocr_frame) <= self.ocr_config.retry_interval_frames:
            return False

        # Kiểm tra chất lượng nếu bật quality selection
        if self.ocr_config.use_quality_selection:
            if crop_quality < self.ocr_config.quality_min_score:
                return False

        return True

    def pop_finished(
        self,
        current_frame_idx: int,
        active_track_ids: Set[int],
    ) -> List[FinalizedSighting]:
        """Phát hiện và hoàn tất các track đã rời khỏi khung hình."""
        finished: List[FinalizedSighting] = []
        to_delete: List[int] = []

        for track_id, track in self.tracks.items():
            if track_id not in active_track_ids:
                # Track không xuất hiện trong active_tracks
                lost_duration = current_frame_idx - track.frame_end
                if lost_duration >= self.lost_track_buffer:
                    finished.append(track.finalize(use_quality_selection=self.ocr_config.use_quality_selection))
                    to_delete.append(track_id)

        for tid in to_delete:
            del self.tracks[tid]

        return finished

    def flush_all(self) -> List[FinalizedSighting]:
        """Chốt toàn bộ các track còn lại khi kết thúc video hoặc nhận lệnh dừng."""
        finished: List[FinalizedSighting] = []
        for track in self.tracks.values():
            finished.append(track.finalize(use_quality_selection=self.ocr_config.use_quality_selection))
        self.tracks.clear()
        return finished

"""
Package cung cấp các module tăng cường độ tin cậy nhận diện biển số:
- quality: Đánh giá chất lượng frame/crop
- correction: Sửa lỗi ký tự OCR theo vị trí chuẩn biển số Việt Nam
- voting: Bỏ phiếu đa frame theo trọng số
- track_state: Quản lý vòng đời và trạng thái của track
"""

from src.reliability.quality import calculate_crop_quality, QualityResult
from src.reliability.correction import correct_vn_plate, CorrectionDetail
from src.reliability.voting import PlateVoter, VoteResult, Observation
from src.reliability.track_state import TrackState, TrackStateManager, FinalizedSighting

__all__ = [
    "calculate_crop_quality",
    "QualityResult",
    "correct_vn_plate",
    "CorrectionDetail",
    "PlateVoter",
    "VoteResult",
    "Observation",
    "TrackState",
    "TrackStateManager",
    "FinalizedSighting",
]

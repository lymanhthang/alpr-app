"""
Unit test kiểm tra tính năng bỏ phiếu đa frame có trọng số.
"""

from src.reliability.voting import Observation, PlateVoter


def test_per_position_voting():
    # Giả lập 3 quan sát: 2 quan sát đúng '51F12345', 1 quan sát lỗi OCR '51F1Z345'
    observations = [
        Observation(
            text_raw="51F1Z345",
            text_corrected="51F1Z345",
            is_corrected=False,
            corrections=[],
            conf=0.75,
            quality=0.6,
            frame_idx=0,
        ),
        Observation(
            text_raw="51F12345",
            text_corrected="51F12345",
            is_corrected=False,
            corrections=[],
            conf=0.90,
            quality=0.8,
            frame_idx=3,
        ),
        Observation(
            text_raw="51F12345",
            text_corrected="51F12345",
            is_corrected=False,
            corrections=[],
            conf=0.88,
            quality=0.85,
            frame_idx=6,
        ),
    ]

    res = PlateVoter.vote(observations, use_quality=True)
    assert res.plate == "51F12345"
    assert res.is_valid is True
    assert res.confidence > 0.6


def test_early_lock_trigger():
    # 3 quan sát liên tiếp đồng thuận với độ tin cậy cao
    obs = [
        Observation(
            text_raw="30A12345",
            text_corrected="30A12345",
            is_corrected=False,
            corrections=[],
            conf=0.95,
            quality=0.9,
            frame_idx=i * 3,
        )
        for i in range(3)
    ]

    res = PlateVoter.vote(obs, early_lock_votes=3, early_lock_conf=0.85)
    assert res.should_early_lock is True
    assert res.plate == "30A12345"
    assert "early_lock" in res.lock_reason

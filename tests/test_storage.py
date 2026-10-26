"""
Unit test kiểm tra SQLite WAL database và truy vấn sightings.
"""

import time
from pathlib import Path
import tempfile
import pytest

from src.storage.db import DatabaseManager


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_alpr.db"
        yield DatabaseManager(str(db_path))


def test_db_source_and_sighting(temp_db):
    src_id = temp_db.get_or_create_source(name="Cam_Cong_Chinh", kind="camera", uri="0")
    assert src_id > 0

    # Lấy lại cùng tên phải ra cùng id
    src_id_2 = temp_db.get_or_create_source(name="Cam_Cong_Chinh")
    assert src_id == src_id_2

    # Chèn một sighting
    s_id = temp_db.insert_sighting({
        "source_id": src_id,
        "track_id": 1,
        "plate": "30A12345",
        "plate_raw": "30A12345",
        "is_corrected": 0,
        "corrections_detail": "[]",
        "plate_conf": 0.95,
        "is_valid": 1,
        "t_start": 1000.0,
        "t_end": 1005.0,
        "video_pos_s": 0.0,
        "frame_start": 0,
        "frame_end": 150,
        "n_ocr": 3,
        "lock_reason": "early_lock",
        "lock_latency_s": 0.5,
        "best_crop_path": "data/crops/dummy.jpg",
        "context_path": "",
        "clip_path": "",
        "frame_w": 1280,
        "frame_h": 720,
        "trajectory": "[]",
        "vehicle_type": "honda-vision",
        "vehicle_conf": 0.92,
        "created_at": time.time(),
    })
    assert s_id > 0

    item = temp_db.get_sighting_by_id(s_id)
    assert item is not None
    assert item["plate"] == "30A12345"
    assert item["vehicle_type"] == "honda-vision"
    assert item["vehicle_conf"] == 0.92
    assert item["source_name"] == "Cam_Cong_Chinh"

    # Truy vấn lọc
    res = temp_db.query_sightings(plate="30A")
    assert len(res) == 1
    assert res[0]["id"] == s_id

    # Truy vấn lọc theo vehicle_type
    res_v = temp_db.query_sightings(vehicle_type="vision")
    assert len(res_v) == 1
    assert res_v[0]["vehicle_type"] == "honda-vision"

    # Thêm một sighting UNKNOWN (không hợp lệ)
    s_unk_id = temp_db.insert_sighting({
        "source_id": src_id,
        "track_id": 2,
        "plate": "UNKNOWN",
        "plate_raw": "",
        "is_corrected": 0,
        "corrections_detail": "[]",
        "plate_conf": 0.0,
        "is_valid": 0,
        "t_start": 1010.0,
        "t_end": 1015.0,
        "video_pos_s": 10.0,
        "frame_start": 151,
        "frame_end": 200,
        "n_ocr": 0,
        "lock_reason": "none",
        "lock_latency_s": 0.0,
        "best_crop_path": "data/crops/dummy_unk.jpg",
        "context_path": "",
        "clip_path": "",
        "frame_w": 1280,
        "frame_h": 720,
        "trajectory": "[]",
        "created_at": time.time(),
    })
    assert s_unk_id > 0

    # Test lọc is_valid_only
    valid_res = temp_db.query_sightings(is_valid_only=True)
    assert len(valid_res) == 1
    assert valid_res[0]["id"] == s_id

    # Test lọc is_unknown_only
    unknown_res = temp_db.query_sightings(is_unknown_only=True)
    assert len(unknown_res) == 1
    assert unknown_res[0]["id"] == s_unk_id

    # Test xóa toàn bộ dữ liệu giữ nguyên cấu trúc bảng
    res_clear = temp_db.clear_all_data(reset_sources=True)
    assert res_clear["deleted_sightings"] == 2
    assert res_clear["deleted_sources"] == 1

    # Kiểm tra bảng rỗng nhưng schema còn nguyên
    stats = temp_db.get_stats()
    assert stats["total_sightings"] == 0
    assert stats["total_unique_plates"] == 0
    assert len(temp_db.get_all_sources()) == 0

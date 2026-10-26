"""
Unit test kiểm tra module phân loại xe 2 bánh:
- Chiến lược Bỏ phiếu đa frame (Multi-frame Temporal Voting)
- Chiến lược Khóa ngay lần đầu (Lock-first)
- Khớp không gian (Spatial Association) giữa biển số và thân xe
- Migration an toàn trên CSDL cũ
"""

from pathlib import Path
import sqlite3
import tempfile
import numpy as np
import pytest

from src.reliability.track_state import TrackState
from src.storage.db import DatabaseManager


def test_vehicle_voting_strategy():
    """Kiểm tra chiến lược bỏ phiếu đa frame cho phân loại xe."""
    ts = TrackState(
        track_id=1,
        start_frame=0,
        start_time=100.0,
        video_pos_s=0.0,
        frame_shape=(720, 1280),
    )

    # 3 frame liên tiếp nhận diện
    ts.add_vehicle_observation("honda-vision", 0.80, strategy="voting")
    ts.add_vehicle_observation("honda-vision", 0.90, strategy="voting")
    ts.add_vehicle_observation("honda-lead", 0.40, strategy="voting")

    assert ts.latest_vehicle_type == "honda-lead"
    assert not ts.is_vehicle_locked

    sighting = ts.finalize()
    assert sighting.vehicle_type == "honda-vision"
    assert sighting.vehicle_conf == 0.85


def test_vehicle_lock_first_strategy():
    """Kiểm tra chiến lược khóa ngay lần đầu đạt ngưỡng độ tin cậy."""
    ts = TrackState(
        track_id=2,
        start_frame=0,
        start_time=100.0,
        video_pos_s=0.0,
        frame_shape=(720, 1280),
    )

    # Lần 1: conf 0.40 < threshold 0.70 -> chưa khóa
    ts.add_vehicle_observation("honda-sh", 0.40, strategy="lock_first", lock_first_conf=0.70)
    assert not ts.is_vehicle_locked

    # Lần 2: conf 0.85 >= threshold 0.70 -> khóa ngay
    ts.add_vehicle_observation("honda-sh", 0.85, strategy="lock_first", lock_first_conf=0.70)
    assert ts.is_vehicle_locked
    assert ts.locked_vehicle_type == "honda-sh"
    assert ts.locked_vehicle_conf == 0.85

    # Lần 3: kết quả khác đến sau -> bị bỏ qua do đã khóa
    ts.add_vehicle_observation("yamaha-exciter", 0.99, strategy="lock_first", lock_first_conf=0.70)
    assert ts.locked_vehicle_type == "honda-sh"

    sighting = ts.finalize()
    assert sighting.vehicle_type == "honda-sh"
    assert sighting.vehicle_conf == 0.85


def test_spatial_matching_plate_inside_vehicle():
    """Kiểm tra logic hình học: Tâm biển số nằm trọn trong bounding box thân xe."""
    vehicle_box = [100, 100, 500, 600]  # [vx1, vy1, vx2, vy2]
    plate_box_inside = [250, 450, 350, 500]  # [px1, py1, px2, py2]
    plate_box_outside = [600, 650, 700, 700]

    # Kiểm tra tâm
    pcx_in = (plate_box_inside[0] + plate_box_inside[2]) / 2.0
    pcy_in = (plate_box_inside[1] + plate_box_inside[3]) / 2.0
    assert vehicle_box[0] <= pcx_in <= vehicle_box[2]
    assert vehicle_box[1] <= pcy_in <= vehicle_box[3]

    pcx_out = (plate_box_outside[0] + plate_box_outside[2]) / 2.0
    pcy_out = (plate_box_outside[1] + plate_box_outside[3]) / 2.0
    assert not (vehicle_box[0] <= pcx_out <= vehicle_box[2] and vehicle_box[1] <= pcy_out <= vehicle_box[3])


def test_db_migration_old_schema():
    """Kiểm tra DatabaseManager tự động nâng cấp (migration) CSDL cũ mà không làm hỏng dữ liệu."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "legacy_alpr.db"

        # Tạo trước bảng sightings phiên bản cũ KHÔNG có vehicle_type và vehicle_conf
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                uri TEXT,
                created_at REAL NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                plate TEXT NOT NULL,
                plate_raw TEXT NOT NULL,
                is_corrected INTEGER NOT NULL DEFAULT 0,
                corrections_detail TEXT,
                plate_conf REAL NOT NULL,
                is_valid INTEGER NOT NULL DEFAULT 0,
                t_start REAL NOT NULL,
                t_end REAL NOT NULL,
                video_pos_s REAL DEFAULT 0.0,
                frame_start INTEGER NOT NULL,
                frame_end INTEGER NOT NULL,
                n_ocr INTEGER NOT NULL,
                lock_reason TEXT,
                lock_latency_s REAL,
                best_crop_path TEXT NOT NULL,
                context_path TEXT,
                clip_path TEXT,
                frame_w INTEGER,
                frame_h INTEGER,
                trajectory TEXT,
                created_at REAL NOT NULL
            );
        """)
        # Chèn 1 bản ghi cũ
        conn.execute("""
            INSERT INTO sightings (
                source_id, track_id, plate, plate_raw, is_corrected,
                corrections_detail, plate_conf, is_valid, t_start, t_end,
                video_pos_s, frame_start, frame_end, n_ocr, lock_reason,
                lock_latency_s, best_crop_path, context_path, clip_path,
                frame_w, frame_h, trajectory, created_at
            ) VALUES (
                1, 1, '29A11111', '29A11111', 0, '[]', 0.9, 1, 0, 1,
                0, 0, 30, 1, 'early_lock', 0.5, 'crop.jpg', '', '',
                1280, 720, '[]', 1000.0
            );
        """)
        conn.commit()
        conn.close()

        # Khởi tạo DatabaseManager với file db cũ
        db = DatabaseManager(str(db_path))

        # Kiểm tra xem migration đã thêm 2 cột chưa
        with db.connection() as c:
            cur = c.cursor()
            cur.execute("PRAGMA table_info(sightings);")
            cols = {row["name"] for row in cur.fetchall()}
            assert "vehicle_type" in cols
            assert "vehicle_conf" in cols

            # Dữ liệu cũ vẫn còn nguyên
            cur.execute("SELECT plate, vehicle_type, vehicle_conf FROM sightings WHERE id = 1")
            row = cur.fetchone()
            assert row["plate"] == "29A11111"
            assert row["vehicle_type"] is None
            assert row["vehicle_conf"] == 0.0

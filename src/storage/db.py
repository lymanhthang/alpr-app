"""
Module quản lý cơ sở dữ liệu SQLite tối ưu với chế độ WAL (Write-Ahead Logging).
Bảng sources và sightings có đầy đủ index và hỗ trợ tìm kiếm theo khoảng thời gian, biển số, camera.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple


class DatabaseManager:
    """Quản lý kết nối và truy vấn SQLite cho hệ thống ALPR."""

    def __init__(self, db_path: str = "data/alpr.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Tạo kết nối SQLite mới kèm cấu hình WAL và timeout an toàn."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Khởi tạo schema bảng và các index cần thiết."""
        with self.connection() as conn:
            # 1. Bảng lưu trữ danh sách các nguồn (Camera / Video)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL CHECK(kind IN ('camera', 'video')),
                    uri TEXT,
                    created_at REAL NOT NULL
                );
            """)

            # 2. Bảng lưu từng lượt xe (Sightings)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
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
                    vehicle_type TEXT,
                    vehicle_conf REAL DEFAULT 0.0,
                    created_at REAL NOT NULL
                );
            """)

            # Tự động nâng cấp schema nếu database đã tồn tại từ các phiên bản trước (Migration an toàn)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(sightings);")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            if "vehicle_type" not in existing_cols:
                conn.execute("ALTER TABLE sightings ADD COLUMN vehicle_type TEXT DEFAULT NULL;")
            if "vehicle_conf" not in existing_cols:
                conn.execute("ALTER TABLE sightings ADD COLUMN vehicle_conf REAL DEFAULT 0.0;")

            # 3. Tạo các Index tối ưu hóa truy vấn tra cứu
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sightings_plate ON sightings(plate);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sightings_time ON sightings(t_start);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sightings_source_time ON sightings(source_id, t_start);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sightings_is_corrected ON sightings(is_corrected);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sightings_is_valid ON sightings(is_valid);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sightings_vehicle_type ON sightings(vehicle_type);")

    def get_or_create_source(self, name: str, kind: str = "video", uri: str = "") -> int:
        """Lấy id của source theo tên hoặc tạo mới nếu chưa tồn tại."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sources WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return int(row["id"])

            cursor.execute(
                "INSERT INTO sources (name, kind, uri, created_at) VALUES (?, ?, ?, ?)",
                (name, kind, uri, datetime.now().timestamp()),
            )
            return int(cursor.lastrowid)

    def insert_sighting(self, data: Dict[str, Any]) -> int:
        """Chèn một bản ghi lượt xe vào bảng sightings."""
        payload = dict(data)
        payload.setdefault("vehicle_type", None)
        payload.setdefault("vehicle_conf", 0.0)

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sightings (
                    source_id, track_id, plate, plate_raw, is_corrected,
                    corrections_detail, plate_conf, is_valid, t_start, t_end,
                    video_pos_s, frame_start, frame_end, n_ocr, lock_reason,
                    lock_latency_s, best_crop_path, context_path, clip_path,
                    frame_w, frame_h, trajectory, vehicle_type, vehicle_conf, created_at
                ) VALUES (
                    :source_id, :track_id, :plate, :plate_raw, :is_corrected,
                    :corrections_detail, :plate_conf, :is_valid, :t_start, :t_end,
                    :video_pos_s, :frame_start, :frame_end, :n_ocr, :lock_reason,
                    :lock_latency_s, :best_crop_path, :context_path, :clip_path,
                    :frame_w, :frame_h, :trajectory, :vehicle_type, :vehicle_conf, :created_at
                )
                """,
                payload,
            )
            return int(cursor.lastrowid)

    def get_sighting_by_id(self, sighting_id: int) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết một lượt xe theo ID kèm thông tin source."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.*, src.name as source_name, src.kind as source_kind
                FROM sightings s
                JOIN sources src ON s.source_id = src.id
                WHERE s.id = ?
                """,
                (sighting_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def query_sightings(
        self,
        plate: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        source_id: Optional[int] = None,
        time_from: Optional[float] = None,
        time_to: Optional[float] = None,
        is_valid_only: Optional[bool] = None,
        is_unknown_only: Optional[bool] = None,
        is_corrected_only: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Truy vấn danh sách lượt xe với các bộ lọc tiêu chuẩn."""
        conditions = []
        params = []

        if plate:
            conditions.append("s.plate LIKE ?")
            params.append(f"%{plate.strip().upper()}%")

        if vehicle_type:
            conditions.append("s.vehicle_type LIKE ?")
            params.append(f"%{vehicle_type.strip().lower()}%")

        if source_id is not None:
            conditions.append("s.source_id = ?")
            params.append(source_id)

        if time_from is not None:
            conditions.append("s.t_start >= ?")
            params.append(time_from)

        if time_to is not None:
            conditions.append("s.t_start <= ?")
            params.append(time_to)

        if is_valid_only:
            conditions.append("s.is_valid = 1")
        elif is_unknown_only:
            conditions.append("s.is_valid = 0")

        if is_corrected_only:
            conditions.append("s.is_corrected = 1")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT s.*, src.name as source_name, src.kind as source_kind
            FROM sightings s
            JOIN sources src ON s.source_id = src.id
            {where_clause}
            ORDER BY s.t_start DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
            if conditions:
                print(f"[DB Query] SQL Filter: {where_clause} -> Trả về {len(rows)} kết quả")
            return rows

    def get_all_plates_distinct(self) -> List[str]:
        """Lấy danh sách các biển số phân biệt để phục vụ tìm kiếm mờ (Fuzzy matching)."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT plate FROM sightings WHERE plate != ''")
            return [row["plate"] for row in cursor.fetchall()]

    def get_all_sources(self) -> List[Dict[str, Any]]:
        """Lấy danh sách các nguồn (Camera/Video)."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sources ORDER BY name ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Thống kê tổng quan số lượng sightings, biển hợp lệ, biển unknown, số lượt hiệu chỉnh và xe đã phân loại."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sightings,
                    SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) as total_valid,
                    SUM(CASE WHEN is_valid = 0 THEN 1 ELSE 0 END) as total_unknown,
                    SUM(CASE WHEN is_corrected = 1 THEN 1 ELSE 0 END) as total_corrected,
                    SUM(CASE WHEN vehicle_type IS NOT NULL AND vehicle_type != '' THEN 1 ELSE 0 END) as total_classified,
                    COUNT(DISTINCT plate) as total_unique_plates
                FROM sightings
            """)
            row = cursor.fetchone()
            return {
                "total_sightings": row["total_sightings"] or 0,
                "total_valid": row["total_valid"] or 0,
                "total_unknown": row["total_unknown"] or 0,
                "total_corrected": row["total_corrected"] or 0,
                "total_classified": row["total_classified"] or 0,
                "total_unique_plates": row["total_unique_plates"] or 0,
            }

    def clear_all_data(self, reset_sources: bool = True) -> Dict[str, int]:
        """
        Xóa toàn bộ dữ liệu trong bảng sightings và sources, reset autoincrement ID về 1.
        Vẫn giữ nguyên cấu trúc bảng và các index.
        """
        with self.connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM sightings")
            sightings_count = c.fetchone()[0]

            c.execute("DELETE FROM sightings;")
            sources_count = 0
            if reset_sources:
                c.execute("SELECT COUNT(*) FROM sources")
                sources_count = c.fetchone()[0]
                c.execute("DELETE FROM sources;")
                c.execute("DELETE FROM sqlite_sequence WHERE name IN ('sightings', 'sources');")
            else:
                c.execute("DELETE FROM sqlite_sequence WHERE name = 'sightings';")

        # Thu hồi dung lượng đĩa (VACUUM thực thi ngoài transaction)
        conn_vac = self._get_connection()
        try:
            conn_vac.isolation_level = None
            conn_vac.execute("VACUUM;")
        finally:
            conn_vac.close()

        return {
            "deleted_sightings": sightings_count,
            "deleted_sources": sources_count,
        }


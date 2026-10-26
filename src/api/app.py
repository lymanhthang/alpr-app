"""
FastAPI Backend phục vụ tra cứu lịch sử xe theo biển số, khoảng thời gian và camera.
Tích hợp thuật toán tìm kiếm mờ (Fuzzy matching) với RapidFuzz và sửa lỗi OCR.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from src.config import load_config
from src.reliability.correction import LETTER_TO_DIGIT
from src.storage.db import DatabaseManager

app = FastAPI(
    title="ALPR Vehicle Traceability API",
    description="Hệ thống tra cứu lịch sử xe và đối chiếu kết quả nhận diện biển số",
    version="2.0.0",
)

# Cho phép CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config("config.yaml")
db = DatabaseManager(config.storage.db_path)

# Thư mục tĩnh và tài nguyên
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DATA_DIR = Path("data").resolve()
CROPS_DIR = Path(config.storage.crops_dir).resolve()
CONTEXTS_DIR = Path(config.storage.contexts_dir).resolve()
CLIPS_DIR = Path(config.storage.clips_dir).resolve()


def normalize_ocr_confusions(text: str) -> str:
    """Chuẩn hóa các ký tự dễ bị OCR nhầm lẫn (Z->2, O->0, I->1, B->8, S->5...) để so khớp mờ."""
    if not text:
        return ""
    res = []
    for c in text.upper():
        if c in LETTER_TO_DIGIT:
            res.append(LETTER_TO_DIGIT[c])
        else:
            res.append(c)
    return "".join(res)


def compute_fuzzy_score(query: str, target: str) -> float:
    """Tính điểm tương đồng mờ kết hợp chuẩn hóa sai số OCR."""
    if not query or not target:
        return 0.0
    q = query.strip().upper()
    t = target.strip().upper()
    if q == t:
        return 100.0

    if not RAPIDFUZZ_AVAILABLE:
        # Fallback nếu chưa cài rapidfuzz
        return 100.0 if q in t or t in q else 0.0

    # 1. So sánh tỷ lệ Levenshtein trực tiếp
    score_raw = fuzz.ratio(q, t)

    # 2. So sánh sau khi đã quy đổi các chữ cái nhầm lẫn sang số
    q_norm = normalize_ocr_confusions(q)
    t_norm = normalize_ocr_confusions(t)
    score_norm = fuzz.ratio(q_norm, t_norm)

    return float(max(score_raw, score_norm))


def safe_file_response(path_str: Optional[str], fallback_subfolder: Path) -> FileResponse:
    """Trả về file ảnh/clip an toàn, ngăn chặn tấn công Path Traversal."""
    if not path_str:
        raise HTTPException(status_code=404, detail="Đường dẫn file trống")

    target_path = Path(path_str).resolve()
    # Kiểm tra xem file có nằm trong thư mục data an toàn không
    if not target_path.is_relative_to(DATA_DIR):
        raise HTTPException(status_code=403, detail="Từ chối quyền truy cập file ngoài thư mục data/")

    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file: {target_path.name}")

    # Nhận diện media type
    suffix = target_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(str(target_path), media_type=media_type)


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Phục vụ trang chủ Dashboard tra cứu."""
    index_html = STATIC_DIR / "index.html"
    if index_html.is_file():
        return FileResponse(str(index_html))
    return HTMLResponse("<h2>ALPR API is running. UI index.html not yet built.</h2>")


@app.get("/api/sources")
def get_sources():
    """Lấy danh sách các nguồn (Camera/Video)."""
    return db.get_all_sources()


@app.get("/api/stats")
def get_stats():
    """Lấy số liệu thống kê tổng quan của hệ thống."""
    return db.get_stats()


@app.get("/api/search")
def search_sightings(
    plate: Optional[str] = Query(None, description="Biển số cần tìm (hỗ trợ tìm kiếm mờ)"),
    vehicle_type: Optional[str] = Query(None, description="Lọc theo loại xe (VD: honda-vision, honda-sh...)"),
    fuzzy: bool = Query(False, description="Bật tìm kiếm mờ với RapidFuzz"),
    min_score: float = Query(70.0, ge=0.0, le=100.0, description="Điểm tương đồng tối thiểu khi tìm mờ (0-100)"),
    source_id: Optional[int] = Query(None, description="Lọc theo ID nguồn camera/video"),
    time_from: Optional[float] = Query(None, description="Thời gian bắt đầu (unix timestamp)"),
    time_to: Optional[float] = Query(None, description="Thời gian kết thúc (unix timestamp)"),
    valid_only: Optional[bool] = Query(None, description="Chỉ lấy biển số chuẩn Việt Nam"),
    unknown_only: Optional[bool] = Query(None, description="Chỉ lấy biển số UNKNOWN (không đọc được hoặc sai format VN)"),
    corrected_only: Optional[bool] = Query(None, description="Chỉ lấy các lượt đã được hệ thống sửa ký tự OCR"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    Tra cứu lịch sử xe theo biển số, loại xe, khoảng thời gian và nguồn camera.
    Hỗ trợ tìm kiếm mờ (Fuzzy Search) giúp tìm ra biển số ngay cả khi người dùng gõ nhầm hoặc OCR đọc sai ký tự.
    """
    print(f"[API Search] plate='{plate}', vehicle_type='{vehicle_type}', fuzzy={fuzzy}, valid_only={valid_only}, unknown_only={unknown_only}, corrected_only={corrected_only}")

    # Trường hợp 1: Tìm kiếm mờ theo biển số
    if plate and fuzzy:
        all_plates = db.get_all_plates_distinct()
        matched_plates: List[Tuple[str, float]] = []

        clean_query = plate.strip().upper()
        for p_cand in all_plates:
            score = compute_fuzzy_score(clean_query, p_cand)
            if score >= min_score:
                matched_plates.append((p_cand, score))

        # Sắp xếp theo điểm tương đồng giảm dần
        matched_plates.sort(key=lambda x: x[1], reverse=True)
        top_plate_names = [p for p, _ in matched_plates]

        if not top_plate_names:
            return {"total": 0, "results": []}

        # Lấy sightings thuộc các biển số khớp
        results = []
        for p_name, score in matched_plates:
            items = db.query_sightings(
                plate=p_name,
                vehicle_type=vehicle_type,
                source_id=source_id,
                time_from=time_from,
                time_to=time_to,
                is_valid_only=valid_only,
                is_unknown_only=unknown_only,
                is_corrected_only=corrected_only,
                limit=10,
                offset=0,
            )
            for item in items:
                item["fuzzy_score"] = round(score, 1)
                results.append(item)

        # Phân trang kết quả
        total = len(results)
        paginated = results[offset : offset + limit]
        return {"total": total, "results": paginated}

    # Trường hợp 2: Truy vấn tiêu chuẩn qua SQL Index
    results = db.query_sightings(
        plate=plate,
        vehicle_type=vehicle_type,
        source_id=source_id,
        time_from=time_from,
        time_to=time_to,
        is_valid_only=valid_only,
        is_unknown_only=unknown_only,
        is_corrected_only=corrected_only,
        limit=limit,
        offset=offset,
    )
    for r in results:
        r["fuzzy_score"] = 100.0 if plate else None

    return {"total": len(results), "results": results}



@app.get("/api/sightings/{sighting_id}")
def get_sighting_detail(sighting_id: int):
    """Lấy toàn bộ thông tin chi tiết của một lượt xe bao gồm quỹ đạo và sửa lỗi."""
    item = db.get_sighting_by_id(sighting_id)
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy lượt xe với ID tương ứng")

    # Parse JSON quỹ đạo và chi tiết sửa ký tự
    try:
        item["trajectory"] = json.loads(item.get("trajectory") or "[]")
    except Exception:
        item["trajectory"] = []

    try:
        item["corrections_detail"] = json.loads(item.get("corrections_detail") or "[]")
    except Exception:
        item["corrections_detail"] = []

    return item


@app.get("/api/sightings/{sighting_id}/crop")
def get_sighting_crop(sighting_id: int):
    """Tải ảnh crop biển số phóng to của lượt xe."""
    item = db.get_sighting_by_id(sighting_id)
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy lượt xe")
    return safe_file_response(item.get("best_crop_path"), CROPS_DIR)


@app.get("/api/sightings/{sighting_id}/context")
def get_sighting_context(sighting_id: int):
    """Tải ảnh khung cảnh toàn khung hình của lượt xe."""
    item = db.get_sighting_by_id(sighting_id)
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy lượt xe")
    return safe_file_response(item.get("context_path"), CONTEXTS_DIR)


@app.get("/api/sightings/{sighting_id}/clip")
def get_sighting_clip(sighting_id: int):
    """Tải video clip ngắn của lượt xe (nếu có)."""
    item = db.get_sighting_by_id(sighting_id)
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy lượt xe")
    return safe_file_response(item.get("clip_path"), CLIPS_DIR)

"""
Unit test kiểm tra các endpoint FastAPI tra cứu lịch sử xe.
"""

from fastapi.testclient import TestClient
from src.api.app import app, compute_fuzzy_score

client = TestClient(app)


def test_api_status_and_sources():
    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    data = res_stats.json()
    assert "total_sightings" in data

    res_sources = client.get("/api/sources")
    assert res_sources.status_code == 200
    assert isinstance(res_sources.json(), list)


def test_fuzzy_score_computation():
    # Kiểm tra so khớp mờ với các ký tự OCR dễ nhầm lẫn
    # 51F12345 so với 51F1Z345 (Z nhầm với 2)
    score = compute_fuzzy_score("51F12345", "51F1Z345")
    assert score >= 85.0

    # Trùng tuyệt đối
    assert compute_fuzzy_score("30A99999", "30A99999") == 100.0


def test_search_endpoint():
    res = client.get("/api/search")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert "results" in data

    # Test với param unknown_only
    res_unk = client.get("/api/search?unknown_only=true")
    assert res_unk.status_code == 200
    data_unk = res_unk.json()
    assert "total" in data_unk
    assert "results" in data_unk

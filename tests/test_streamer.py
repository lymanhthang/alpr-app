"""
Unit test kiểm tra tính năng phát luồng Web (WebStreamer) qua HTTP MJPEG.
"""

import time
import urllib.request
import numpy as np
import pytest

from src.streamer import WebStreamer, StreamingOutput, get_local_ip


def test_get_local_ip():
    ip = get_local_ip()
    assert isinstance(ip, str)
    assert len(ip.split(".")) == 4


def test_streaming_output_buffer():
    output = StreamingOutput()
    assert output.client_count == 0
    assert output.frame_bytes is None

    output.add_client()
    assert output.client_count == 1

    dummy_bytes = b"fake_jpeg_data"
    output.set_frame(dummy_bytes)
    assert output.frame_bytes == dummy_bytes

    fetched = output.get_frame(timeout=0.2)
    assert fetched == dummy_bytes

    output.remove_client()
    assert output.client_count == 0


def test_web_streamer_server_endpoints():
    test_port = 8899
    streamer = WebStreamer(
        host="127.0.0.1",
        port=test_port,
        source_name="Test Camera",
        jpeg_quality=75,
    )

    try:
        streamer.start()
        time.sleep(0.3)  # Chờ server lắng nghe

        # Cung cấp 1 frame giả lập
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        streamer.output.add_client()  # Giả lập có client để update_frame thực hiện nén
        streamer.update_frame(dummy_frame, fps=30.0, active_tracks_count=2, total_locked_count=1)
        streamer.output.remove_client()

        base_url = f"http://127.0.0.1:{test_port}"

        # 1. Test GET / (HTML Page)
        with urllib.request.urlopen(f"{base_url}/") as response:
            assert response.status == 200
            content_type = response.headers.get("Content-Type", "")
            assert "text/html" in content_type
            body = response.read().decode("utf-8")
            assert "ALPR Live Stream" in body
            assert "Raspberry Pi 5" in body

        # 2. Test GET /status (JSON API)
        with urllib.request.urlopen(f"{base_url}/status") as response:
            assert response.status == 200
            import json
            data = json.loads(response.read().decode("utf-8"))
            assert data["status"] == "online"
            assert data["source_name"] == "Test Camera"
            assert data["fps"] == 30.0
            assert data["active_tracks"] == 2
            assert data["total_locked"] == 1

        # 3. Test GET /snapshot (JPEG image)
        with urllib.request.urlopen(f"{base_url}/snapshot") as response:
            assert response.status == 200
            content_type = response.headers.get("Content-Type", "")
            assert "image/jpeg" in content_type
            img_bytes = response.read()
            assert len(img_bytes) > 0

    finally:
        streamer.stop()

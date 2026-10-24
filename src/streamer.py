"""
Module phát luồng video trực tiếp qua HTTP (MJPEG Web Streamer).
Tối ưu hóa đặc biệt cho Raspberry Pi 5:
- Hoạt động trên background thread riêng biệt, không chặn pipeline ALPR.
- Chỉ nén JPEG khi có ít nhất 1 client đang xem, tiết kiệm 100% CPU khi không có kết nối.
- Sử dụng thư viện chuẩn của Python (http.server.ThreadingHTTPServer), không cần cài thêm dependency nặng.
- Cung cấp giao diện Web trực tiếp hiện đại (Dark mode, responsive) để xem trên mọi trình duyệt (PC, mobile, tablet).
- Cung cấp endpoint luồng thô (/video_feed) để dễ dàng nhúng vào các hệ thống khác.
"""

from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


def get_local_ip() -> str:
    """Lấy địa chỉ IP nội mạng (LAN) của thiết bị để người dùng dễ truy cập."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Không thực sự gửi gói tin đi, chỉ để socket binding đúng card mạng
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


HTML_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALPR Live Stream - Raspberry Pi 5</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-card: rgba(18, 24, 38, 0.85);
            --bg-card-hover: rgba(26, 34, 52, 0.95);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #00dc82;
            --primary-glow: rgba(0, 220, 130, 0.25);
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.4);
            --accent: #38bdf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 220, 130, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(56, 189, 248, 0.05) 0%, transparent 40%);
        }

        /* Header Bar */
        .stream-header {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand-box {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #041019;
            font-size: 1.25rem;
            box-shadow: 0 0 16px var(--primary-glow);
        }

        .brand-title h1 {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .brand-title span.tag {
            background: rgba(0, 220, 130, 0.15);
            color: var(--primary);
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
            border: 1px solid rgba(0, 220, 130, 0.3);
        }

        .brand-title p {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .live-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(239, 68, 68, 0.12);
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }

        .live-dot {
            width: 8px;
            height: 8px;
            background: var(--danger);
            border-radius: 50%;
            animation: pulse-dot 1.5s infinite ease-in-out;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 8px var(--danger-glow); }
            50% { opacity: 0.4; transform: scale(0.85); box-shadow: none; }
        }

        /* Main Container */
        .main-content {
            flex: 1;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
            padding: 24px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }

        /* Video Player Wrapper */
        .player-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
        }

        .video-container {
            position: relative;
            background: #000;
            width: 100%;
            min-height: 480px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .video-container img {
            width: 100%;
            height: auto;
            max-height: 78vh;
            object-fit: contain;
            display: block;
        }

        /* Stream Overlays */
        .overlay-top-left {
            position: absolute;
            top: 14px;
            left: 14px;
            background: rgba(11, 15, 25, 0.75);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .overlay-top-right {
            position: absolute;
            top: 14px;
            right: 14px;
            background: rgba(11, 15, 25, 0.75);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--primary);
            font-weight: 600;
        }

        /* Control Toolbar */
        .toolbar {
            padding: 14px 20px;
            background: rgba(15, 23, 42, 0.8);
            border-top: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }

        .btn-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .btn {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }

        .btn-primary {
            background: linear-gradient(135deg, rgba(0, 220, 130, 0.2), rgba(56, 189, 248, 0.2));
            border-color: rgba(0, 220, 130, 0.4);
            color: var(--primary);
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, rgba(0, 220, 130, 0.35), rgba(56, 189, 248, 0.35));
            border-color: var(--primary);
            box-shadow: 0 0 12px var(--primary-glow);
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .stat-icon {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }

        .icon-emerald { background: rgba(0, 220, 130, 0.12); color: var(--primary); }
        .icon-sky { background: rgba(56, 189, 248, 0.12); color: var(--accent); }
        .icon-amber { background: rgba(245, 158, 11, 0.12); color: #f59e0b; }
        .icon-purple { background: rgba(168, 85, 247, 0.12); color: #c084fc; }

        .stat-meta .label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat-meta .val {
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 2px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Toast notification */
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: #1e293b;
            color: #fff;
            padding: 10px 18px;
            border-radius: 8px;
            border: 1px solid var(--primary);
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            display: none;
            align-items: center;
            gap: 10px;
            font-size: 0.9rem;
            z-index: 100;
        }
    </style>
</head>
<body>
    <header class="stream-header">
        <div class="brand-box">
            <div class="brand-icon">
                <i class="fa-solid fa-camera"></i>
            </div>
            <div class="brand-title">
                <h1>ALPR Live Stream <span class="tag">Raspberry Pi 5</span></h1>
                <p>Nguồn: <strong id="source-name-display">Đang nạp...</strong></p>
            </div>
        </div>

        <div class="live-badge">
            <div class="live-dot"></div>
            <span>TRỰC TIẾP</span>
        </div>
    </header>

    <main class="main-content">
        <div class="player-card" id="player-card">
            <div class="video-container" id="video-container">
                <img id="stream-img" src="/video_feed" alt="Đang nạp luồng video ALPR...">
                <div class="overlay-top-left">
                    <i class="fa-solid fa-video"></i>
                    <span id="overlay-source">Camera ALPR</span>
                </div>
                <div class="overlay-top-right">
                    <span id="overlay-fps">--.- FPS</span>
                </div>
            </div>

            <div class="toolbar">
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="toggleFullscreen()">
                        <i class="fa-solid fa-expand"></i> Toàn màn hình
                    </button>
                    <button class="btn" onclick="takeSnapshot()">
                        <i class="fa-solid fa-camera-retro"></i> Chụp ảnh
                    </button>
                    <button class="btn" onclick="copyStreamUrl()">
                        <i class="fa-solid fa-link"></i> Sao chép link luồng
                    </button>
                </div>

                <div class="btn-group">
                    <button class="btn" onclick="reloadStream()">
                        <i class="fa-solid fa-rotate-right"></i> Tải lại luồng
                    </button>
                </div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon icon-emerald">
                    <i class="fa-solid fa-gauge-high"></i>
                </div>
                <div class="stat-meta">
                    <div class="label">Tốc độ xử lý</div>
                    <div class="val" id="stat-fps">0.0 FPS</div>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-icon icon-sky">
                    <i class="fa-solid fa-car-side"></i>
                </div>
                <div class="stat-meta">
                    <div class="label">Xe đang theo dõi</div>
                    <div class="val" id="stat-active-tracks">0</div>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-icon icon-amber">
                    <i class="fa-solid fa-id-card"></i>
                </div>
                <div class="stat-meta">
                    <div class="label">Biển số đã chốt</div>
                    <div class="val" id="stat-locked-plates">0</div>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-icon icon-purple">
                    <i class="fa-solid fa-users-viewfinder"></i>
                </div>
                <div class="stat-meta">
                    <div class="label">Người xem trực tiếp</div>
                    <div class="val" id="stat-clients">1</div>
                </div>
            </div>
        </div>
    </main>

    <div class="toast" id="toast">
        <i class="fa-solid fa-check-circle" style="color: var(--primary);"></i>
        <span id="toast-text">Thông báo</span>
    </div>

    <script>
        function showToast(text) {
            const toast = document.getElementById('toast');
            document.getElementById('toast-text').innerText = text;
            toast.style.display = 'flex';
            setTimeout(() => { toast.style.display = 'none'; }, 2500);
        }

        function toggleFullscreen() {
            const elem = document.getElementById('player-card');
            if (!document.fullscreenElement) {
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                } else if (elem.webkitRequestFullscreen) {
                    elem.webkitRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                }
            }
        }

        function takeSnapshot() {
            const link = document.createElement('a');
            link.href = '/snapshot?t=' + Date.now();
            link.download = 'alpr_snapshot_' + new Date().toISOString().replace(/[:.]/g, '-') + '.jpg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            showToast('Đã chụp ảnh khung hình!');
        }

        function copyStreamUrl() {
            const url = window.location.origin + '/video_feed';
            navigator.clipboard.writeText(url).then(() => {
                showToast('Đã sao chép liên kết luồng MJPEG!');
            }).catch(() => {
                showToast('Link luồng: ' + url);
            });
        }

        function reloadStream() {
            const img = document.getElementById('stream-img');
            img.src = '/video_feed?t=' + Date.now();
            showToast('Đã kết nối lại luồng video.');
        }

        // Cập nhật thống kê định kỳ từ /status
        async function fetchStatus() {
            try {
                const res = await fetch('/status');
                if (res.ok) {
                    const data = await res.json();
                    if (data.source_name) {
                        document.getElementById('source-name-display').innerText = data.source_name;
                        document.getElementById('overlay-source').innerText = data.source_name;
                    }
                    if (data.fps !== undefined) {
                        const fpsStr = Number(data.fps).toFixed(1) + ' FPS';
                        document.getElementById('stat-fps').innerText = fpsStr;
                        document.getElementById('overlay-fps').innerText = fpsStr;
                    }
                    if (data.active_tracks !== undefined) {
                        document.getElementById('stat-active-tracks').innerText = data.active_tracks;
                    }
                    if (data.total_locked !== undefined) {
                        document.getElementById('stat-locked-plates').innerText = data.total_locked;
                    }
                    if (data.clients !== undefined) {
                        document.getElementById('stat-clients').innerText = data.clients;
                    }
                }
            } catch (e) {
                // Không log lỗi nếu server đang tạm bận
            }
        }

        setInterval(fetchStatus, 1000);
        fetchStatus();
    </script>
</body>
</html>
"""


class StreamingOutput:
    """Bộ đệm khung hình an toàn đa luồng kèm cơ chế broadcast qua Condition."""

    def __init__(self):
        self.frame_bytes: Optional[bytes] = None
        self.condition = threading.Condition()
        self.client_count: int = 0
        self._count_lock = threading.Lock()

    def add_client(self):
        with self._count_lock:
            self.client_count += 1

    def remove_client(self):
        with self._count_lock:
            self.client_count = max(0, self.client_count - 1)

    def set_frame(self, frame_bytes: bytes):
        """Gán frame mới và đánh thức mọi client đang chờ."""
        with self.condition:
            self.frame_bytes = frame_bytes
            self.condition.notify_all()

    def get_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        """Chờ frame mới với timeout an toàn."""
        with self.condition:
            if self.condition.wait(timeout=timeout):
                return self.frame_bytes
            return self.frame_bytes


class StreamHandler(BaseHTTPRequestHandler):
    """Bộ xử lý HTTP request phục vụ luồng MJPEG và trang Web viewer."""

    server: "CustomStreamingServer"

    def log_message(self, format: str, *args: Any):
        """Tắt các log request 200 GET định kỳ để không làm tràn màn hình terminal."""
        # Chỉ ghi log nếu xảy ra lỗi 4xx, 5xx
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)

    def do_GET(self):
        path = self.path.split("?")[0]

        # 1. Trang Web giao diện người dùng
        if path in ("/", "/index.html"):
            content = HTML_VIEWER_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        # 2. Luồng Video trực tiếp chuẩn MJPEG
        elif path in ("/video_feed", "/stream", "/stream.mjpg"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            self.server.output.add_client()
            try:
                while not self.server.is_stopped():
                    frame_bytes = self.server.output.get_frame(timeout=0.5)
                    if frame_bytes is None:
                        continue

                    # Đóng gói frame JPEG vào luồng multipart
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b"\r\n")
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass
            finally:
                self.server.output.remove_client()

        # 3. Snapshot ảnh JPEG đơn lẻ
        elif path == "/snapshot":
            frame_bytes = self.server.output.frame_bytes
            if frame_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_error(503, "Khung hình chưa sẵn sàng")

        # 4. JSON trạng thái thời gian thực
        elif path == "/status":
            status_data = self.server.get_status()
            content = json.dumps(status_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        else:
            self.send_error(404, "Không tìm thấy đường dẫn")


class CustomStreamingServer(ThreadingHTTPServer):
    """Máy chủ HTTP đa luồng lưu trữ tham chiếu đến bộ đệm và trạng thái."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: Any,
        output: StreamingOutput,
        source_name: str = "Camera ALPR",
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.output = output
        self.source_name = source_name
        self.stopped = False
        self.fps: float = 0.0
        self.active_tracks: int = 0
        self.total_locked: int = 0
        self.start_time = time.time()

    def is_stopped(self) -> bool:
        return self.stopped

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            "source_name": self.source_name,
            "fps": round(self.fps, 1),
            "active_tracks": self.active_tracks,
            "total_locked": self.total_locked,
            "clients": self.output.client_count,
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }


class WebStreamer:
    """
    Quản lý máy chủ phát luồng Web (MJPEG) cho Raspberry Pi 5.
    Hỗ trợ bật/tắt an toàn và tự động tối ưu hóa nén ảnh.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        source_name: str = "Camera ALPR",
        jpeg_quality: int = 80,
    ):
        self.host = host
        self.port = port
        self.source_name = source_name
        self.jpeg_quality = max(10, min(100, int(jpeg_quality)))
        self.output = StreamingOutput()
        self.server: Optional[CustomStreamingServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self._is_running = False

    def start(self):
        """Khởi động máy chủ phát luồng trên một thread daemon riêng biệt."""
        if self._is_running:
            return

        try:
            self.server = CustomStreamingServer(
                (self.host, self.port),
                StreamHandler,
                self.output,
                source_name=self.source_name,
            )
        except OSError as e:
            print(f"\n[Web Streamer Lỗi] Không thể mở cổng {self.port}: {e}")
            print(f"-> Gợi ý: Hãy đổi sang cổng khác bằng tham số --stream-port <port> (ví dụ: --stream-port 8088)\n")
            return

        self._is_running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            name="ALPR_WebStreamer_Thread",
            daemon=True,
        )
        self.server_thread.start()

        local_ip = get_local_ip()
        print("\n" + "=" * 65)
        print("🌐 MÁY CHỦ PHÁT LUỒNG WEB VIDEO (MJPEG STREAMER) ĐÃ SẴN SÀNG:")
        print(f"  -> Truy cập trên Raspberry Pi : http://localhost:{self.port}")
        print(f"  -> Truy cập từ máy tính/điện thoại: http://{local_ip}:{self.port}")
        print(f"  -> Liên kết luồng trực tiếp   : http://{local_ip}:{self.port}/video_feed")
        print("=" * 65 + "\n")

    def _run_server(self):
        if self.server:
            try:
                self.server.serve_forever()
            except Exception:
                pass

    def update_frame(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        active_tracks_count: int = 0,
        total_locked_count: int = 0,
    ):
        """
        Cập nhật khung hình đã xử lý vào luồng.
        TỐI ƯU HÓA: Nếu hiện tại không có client nào mở xem, bỏ qua khâu nén JPEG để tiết kiệm CPU.
        """
        if not self._is_running or self.server is None or frame is None or frame.size == 0:
            return

        # Cập nhật thông số thống kê
        self.server.fps = fps
        self.server.active_tracks = active_tracks_count
        self.server.total_locked = total_locked_count

        # Kiểm tra xem có người đang xem không (hoặc chưa có frame nào)
        if self.output.client_count == 0 and self.output.frame_bytes is not None:
            return

        # Nén ảnh sang JPEG
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if success:
            self.output.set_frame(encoded.tobytes())

    def stop(self):
        """Dừng máy chủ phát luồng và giải phóng socket."""
        if not self._is_running:
            return

        self._is_running = False
        if self.server:
            self.server.stopped = True
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
            print("[Web Streamer] Đã dừng máy chủ phát luồng video.")

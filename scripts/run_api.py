"""
Script khởi chạy máy chủ FastAPI & Giao diện Dashboard tra cứu lịch sử xe.
Cách chạy:
    python scripts/run_api.py
    python scripts/run_api.py --port 8000
"""

import argparse
import sys
from pathlib import Path
import uvicorn

# Đảm bảo đường dẫn gốc project nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Khởi chạy Web API & Dashboard Tra Cứu Biển Số Xe")
    parser.add_argument("--config", type=str, default="config.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--host", type=str, default=None, help="Địa chỉ IP host (mặc định đọc từ config)")
    parser.add_argument("--port", type=int, default=None, help="Cổng port (mặc định đọc từ config: 8000)")
    parser.add_argument("--reload", action="store_true", default=True, help="Tự động nạp lại khi file thay đổi")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    host = args.host or config.api.host or "0.0.0.0"
    port = args.port or config.api.port or 8000

    print("=" * 60)
    print(" KHỞI CHẠY ALPR VEHICLE TRACEABILITY WEB DASHBOARD")
    print("=" * 60)
    print(f" -> Địa chỉ truy cập cục bộ: http://127.0.0.1:{port}")
    print(f" -> Địa chỉ truy cập mạng LAN: http://{host}:{port}")
    print(f" -> CSDL lưu trữ: {config.storage.db_path}")
    print(f" -> Auto reload: {args.reload}")
    print("=" * 60)

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

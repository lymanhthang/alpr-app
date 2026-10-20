"""
Script tiện ích xóa sạch dữ liệu trong cơ sở dữ liệu SQLite nhưng giữ nguyên cấu trúc bảng & index.
Cách chạy:
    python scripts/clean_db.py
    python scripts/clean_db.py -y                # Không cần gõ xác nhận
    python scripts/clean_db.py --keep-sources    # Giữ lại danh sách nguồn Camera/Video
    python scripts/clean_db.py --keep-images     # Chỉ xóa DB, giữ lại file ảnh crop trên đĩa
"""

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.storage.db import DatabaseManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Xóa sạch dữ liệu trong CSDL ALPR (giữ nguyên cấu trúc bảng & index)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Đường dẫn file cấu hình YAML (mặc định: config.yaml)",
    )
    parser.add_argument(
        "--keep-sources",
        action="store_true",
        help="Giữ lại danh sách Camera/Video trong bảng sources (chỉ xóa lượt xe sightings)",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Giữ lại các file ảnh crop/context trên ổ đĩa, chỉ xóa bảng trong SQLite",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Tự động xác nhận xóa mà không cần hỏi lại",
    )
    return parser.parse_args()


def clean_directory_contents(dir_path: Path) -> int:
    """Xóa toàn bộ file và thư mục con bên trong một thư mục, giữ lại thư mục gốc."""
    if not dir_path.is_dir():
        return 0

    deleted_count = 0
    for child in dir_path.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
                deleted_count += 1
            elif child.is_dir():
                shutil.rmtree(child)
                deleted_count += 1
        except Exception as e:
            print(f"  [!] Không thể xóa {child}: {e}")

    return deleted_count


def main():
    args = parse_args()
    config = load_config(args.config)

    db_path = Path(config.storage.db_path)
    crops_dir = Path(config.storage.crops_dir)
    contexts_dir = Path(config.storage.contexts_dir)

    print("=" * 60)
    print(" CÔNG CỤ DỌN DẸP CƠ SỞ DỮ LIỆU ALPR (CLEAN DB)")
    print("=" * 60)
    print(f" -> Đường dẫn CSDL     : {db_path.resolve()}")
    print(f" -> Thư mục ảnh crop   : {crops_dir.resolve()}")
    print(f" -> Thư mục ảnh context: {contexts_dir.resolve()}")
    print(f" -> Xóa bảng sources   : {'KHÔNG' if args.keep_sources else 'CÓ'}")
    print(f" -> Xóa file ảnh đĩa   : {'KHÔNG' if args.keep_images else 'CÓ'}")
    print("=" * 60)

    if not db_path.exists():
        print(f"[!] File CSDL {db_path} không tồn tại. Chưa có dữ liệu nào để xóa.")
        return

    if not args.yes:
        confirm = input("\nBạn có chắc chắn muốn XÓA TOÀN BỘ dữ liệu lượt xe? (y/N): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("[ALPR] Đã hủy thao tác. Dữ liệu vẫn được giữ nguyên.")
            return

    # 1. Xóa dữ liệu trong SQLite
    db = DatabaseManager(str(db_path))
    res = db.clear_all_data(reset_sources=not args.keep_sources)

    print(f"\n[+] Đã xóa {res['deleted_sightings']} bản ghi trong bảng 'sightings'.")
    if not args.keep_sources:
        print(f"[+] Đã xóa {res['deleted_sources']} nguồn trong bảng 'sources'.")
    print("[+] Đã reset bộ đếm AUTOINCREMENT về 1.")
    print("[+] Đã chạy lệnh VACUUM thu hồi dung lượng đĩa thành công.")

    # 2. Xóa các file ảnh crop và context trên đĩa
    if not args.keep_images:
        del_crops = clean_directory_contents(crops_dir)
        del_ctx = clean_directory_contents(contexts_dir)
        print(f"[+] Đã dọn dẹp thư mục ảnh: {del_crops} mục crop, {del_ctx} mục context.")

    print("\n✅ HOÀN TẤT: CSDL đã sạch hoàn toàn nhưng vẫn giữ nguyên đầy đủ cấu trúc bảng và index.")


if __name__ == "__main__":
    main()

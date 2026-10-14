"""
Script kiểm tra độc lập module PaddleOCR ONNX Runtime trên ảnh mẫu.
Hệ thống sẽ tự động tải model PP-OCRv6 về máy trong lần chạy đầu tiên.

Cách chạy:
    python scripts/test_paddleocr.py --image path/to/sample.jpg
"""

import argparse
from pathlib import Path
import re
import sys
import cv2
import numpy as np
from paddleocr import PaddleOCR


def parse_args():
    parser = argparse.ArgumentParser(description="Test PaddleOCR với ONNX Runtime")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Đường dẫn đến file ảnh cần test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)

    if not img_path.is_file():
        print(f"[Lỗi] Không tìm thấy file ảnh: {img_path.resolve()}")
        sys.exit(1)

    det_dir = str(Path("models/ch_det").resolve())
    rec_dir = str(Path("models/ch_rec").resolve())

    det_name = "PP-OCRv6_tiny_det"
    rec_name = "PP-OCRv6_tiny_rec"

    print(f"[Test PaddleOCR] Đang khởi tạo PaddleOCR với model cục bộ:")
    print(f"  - Det: {det_name} ({det_dir})")
    print(f"  - Rec: {rec_name} ({rec_dir})")

    ocr = PaddleOCR(
        text_detection_model_name=det_name,
        text_detection_model_dir=det_dir,
        text_recognition_model_name=rec_name,
        text_recognition_model_dir=rec_dir,
        lang="en",
        engine="onnxruntime",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print(f"\n[Test PaddleOCR] Đang đọc ảnh: {img_path}...")
    img = cv2.imread(str(img_path))
    if img is None:
        print("[Lỗi] cv2 không thể đọc ảnh.")
        sys.exit(1)

    print("[Test PaddleOCR] Đang chạy nhận diện PP-OCRv6 predict...")
    results = ocr.predict(img)

    print("\n" + "=" * 50)
    print("KẾT QUẢ NHẬN DIỆN TỪ PADDLEOCR PP-OCRv6:")
    print("=" * 50)

    found = False
    if results:
        for item in results:
            if item is None:
                continue

            # Format PP-OCRv6 / PaddleOCR 3.x
            texts = item.get("rec_texts", []) if hasattr(item, "get") else getattr(item, "rec_texts", [])
            scores = item.get("rec_scores", []) if hasattr(item, "get") else getattr(item, "rec_scores", [])
            polys = item.get("dt_polys", item.get("rec_polys", [])) if hasattr(item, "get") else getattr(item, "dt_polys", getattr(item, "rec_polys", []))

            for i, t in enumerate(texts):
                clean_t = re.sub(r"[^A-Z0-9]", "", str(t).upper())
                if not clean_t:
                    continue
                sc = scores[i] if i < len(scores) else 1.0
                bx = polys[i] if i < len(polys) else []
                print(f"Text (A-Z, 0-9): {clean_t:<12} (Gốc: {t:<10}) | Confidence: {float(sc):.2f} | BBox: {bx}")
                found = True

    from src.ocr import PlateOCR
    merged = "".join([re.sub(r"[^A-Z0-9]", "", str(t).upper()) for res in (results or []) for t in res.get("rec_texts", [])])
    is_valid = PlateOCR.is_valid_vn_plate(merged)
    print("-" * 50)
    print(f"Biển số ghép được: '{merged}'")
    print(f"Cấu trúc biển số VN hợp lệ: {'[CÓ]' if is_valid else '[KHÔNG]'}")

    if not found:
        print("Không phát hiện được văn bản nào trên ảnh.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

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

    try:
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
    except Exception as e1:
        print(f"[Test PaddleOCR] Thử nạp bằng TextDetection & TextRecognition ({e1})...")
        from paddleocr import TextDetection, TextRecognition
        det_m = TextDetection(model_dir=det_dir, engine="onnxruntime")
        rec_m = TextRecognition(model_dir=rec_dir, engine="onnxruntime")
        ocr = None

    print(f"\n[Test PaddleOCR] Đang đọc ảnh: {img_path}...")
    img = cv2.imread(str(img_path))
    if img is None:
        print("[Lỗi] cv2 không thể đọc ảnh.")
        sys.exit(1)

    print("[Test PaddleOCR] Đang chạy nhận diện...")
    if ocr is not None:
        if hasattr(ocr, "predict"):
            results = ocr.predict(img)
        else:
            results = ocr.ocr(img)
    else:
        det_out = list(det_m.predict(img, batch_size=1))
        results = []
        if det_out:
            polys = det_out[0].get("dt_polys", [])
            h_img, w_img = img.shape[:2]
            rec_texts = []
            rec_scores = []
            rec_polys = []
            for poly in polys:
                box = np.array(poly).astype(int)
                x1, y1 = max(0, np.min(box[:, 0])), max(0, np.min(box[:, 1]))
                x2, y2 = min(w_img, np.max(box[:, 0])), min(h_img, np.max(box[:, 1]))
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    crop = img[y1:y2, x1:x2]
                    rec_out = list(rec_m.predict(crop, batch_size=1))
                    if rec_out:
                        t = rec_out[0].get("rec_text", "")
                        sc = rec_out[0].get("rec_score", 1.0)
                        if t:
                            rec_texts.append(t)
                            rec_scores.append(sc)
                            rec_polys.append(poly)
            results = [{"rec_texts": rec_texts, "rec_scores": rec_scores, "dt_polys": rec_polys}]

    print("\n" + "=" * 50)
    print("KẾT QUẢ NHẬN DIỆN TỪ PADDLEOCR:")
    print("=" * 50)

    found = False
    if results:
        for item in results:
            if item is None:
                continue

            # Format PaddleOCR 3.x
            if hasattr(item, "get") or hasattr(item, "__getitem__"):
                try:
                    texts = item.get("rec_texts", [])
                    scores = item.get("rec_scores", [])
                    polys = item.get("dt_polys", item.get("rec_polys", []))
                    for i, t in enumerate(texts):
                        clean_t = re.sub(r"[^A-Z0-9]", "", str(t).upper())
                        if not clean_t:
                            continue
                        sc = scores[i] if i < len(scores) else 1.0
                        bx = polys[i] if i < len(polys) else []
                        print(f"Text (A-Z, 0-9): {clean_t:<12} (Gốc: {t:<10}) | Confidence: {float(sc):.2f} | BBox: {bx}")
                        found = True
                    continue
                except Exception:
                    pass

            # Format PaddleOCR 2.x
            if isinstance(item, (list, tuple)):
                for line in item:
                    if not line or len(line) < 2:
                        continue
                    bbox = line[0]
                    text, score = line[1]
                    clean_t = re.sub(r"[^A-Z0-9]", "", str(text).upper())
                    if not clean_t:
                        continue
                    print(f"Text (A-Z, 0-9): {clean_t:<12} (Gốc: {text:<10}) | Confidence: {float(score):.2f} | BBox: {bbox}")
                    found = True

    if not found:
        print("Không phát hiện được văn bản nào trên ảnh.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

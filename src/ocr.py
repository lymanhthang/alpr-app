"""
Module nhận diện ký tự biển số sử dụng PaddleOCR với backend ONNX Runtime.
Hỗ trợ tự động tải model PP-OCRv6 qua mạng hoặc chỉ định đường dẫn model cục bộ.
"""

import time
from pathlib import Path
import re
from typing import List, Optional, Tuple
import numpy as np
from paddleocr import PaddleOCR
from src.config import OCRConfig


class PlateOCR:
    """Wrapper cho PaddleOCR ONNX Runtime chuyên dụng nhận diện biển số."""

    def __init__(self, config: OCRConfig):
        self.config = config
        det_dir = str(Path(config.det_model_dir or "models/ch_det").resolve())
        rec_dir = str(Path(config.rec_model_dir or "models/ch_rec").resolve())

        det_name = config.det_model_name or "PP-OCRv6_tiny_det"
        rec_name = config.rec_model_name or "PP-OCRv6_tiny_rec"

        print(
            f"[OCR] Nạp model PP-OCRv6 cục bộ từ thư mục models/:\n"
            f"  - Det: {det_name} ({det_dir})\n"
            f"  - Rec: {rec_name} ({rec_dir})\n"
            f"  - Engine: {config.engine}"
        )

        # Cấu hình nạp model theo chuẩn PP-OCRv6 / PaddleOCR 3.x
        ocr_kwargs = {
            "text_detection_model_name": det_name,
            "text_detection_model_dir": det_dir,
            "text_recognition_model_name": rec_name,
            "text_recognition_model_dir": rec_dir,
            "engine": config.engine or "onnxruntime",
            "lang": config.lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }

        self.ocr = None
        t0 = time.perf_counter()
        try:
            self.ocr = PaddleOCR(**ocr_kwargs)
            print("[OCR] Khởi tạo PP-OCRv6 (PaddleOCR 3.x) thành công.")
        except Exception as e:
            print(f"[OCR Warning] Lỗi nạp model cục bộ ({e}), thử tải cấu hình mặc định PP-OCRv6...")
            self.ocr = PaddleOCR(
                text_detection_model_name=det_name,
                text_recognition_model_name=rec_name,
                engine=config.engine or "onnxruntime",
                lang=config.lang,
                use_doc_orientation_classify= False,
                use_doc_unwarping= False,
                use_textline_orientation= False,
            )

        self.load_time_ms = (time.perf_counter() - t0) * 1000
        print(f"[OCR] Nạp mô hình OCR hoàn tất trong {self.load_time_ms:.2f} ms.")

    def warmup(self):
        """Khởi động (warm-up) mô hình PP-OCRv6 ONNX Runtime với dummy input."""
        try:
            dummy = np.zeros((48, 160, 3), dtype=np.uint8)
            if self.ocr is not None:
                _ = list(self.ocr.predict(dummy))
        except Exception:
            pass

    def recognize_raw(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Nhận diện ký tự thô từ ảnh crop mà không áp đặt lọc độ dài hay regex.
        Phục vụ cho các bộ xử lý độ tin cậy và bỏ phiếu đa frame.
        """
        if plate_crop is None or plate_crop.size == 0 or self.ocr is None:
            return None, 0.0

        try:
            results = self.ocr.predict(plate_crop)
            all_texts, all_scores = [], []

            for res in (results or []):
                for text, score in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
                    clean = re.sub(r"[^A-Z0-9]", "", str(text).upper())
                    if clean and score >= self.config.min_confidence:
                        all_texts.append(clean)
                        all_scores.append(float(score))

            final_text = "".join(all_texts)
            if final_text:
                return final_text, float(np.mean(all_scores)) if all_scores else 0.0
        except Exception as e:
            print(f"[OCR Warning] {e}")

        return None, 0.0

    def recognize_plate(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Nhận diện ký tự trên ảnh cắt biển số bằng PP-OCRv6 (PaddleOCR 3.x).
        Rút gọn tinh gọn: ghép trực tiếp các cụm chữ theo thứ tự tự nhiên của PaddleOCR.
        """
        raw_text, score = self.recognize_raw(plate_crop)
        if raw_text and len(raw_text) >= 5:
            return raw_text, score
        return None, 0.0


    @staticmethod
    def is_valid_vn_plate(plate: str) -> bool:
        """
        Kiểm tra cấu trúc biển số xe Việt Nam hợp lệ (7 - 9 ký tự):
        - 2 ký tự đầu: mã tỉnh thành (11-99)
        - Ký tự thứ 3: chữ cái series (A-Z)
        - Ký tự thứ 4: chữ cái hoặc chữ số tùy chọn (A-Z, 0-9)
        - 4 hoặc 5 ký tự cuối: dãy số đăng ký (0-9)
        Ví dụ: 30A12345, 29X112345, 51F1234, 29LD12345
        """
        if not plate:
            return False
        pattern = r"^(?:1[1-9]|[2-9][0-9])[A-Z][A-Z0-9]?[0-9]{4,5}$"
        return bool(re.match(pattern, plate))


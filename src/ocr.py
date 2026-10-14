"""
Module nhận diện ký tự biển số sử dụng PaddleOCR với backend ONNX Runtime.
Hỗ trợ tự động tải model PP-OCRv6 qua mạng hoặc chỉ định đường dẫn model cục bộ.
"""

import time
from pathlib import Path
import re
from typing import List, Optional, Tuple
import cv2
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
            f"[OCR] Nạp model cục bộ từ thư mục models/:\n"
            f"  - Det: {det_name} ({det_dir})\n"
            f"  - Rec: {rec_name} ({rec_dir})\n"
            f"  - Engine: {config.engine}"
        )

        # Cấu hình nạp model cục bộ cho PaddleOCR 3.x
        ocr_kwargs_v3 = {
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
        self.det_model = None
        self.rec_model = None

        t0 = time.perf_counter()
        try:
            self.ocr = PaddleOCR(**ocr_kwargs_v3)
            print("[OCR] Khởi tạo PaddleOCR 3.x với model cục bộ thành công.")
        except Exception as e1:
            print(f"[OCR Info] Thử nạp theo chuẩn PaddleOCR v2.x ({e1})...")
            ocr_kwargs_v2 = {
                "det_model_dir": det_dir,
                "rec_model_dir": rec_dir,
                "engine": config.engine or "onnxruntime",
                "lang": config.lang,
            }
            try:
                self.ocr = PaddleOCR(**ocr_kwargs_v2)
                print("[OCR] Khởi tạo PaddleOCR 2.x với model cục bộ thành công.")
            except Exception as e2:
                print(f"[OCR Info] Thử nạp bằng TextDetection & TextRecognition riêng lẻ ({e2})...")
                try:
                    from paddleocr import TextDetection, TextRecognition
                    self.det_model = TextDetection(model_dir=det_dir, engine=config.engine or "onnxruntime")
                    self.rec_model = TextRecognition(model_dir=rec_dir, engine=config.engine or "onnxruntime")
                    print("[OCR] Khởi tạo TextDetection & TextRecognition cục bộ thành công.")
                except Exception as e3:
                    print(f"[OCR Warning] Không nạp được model cục bộ ({e3}), fallback sang model online...")
                    self.ocr = PaddleOCR(lang=config.lang)

        self.load_time_ms = (time.perf_counter() - t0) * 1000
        print(f"[OCR] Nạp mô hình OCR thành công trong {self.load_time_ms:.2f} ms.")

    def warmup(self):
        """Khởi động (warm-up) mô hình PaddleOCR ONNX Runtime với dummy input."""
        try:
            dummy = np.zeros((48, 160, 3), dtype=np.uint8)
            if self.ocr is not None:
                if hasattr(self.ocr, "predict"):
                    _ = list(self.ocr.predict(dummy))
                else:
                    _ = self.ocr.ocr(dummy)
            elif self.det_model is not None and self.rec_model is not None:
                _ = list(self.det_model.predict(dummy, batch_size=1))
                _ = list(self.rec_model.predict(dummy, batch_size=1))
        except Exception:
            pass

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh crop biển số trước khi đưa vào OCR:
        - Phóng to nếu ảnh quá nhỏ để model det bắt chữ rõ hơn.
        """
        if crop is None or crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        # Nếu chiều cao quá nhỏ (< 48px), resize upscale với phép nội suy CUBIC
        if h < 48:
            scale = 48.0 / h
            crop = cv2.resize(
                crop, (int(w * scale), 48), interpolation=cv2.INTER_CUBIC
            )

        return crop

    def recognize_plate(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Nhận diện ký tự trên ảnh cắt của biển số.
        Hỗ trợ cả format kết quả của PaddleOCR 3.x (PaddleX) và 2.x.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None, 0.0

        processed = self.preprocess_crop(plate_crop)
        h_crop, w_crop = processed.shape[:2]

        if self.ocr is not None:
            try:
                # Ưu tiên dùng predict() trên PaddleOCR v3.x, fallback sang ocr()
                if hasattr(self.ocr, "predict"):
                    results = self.ocr.predict(processed)
                else:
                    results = self.ocr.ocr(processed)
            except Exception as e:
                try:
                    results = self.ocr.ocr(processed)
                except Exception as e2:
                    print(f"[OCR Warning] Lỗi khi chạy PaddleOCR: {e2}")
                    return None, 0.0
        elif self.det_model is not None and self.rec_model is not None:
            try:
                det_out = list(self.det_model.predict(processed, batch_size=1))
                results = []
                if det_out:
                    polys = det_out[0].get("dt_polys", [])
                    rec_texts, rec_scores, rec_polys = [], [], []
                    for poly in polys:
                        box = np.array(poly).astype(int)
                        x1, y1 = max(0, np.min(box[:, 0])), max(0, np.min(box[:, 1]))
                        x2, y2 = min(w_crop, np.max(box[:, 0])), min(h_crop, np.max(box[:, 1]))
                        if (x2 - x1) > 5 and (y2 - y1) > 5:
                            crop = processed[y1:y2, x1:x2]
                            rec_out = list(self.rec_model.predict(crop, batch_size=1))
                            if rec_out:
                                t = rec_out[0].get("rec_text", "")
                                sc = rec_out[0].get("rec_score", 1.0)
                                if t:
                                    clean_t = self._clean_plate_text(t)
                                    if clean_t:
                                        rec_texts.append(clean_t)
                                        rec_scores.append(sc)
                                        rec_polys.append(poly)
                    results = [{"rec_texts": rec_texts, "rec_scores": rec_scores, "dt_polys": rec_polys}]
            except Exception as e:
                print(f"[OCR Warning] Lỗi khi chạy TextDetection/TextRecognition: {e}")
                return None, 0.0
        else:
            return None, 0.0

        if not results:
            return None, 0.0

        lines_info: List[Tuple[float, float, str, float]] = []

        # Xử lý kết quả trả về từ PaddleOCR v3.x hoặc v2.x
        for item in results:
            if item is None:
                continue

            # Trường hợp 1: PaddleOCR v3.x (Object/dict chứa rec_texts, dt_polys, rec_scores)
            if hasattr(item, "get") or hasattr(item, "__getitem__"):
                try:
                    texts = item.get("rec_texts", [])
                    scores = item.get("rec_scores", [])
                    polys = item.get("dt_polys", item.get("rec_polys", []))

                    if texts and len(texts) > 0:
                        for idx, text in enumerate(texts):
                            score = float(scores[idx]) if idx < len(scores) else 1.0
                            if score < self.config.min_confidence:
                                continue
                            clean_text = self._clean_plate_text(text)
                            if not clean_text:
                                continue

                            if idx < len(polys) and polys[idx] is not None:
                                box_np = np.array(polys[idx])
                                center_y = float(np.mean(box_np[:, 1]))
                                center_x = float(np.mean(box_np[:, 0]))
                            else:
                                center_y, center_x = 0.0, float(idx)

                            lines_info.append((center_y, center_x, clean_text, score))
                        continue
                except Exception:
                    pass

            # Trường hợp 2: PaddleOCR v2.x (list các [box, (text, score)])
            if isinstance(item, (list, tuple)):
                for line in item:
                    if not line or len(line) < 2:
                        continue
                    box_points = line[0]
                    text, score = line[1]
                    score = float(score)
                    if score < self.config.min_confidence:
                        continue
                    clean_text = self._clean_plate_text(text)
                    if not clean_text:
                        continue
                    box_np = np.array(box_points)
                    center_y = float(np.mean(box_np[:, 1]))
                    center_x = float(np.mean(box_np[:, 0]))
                    lines_info.append((center_y, center_x, clean_text, score))

        if not lines_info:
            return None, 0.0

        sorted_text, avg_conf = self._order_and_merge_lines(lines_info, h_crop)
        return sorted_text, avg_conf

    def _clean_plate_text(self, text: str) -> str:
        """
        Bộ lọc nghiêm ngặt cho biển số xe:
        Chỉ chấp nhận các ký tự in hoa từ A-Z và chữ số từ 0-9.
        Loại bỏ hoàn toàn dấu gạch ngang, dấu chấm, khoảng trắng và ký tự đặc biệt.
        """
        if not text:
            return ""
        # 1. Chuyển thành in hoa
        text = str(text).upper()
        # 2. Lọc chỉ giữ lại [A-Z0-9]
        cleaned = re.sub(r"[^A-Z0-9]", "", text)
        return cleaned

    def _order_and_merge_lines(
        self,
        lines_info: List[Tuple[float, float, str, float]],
        crop_height: int,
    ) -> Tuple[Optional[str], float]:
        """
        Ghép các dòng chữ hợp lý cho biển số Việt Nam (biển dài 1 dòng hoặc vuông 2 dòng).
        Kết quả cuối cùng chỉ chứa các ký tự [A-Z0-9].
        """
        if not lines_info:
            return None, 0.0

        # Nếu chỉ có 1 box duy nhất
        if len(lines_info) == 1:
            clean = self._clean_plate_text(lines_info[0][2])
            return (clean, lines_info[0][3]) if len(clean) >= 3 else (None, 0.0)

        # Phân loại thành dòng trên / dòng dưới dựa vào độ chênh lệch Y
        y_threshold = max(10.0, crop_height * 0.25)

        # Sắp xếp thô theo Y
        lines_info.sort(key=lambda item: item[0])

        top_line: List[Tuple[float, float, str, float]] = []
        bottom_line: List[Tuple[float, float, str, float]] = []

        first_y = lines_info[0][0]
        for item in lines_info:
            y, x, text, score = item
            clean_t = self._clean_plate_text(text)
            if not clean_t:
                continue
            item_clean = (y, x, clean_t, score)

            if abs(y - first_y) < y_threshold:
                top_line.append(item_clean)
            else:
                bottom_line.append(item_clean)

        # Sắp xếp từ trái qua phải (theo X) cho mỗi dòng
        top_line.sort(key=lambda item: item[1])
        bottom_line.sort(key=lambda item: item[1])

        top_text = "".join([item[2] for item in top_line])
        bottom_text = "".join([item[2] for item in bottom_line])

        all_scores = [item[3] for item in (top_line + bottom_line)]
        avg_score = float(np.mean(all_scores)) if all_scores else 0.0

        # Ghép chuỗi thuần ký tự in hoa và số (không dùng dấu gạch nối)
        final_text = f"{top_text}{bottom_text}"
        final_text = self._clean_plate_text(final_text)

        # Loại bỏ nếu kết quả quá ngắn (< 3 ký tự) do nhiễu
        if len(final_text) < 3:
            return None, 0.0

        return final_text, avg_score

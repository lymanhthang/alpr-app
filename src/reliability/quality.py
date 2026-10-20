"""
Module đánh giá chất lượng ảnh crop biển số xe.
Điểm chất lượng được chuẩn hóa tự động trong khoảng [0.0, 1.0] mà không dùng hằng số ngưỡng cứng.
"""

from dataclasses import dataclass
import math
from typing import Tuple
import cv2
import numpy as np


@dataclass
class QualityResult:
    score: float
    size_score: float
    sharp_score: float
    det_conf: float
    edge_score: float

    def __float__(self) -> float:
        return self.score


def calculate_crop_quality(
    crop: np.ndarray,
    bbox: np.ndarray,
    frame_shape: Tuple[int, int],
    det_conf: float = 1.0,
) -> QualityResult:
    """
    Tính điểm chất lượng tổng hợp cho vùng crop biển số.

    Args:
        crop: Mảng ảnh BGR của vùng crop biển số.
        bbox: Tọa độ [x1, y1, x2, y2] trên frame gốc.
        frame_shape: (height, width) của frame gốc.
        det_conf: Độ tin cậy từ mô hình phát hiện (0.0 - 1.0).

    Returns:
        QualityResult: Chứa điểm tổng hợp và các thành phần điểm.
    """
    if crop is None or crop.size == 0 or len(bbox) < 4:
        return QualityResult(0.0, 0.0, 0.0, 0.0, 0.0)

    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))

    # 1. Điểm kích thước (Size score): Tỷ lệ diện tích crop so với toàn bộ frame.
    # Sử dụng hàm bão hòa mũ tự nhiên: area_ratio ~ 1.5% - 3% là kích thước lý tưởng cho biển số.
    frame_area = max(1.0, float(frame_w * frame_h))
    area_ratio = (bw * bh) / frame_area
    size_score = float(1.0 - math.exp(-area_ratio / 0.015))
    size_score = max(0.0, min(1.0, size_score))

    # 2. Điểm độ nét (Sharpness score): Phương sai toán tử Laplace.
    # Chuẩn hóa mềm dẻo qua hàm Rational Saturating: var / (var + 50.0)
    # Ảnh mờ (var < 20) -> score < 0.28; ảnh rõ (var > 150) -> score > 0.75; ảnh rất nét -> tiệm cận 1.0.
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Chuẩn hóa chiều cao về 48px để đo độ nét đồng nhất, không phụ thuộc độ phân giải crop
    target_h = 48
    if gray.shape[0] > 0 and gray.shape[1] > 0 and gray.shape[0] != target_h:
        target_w = max(16, int(gray.shape[1] * (target_h / gray.shape[0])))
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_score = float(lap_var / (lap_var + 50.0)) if (lap_var + 50.0) > 0 else 0.0
    sharp_score = max(0.0, min(1.0, sharp_score))

    # 3. Điểm khoảng cách biên (Edge score): Tránh trường hợp biển số bị cắt xén một phần ở rìa khung hình.
    # dist_to_edge tính khoảng cách từ bbox tới 4 cạnh của khung hình.
    dist_left = max(0.0, float(x1))
    dist_top = max(0.0, float(y1))
    dist_right = max(0.0, float(frame_w - x2))
    dist_bottom = max(0.0, float(frame_h - y2))
    min_dist = min(dist_left, dist_top, dist_right, dist_bottom)

    # Nếu bbox chạm sát biên (dưới 3px), điểm biên = 0; xa biên > 15% kích thước biển thì đạt 1.0.
    safe_margin = max(4.0, 0.15 * min(bw, bh))
    edge_score = float(min(1.0, max(0.0, (min_dist - 2.0) / safe_margin)))

    # 4. Độ tin cậy từ Detector
    det_conf_score = float(max(0.0, min(1.0, det_conf)))

    # Tổng hợp trọng số: Size (30%) + Sharpness (30%) + DetConf (25%) + Edge (15%)
    total_score = (
        0.30 * size_score
        + 0.30 * sharp_score
        + 0.25 * det_conf_score
        + 0.15 * edge_score
    )
    total_score = max(0.0, min(1.0, total_score))

    return QualityResult(
        score=total_score,
        size_score=size_score,
        sharp_score=sharp_score,
        det_conf=det_conf_score,
        edge_score=edge_score,
    )

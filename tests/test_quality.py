"""
Unit test kiểm tra tính năng đo chất lượng crop tự chuẩn hóa.
"""

import cv2
import numpy as np
from src.reliability.quality import calculate_crop_quality


def test_quality_range_and_sharpness():
    # Tạo ảnh nét (đường kẻ đen trắng đan xen)
    sharp_crop = np.zeros((48, 160, 3), dtype=np.uint8)
    sharp_crop[:, ::4] = 255

    # Tạo ảnh mờ (gaussian blur cực mạnh)
    blur_crop = cv2.GaussianBlur(sharp_crop, (21, 21), 0)

    bbox = np.array([50, 50, 210, 98], dtype=np.float32)
    frame_shape = (720, 1280)

    q_sharp = calculate_crop_quality(sharp_crop, bbox, frame_shape, det_conf=0.9)
    q_blur = calculate_crop_quality(blur_crop, bbox, frame_shape, det_conf=0.9)

    # Đảm bảo điểm chất lượng luôn nằm trong [0.0, 1.0]
    assert 0.0 <= q_sharp.score <= 1.0
    assert 0.0 <= q_blur.score <= 1.0

    # Ảnh nét phải có điểm độ nét cao hơn ảnh mờ
    assert q_sharp.sharp_score > q_blur.sharp_score
    assert q_sharp.score > q_blur.score


def test_quality_edge_penalty():
    # Crop tại giữa khung hình
    crop = np.full((48, 160, 3), 128, dtype=np.uint8)
    frame_shape = (720, 1280)

    center_bbox = np.array([400, 300, 560, 348], dtype=np.float32)
    edge_bbox = np.array([0, 0, 160, 48], dtype=np.float32)  # Dính sát mép trên-trái

    q_center = calculate_crop_quality(crop, center_bbox, frame_shape, det_conf=0.8)
    q_edge = calculate_crop_quality(crop, edge_bbox, frame_shape, det_conf=0.8)

    assert q_center.edge_score > q_edge.edge_score
    assert q_edge.edge_score == 0.0  # Dính biên x=0, y=0

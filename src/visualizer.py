"""
Module vẽ bounding box, track ID, text biển số và thông số FPS lên frame.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.ocr import PlateOCR


class Visualizer:
    """Bộ trực quan hóa kết quả nhận diện và theo dõi biển số."""

    def __init__(self):
        # Màu sắc chủ đạo: Xanh ngọc (Emerald / Cyan) cho bounding box biển số
        self.box_color = (0, 220, 100)       # BGR
        self.badge_bg_color = (0, 160, 70)   # BGR
        self.text_color = (255, 255, 255)    # Trắng
        self.pending_color = (0, 165, 255)   # Cam (khi chưa đọc xong OCR)
        # Màu sắc cho thân xe (Tím lam / Violet dịu mắt)
        self.vehicle_box_color = (220, 120, 160)   # BGR
        self.vehicle_badge_bg = (170, 70, 120)     # BGR

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[Tuple[int, np.ndarray, float]],
        ocr_cache: Dict[int, str],
        provisional_cache: Optional[Dict[int, str]] = None,
        vehicle_tracks: Optional[List[Tuple[int, np.ndarray, float, int, str]]] = None,
        vehicle_type_map: Optional[Dict[int, str]] = None,
        draw_vehicle_box: bool = True,
        fps: float = 0.0,
        ocr_calls: int = 0,
    ) -> np.ndarray:
        """
        Vẽ thông tin biển số và loại xe lên frame ảnh.
        Biển số đã chốt hợp lệ hiển thị màu xanh lá; biển số tạm thời/đang chờ hiển thị màu cam.

        Args:
            frame: Ảnh BGR gốc.
            tracks: Danh sách track biển số (track_id, xyxy, conf).
            ocr_cache: Bảng tra {track_id: plate_text} đã chốt.
            provisional_cache: Bảng tra {track_id: plate_text} tạm thời trong lúc chờ bỏ phiếu.
            vehicle_tracks: Danh sách track thân xe (v_track_id, xyxy, conf, class_id, class_name).
            vehicle_type_map: Bảng tra {plate_track_id: vehicle_type} đã gắn kết.
            draw_vehicle_box: Có vẽ bounding box thân xe không.
            fps: Tốc độ xử lý khung hình / giây.
            ocr_calls: Tổng số lần đã gọi OCR thực tế.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        prov = provisional_cache or {}
        v_map = vehicle_type_map or {}

        # 1. Vẽ bounding box thân xe nếu được bật
        if draw_vehicle_box and vehicle_tracks:
            for v_tid, v_box, v_conf, v_cls_id, v_cls_name in vehicle_tracks:
                vx1, vy1, vx2, vy2 = [int(v) for v in v_box]
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(w - 1, vx2), min(h - 1, vy2)

                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), self.vehicle_box_color, thickness=2)
                v_label = f"{v_cls_name} ({int(v_conf * 100)}%)"
                self._draw_label_badge(annotated, v_label, vx1, vy1, self.vehicle_badge_bg)

        # 2. Vẽ bounding box và badge biển số (kèm loại xe)
        for track_id, box, conf in tracks:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            plate_text = ocr_cache.get(track_id)
            is_valid = plate_text is not None and PlateOCR.is_valid_vn_plate(plate_text)
            v_type_str = f" | {v_map[track_id]}" if (track_id in v_map and v_map[track_id]) else ""

            if is_valid:
                color = self.box_color
                label = f"#{track_id}: {plate_text}{v_type_str}"
            elif track_id in prov and prov[track_id]:
                color = self.pending_color
                label = f"#{track_id}: {prov[track_id]}?{v_type_str}"
            else:
                color = self.pending_color
                label = f"#{track_id}: UNKNOWN{v_type_str}"

            # Vẽ bounding box biển số
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

            # Vẽ badge nền chứa nhãn biển số + loại xe
            self._draw_label_badge(annotated, label, x1, y1, color)

        # 3. Vẽ bảng thống kê thông tin ở góc trên bên trái
        valid_count = sum(
            1 for text in ocr_cache.values() if PlateOCR.is_valid_vn_plate(text)
        )
        self._draw_info_panel(annotated, fps, len(tracks), valid_count, ocr_calls)

        return annotated

    def _draw_label_badge(
        self,
        img: np.ndarray,
        label: str,
        x: int,
        y: int,
        bg_color: Tuple[int, int, int],
    ):
        """Vẽ nhãn hiển thị phía trên bounding box."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        badge_y1 = max(0, y - label_h - 10)
        badge_y2 = y
        badge_x1 = x
        badge_x2 = min(img.shape[1] - 1, x + label_w + 10)

        # Vẽ hình chữ nhật nền
        cv2.rectangle(
            img,
            (badge_x1, badge_y1),
            (badge_x2, badge_y2),
            bg_color,
            thickness=-1,
        )

        # Vẽ text
        cv2.putText(
            img,
            label,
            (badge_x1 + 5, badge_y2 - 4),
            font,
            font_scale,
            self.text_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_info_panel(
        self,
        img: np.ndarray,
        fps: float,
        active_tracks: int,
        total_cached: int,
        ocr_calls: int,
    ):
        """Vẽ panel hiển thị FPS và trạng thái hệ thống."""
        panel_text = (
            f"FPS: {fps:.1f} | Active: {active_tracks} | "
            f"Valid Plates: {total_cached} | OCR Calls: {ocr_calls}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), _ = cv2.getTextSize(panel_text, font, font_scale, thickness)

        # Tạo vùng nền tối mờ
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (tw + 24, th + 22), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Viết text
        cv2.putText(
            img,
            panel_text,
            (16, th + 18),
            font,
            font_scale,
            (0, 255, 200),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

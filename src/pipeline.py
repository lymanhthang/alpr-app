"""
Pipeline điều phối toàn bộ luồng ALPR:
Frame -> Skip-frame / YOLO NCNN -> ByteTrack -> Độ tin cậy (Quality + Voting + Correction) -> Lưu CSDL SQLite -> Visualization
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from src.clip_buffer import ClipRingBuffer
from src.config import AppConfig
from src.detector import PlateDetector
from src.ocr import PlateOCR
from src.reliability.correction import correct_vn_plate
from src.reliability.quality import calculate_crop_quality
from src.reliability.track_state import FinalizedSighting, TrackStateManager
from src.storage.writer import SightingWriter
from src.tracker import PlateTracker
from src.vehicle_classifier import VehicleClassifier, VehicleTracker
from src.visualizer import Visualizer


class ALPRPipeline:
    """Hệ thống ALPR hoàn chỉnh cho video, webcam và ảnh với cơ chế lưu trữ và tăng độ tin cậy."""

    def __init__(
        self,
        config: AppConfig,
        benchmark: bool = False,
        source_id: int = 1,
        fps_in: float = 25.0,
        start_time: Optional[float] = None,
        no_store: bool = False,
    ):
        self.config = config
        self.benchmark = benchmark
        self.source_id = source_id
        self.fps_in = fps_in if fps_in and fps_in > 0 else 25.0
        self.start_time = start_time if start_time is not None else time.time()
        self.strategy = getattr(config.ocr, "strategy", "voting")

        print("\n" + "=" * 50)
        print("KHỞI TẠO HỆ THỐNG ALPR CHO RASPBERRY PI 5")
        print(f"Chiến lược OCR: {self.strategy.upper()}")
        print("=" * 50)

        # 1. Khởi tạo các module thị giác
        self.detector = PlateDetector(config.detector)
        self.tracker = PlateTracker(config.tracker)
        self.ocr = PlateOCR(config.ocr)
        self.visualizer = Visualizer()

        # Module phân loại loại xe 2 bánh (YOLO NCNN + ByteTrack)
        self.vehicle_classifier: Optional[VehicleClassifier] = None
        self.vehicle_tracker: Optional[VehicleTracker] = None
        self.vc_times: List[float] = []
        self.vehicle_active_tracks: List[Tuple[int, np.ndarray, float, int, str]] = []

        if getattr(config, "vehicle_classifier", None) and config.vehicle_classifier.enabled:
            print("[Pipeline] Kích hoạt module Phân Loại Xe 2 Bánh (YOLO NCNN)...")
            self.vehicle_classifier = VehicleClassifier(config.vehicle_classifier)
            self.vehicle_tracker = VehicleTracker(config.tracker)

        # 2. Khởi tạo module độ tin cậy và lưu trữ
        self.track_manager = TrackStateManager(config.tracker, config.ocr)

        # Tắt lưu trữ nếu người dùng truyền --no-store
        if no_store:
            config.storage.enabled = False

        self.storage_writer = SightingWriter(config.storage)
        self.clip_buffer = ClipRingBuffer(config.clip, fps=self.fps_in)

        # 3. Warm-up các mô hình với dummy input
        print("[Pipeline] Đang warm-up mô hình YOLO NCNN & PaddleOCR...")
        t_warmup = time.perf_counter()
        self.detector.warmup(imgsz=config.detector.imgsz)
        self.ocr.warmup()
        if self.vehicle_classifier:
            self.vehicle_classifier.warmup(imgsz=config.vehicle_classifier.imgsz)
        self.warmup_time_ms = (time.perf_counter() - t_warmup) * 1000
        print(f"[Pipeline] Warm-up hoàn tất trong {self.warmup_time_ms:.2f} ms.")

        # 4. Quản lý trạng thái và bộ nhớ đệm OCR
        self.ocr_cache: Dict[int, str] = {}           # {track_id: plate_text} đã chốt
        self.provisional_cache: Dict[int, str] = {}   # {track_id: plate_text} tạm thời (đang voting)
        self.ocr_calls_count: int = 0
        self.skip_frames: int = max(1, config.tracker.skip_frames)
        self.lock_first_last_frame: Dict[int, int] = {}  # Hỗ trợ riêng cho lock_first

        # 5. Thống kê benchmark
        self.det_times: List[float] = []
        self.ocr_times: List[float] = []
        self.pipeline_times: List[float] = []
        self.early_locks_count: int = 0
        self.track_end_locks_count: int = 0

        # 6. Tính toán FPS — sliding window + EMA kết hợp
        # _frame_ts: timestamp chính xác ngay đầu process_frame (trước mọi logic)
        self._frame_ts: float = 0.0          # timestamp frame trước
        self.fps: float = 0.0                # FPS EMA hiển thị
        # Sliding window lưu N khoảng thời gian gần nhất để warmup nhanh
        self._fps_window: List[float] = []   # danh sách các dt gần nhất (giây)
        self._FPS_WINDOW_SIZE: int = 30      # độ dài cửa sổ

        print(f"[Pipeline] Sẵn sàng. Tỷ lệ Skip-frame: mỗi {self.skip_frames} frame.")
        if self.benchmark:
            print("[Pipeline] Chế độ Benchmark CHI TIẾT: ĐÃ BẬT.")
        print("=" * 50 + "\n")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        is_live: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray, float]], Dict[int, str]]:
        """
        Xử lý 1 khung hình qua toàn bộ pipeline.

        Args:
            frame: Ảnh BGR.
            frame_idx: Chỉ số frame hiện tại.
            is_live: True nếu nguồn là webcam/camera thời gian thực.

        Returns:
            Tuple gồm:
                - annotated_frame: Ảnh đã được vẽ thông tin.
                - active_tracks: Danh sách track đang active.
                - ocr_cache: Bảng cache biển số đã chốt.
        """
        # ── ĐO TIMESTAMP NGAY ĐẦU (trước bất kỳ logic nào) ──────────────────
        # Dùng time.perf_counter() để đo thời gian xử lý pipeline (microsecond precision).
        # Dùng time.time() làm wall-clock độc lập để tính FPS thực tế giữa các frame.
        t_frame_start = time.perf_counter()
        now_wall = time.time()             # wall-clock ngay khi frame vào pipeline

        # ── TÍNH FPS BẰNG SLIDING WINDOW + EMA KẾT HỢP ──────────────────────
        # Đo dt chính xác: thời gian thực tế từ đầu frame trước đến đầu frame này.
        # Giới hạn dt tối đa 2 giây để loại trừ pause/debug làm méo FPS.
        if self._frame_ts > 0:
            dt = now_wall - self._frame_ts
            if 0.0001 < dt < 2.0:          # bỏ dt = 0 và dt quá lớn (pause)
                self._fps_window.append(dt)
                if len(self._fps_window) > self._FPS_WINDOW_SIZE:
                    self._fps_window.pop(0)

                # FPS từ sliding window (ổn định hơn instant FPS)
                window_fps = len(self._fps_window) / sum(self._fps_window)

                # EMA với α = 0.15: phản ứng nhanh với thay đổi thực sự
                # (α lớn = giá trị mới có trọng số cao hơn — đúng với kiến thức EMA)
                alpha = 0.15
                self.fps = window_fps if self.fps == 0.0 else (
                    alpha * window_fps + (1.0 - alpha) * self.fps
                )
        self._frame_ts = now_wall

        # 1. Tính toán mốc thời gian của frame
        if is_live:
            curr_time = now_wall          # tái sử dụng wall-clock đã đo bên trên
        else:
            curr_time = self.start_time + (frame_idx / self.fps_in)
        video_pos_s = frame_idx / self.fps_in

        # Ghi frame vào ring buffer (nếu bật tính năng clip)
        if self.clip_buffer.enabled:
            self.clip_buffer.add_frame(frame, curr_time)

        # 3. Chạy Detector (YOLO) hoặc tái sử dụng từ Tracker (Skip-frame)
        is_detection_frame = (frame_idx % self.skip_frames == 0)

        if is_detection_frame:
            t_det = time.perf_counter()
            xyxy, confidences, class_ids = self.detector.detect(frame)
            det_ms = (time.perf_counter() - t_det) * 1000
            self.det_times.append(det_ms)

            active_tracks = self.tracker.update(xyxy, confidences, class_ids)

            # Chạy phân loại xe nếu được kích hoạt
            if self.vehicle_classifier and self.vehicle_tracker:
                t_vc = time.perf_counter()
                v_xyxy, v_confs, v_class_ids = self.vehicle_classifier.detect(frame)
                vc_ms = (time.perf_counter() - t_vc) * 1000
                self.vc_times.append(vc_ms)
                self.vehicle_active_tracks = self.vehicle_tracker.update(
                    v_xyxy, v_confs, v_class_ids,
                    class_name_fn=self.vehicle_classifier.get_class_name,
                )
        else:
            active_tracks = self.tracker.get_tracks_for_skipped_frame()
            if self.vehicle_classifier and self.vehicle_tracker:
                self.vehicle_active_tracks = self.vehicle_tracker.get_tracks_for_skipped_frame()

        active_track_ids: Set[int] = {t[0] for t in active_tracks}
        h, w = frame.shape[:2]

        # 4. Xử lý OCR và Cập nhật TrackState
        for track_id, box, conf in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 10 or bh <= 10:
                continue

            plate_crop = frame[y1:y2, x1:x2]

            # Lấy hoặc khởi tạo TrackState
            track_state = self.track_manager.get_or_create(
                track_id=track_id,
                frame_idx=frame_idx,
                curr_time=curr_time,
                video_pos_s=video_pos_s,
                frame_shape=(h, w),
            )

            # Chỉ ghi quỹ đạo và tính chất lượng ở các frame chạy detector
            crop_quality = 0.5
            if is_detection_frame:
                track_state.add_trajectory_point(box, frame_idx, curr_time)
                quality_res = calculate_crop_quality(
                    crop=plate_crop,
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    frame_shape=(h, w),
                    det_conf=conf,
                )
                crop_quality = quality_res.score
                track_state.update_best_crop(plate_crop, crop_quality, frame_idx, full_frame=frame)

            # Gán kết loại xe (Spatial Association: tâm biển số nằm trong thân xe hoặc IoU cao nhất)
            if self.vehicle_classifier and self.vehicle_active_tracks:
                best_match_area = float("inf")
                best_match = None
                pcx = (x1 + x2) / 2.0
                pcy = (y1 + y2) / 2.0

                for v_tid, v_box, v_conf, v_cls_id, v_cls_name in self.vehicle_active_tracks:
                    vx1, vy1, vx2, vy2 = v_box
                    if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                        v_area = (vx2 - vx1) * (vy2 - vy1)
                        if v_area < best_match_area:
                            best_match_area = v_area
                            best_match = (v_cls_name, v_conf)

                # Fallback nếu lệch tâm nhẹ: kiểm tra tỷ lệ giao nhau (Intersection / Plate Area)
                if best_match is None:
                    max_overlap = 0.0
                    for v_tid, v_box, v_conf, v_cls_id, v_cls_name in self.vehicle_active_tracks:
                        vx1, vy1, vx2, vy2 = v_box
                        ix1, iy1 = max(x1, vx1), max(y1, vy1)
                        ix2, iy2 = min(x2, vx2), min(y2, vy2)
                        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        overlap_ratio = inter_area / max(1, bw * bh)
                        if overlap_ratio > 0.35 and overlap_ratio > max_overlap:
                            max_overlap = overlap_ratio
                            best_match = (v_cls_name, v_conf)

                if best_match:
                    v_type, v_conf = best_match
                    track_state.add_vehicle_observation(
                        class_name=v_type,
                        conf=v_conf,
                        strategy=self.config.vehicle_classifier.strategy,
                        lock_first_conf=self.config.vehicle_classifier.lock_first_conf,
                    )

            # --- CHIẾN LƯỢC 1: BỎ PHIẾU ĐA FRAME (VOTING - MẶC ĐỊNH) ---
            if self.strategy == "voting":
                if self.track_manager.should_ocr(track_state, frame_idx, crop_quality):
                    self.ocr_calls_count += 1
                    t_ocr = time.perf_counter()
                    raw_text, ocr_score = self.ocr.recognize_raw(plate_crop)
                    ocr_ms = (time.perf_counter() - t_ocr) * 1000
                    self.ocr_times.append(ocr_ms)

                    if raw_text:
                        vote_res = track_state.add_ocr_result(
                            raw_text=raw_text,
                            conf=ocr_score,
                            quality=crop_quality,
                            frame_idx=frame_idx,
                            use_char_correction=self.config.ocr.use_char_correction,
                            use_quality_selection=self.config.ocr.use_quality_selection,
                            early_lock_votes=self.config.ocr.early_lock_votes,
                            early_lock_conf=self.config.ocr.early_lock_conf,
                        )

                        if track_state.is_locked:
                            self.ocr_cache[track_id] = track_state.locked_plate
                            self.early_locks_count += 1
                            corr_str = " (đã hiệu chỉnh)" if track_state.locked_is_corrected else ""
                            print(
                                f"[ALPR] Đã CHỐT SỚM biển số: [Track #{track_id}] -> '{track_state.locked_plate}'{corr_str} "
                                f"(conf: {track_state.locked_conf:.2f}, votes: {vote_res.vote_count}) ở frame {frame_idx}"
                            )
                        else:
                            self.provisional_cache[track_id] = track_state.latest_plate

            # --- CHIẾN LƯỢC 2: KHÓA LẦN ĐẦU (LOCK_FIRST) ---
            else:
                if track_id not in self.ocr_cache:
                    last_f = self.lock_first_last_frame.get(track_id, -999)
                    if (frame_idx - last_f) > self.config.ocr.retry_interval_frames:
                        self.lock_first_last_frame[track_id] = frame_idx
                        self.ocr_calls_count += 1
                        t_ocr = time.perf_counter()
                        plate_text, score = self.ocr.recognize_plate(plate_crop)
                        ocr_ms = (time.perf_counter() - t_ocr) * 1000
                        self.ocr_times.append(ocr_ms)

                        if plate_text:
                            # TH 1: Chuỗi OCR nguyên bản đã đạt chuẩn cấu trúc biển số Việt Nam
                            if self.ocr.is_valid_vn_plate(plate_text):
                                self.ocr_cache[track_id] = plate_text
                                track_state.is_locked = True
                                track_state.locked_plate = plate_text
                                track_state.locked_plate_raw = plate_text
                                track_state.locked_is_corrected = False
                                track_state.locked_corrections = []
                                track_state.locked_conf = score
                                track_state.locked_is_valid = True
                                track_state.locked_at_frame = frame_idx
                                track_state.lock_reason = "lock_first_valid"
                                self.early_locks_count += 1
                                print(
                                    f"[ALPR] Khóa lần đầu: [Track #{track_id}] -> '{plate_text}' "
                                    f"(conf: {score:.2f}) ở frame {frame_idx}"
                                )
                            else:
                                # TH 2: Chuỗi chưa đạt chuẩn biển số VN
                                # TH 2a: Chuỗi quá ngắn (< 7 ký tự) -> skip và tiếp tục chờ cooldown
                                if len(plate_text) < 7:
                                    self.provisional_cache[track_id] = plate_text
                                else:
                                    # TH 2b: Chuỗi >= 7 ký tự -> Thêm cơ chế Sửa Ký Tự Theo Vị Trí Biển Số Việt Nam
                                    corr = correct_vn_plate(plate_text)
                                    corrected_plate = corr.plate_corrected

                                    # Nếu chuỗi sau sửa đạt chuẩn VN -> chuyển trạng thái và không OCR biển số này nữa
                                    if self.ocr.is_valid_vn_plate(corrected_plate):
                                        self.ocr_cache[track_id] = corrected_plate
                                        track_state.is_locked = True
                                        track_state.locked_plate = corrected_plate
                                        track_state.locked_plate_raw = plate_text
                                        track_state.locked_is_corrected = True
                                        track_state.locked_corrections = corr.corrections
                                        track_state.locked_conf = score
                                        track_state.locked_is_valid = True
                                        track_state.locked_at_frame = frame_idx
                                        track_state.lock_reason = "lock_first_corrected"
                                        self.early_locks_count += 1
                                        print(
                                            f"[ALPR] Khóa lần đầu (Đã sửa ký tự): [Track #{track_id}] -> '{corrected_plate}' "
                                            f"(Gốc: '{plate_text}', conf: {score:.2f}) ở frame {frame_idx}"
                                        )
                                    else:
                                        # Chuỗi sửa xong vẫn không đạt chuẩn -> tiếp tục chờ cooldown
                                        self.provisional_cache[track_id] = corrected_plate


        # 5. Phát hiện các track đã kết thúc và lưu vào CSDL qua writer thread
        finished_sightings = self.track_manager.pop_finished(frame_idx, active_track_ids)
        for s in finished_sightings:
            self._handle_finished_sighting(s)

        # 6. Chuẩn bị bảng tra loại xe cho từng plate track_id để hiển thị lên badge
        vehicle_type_map: Dict[int, str] = {}
        for tid in active_track_ids:
            if tid in self.track_manager.tracks:
                t_state = self.track_manager.tracks[tid]
                v_type = t_state.locked_vehicle_type if t_state.is_vehicle_locked else t_state.latest_vehicle_type
                if v_type:
                    vehicle_type_map[tid] = v_type

        # Vẽ thông tin trực quan lên khung hình
        draw_v_box = (
            self.config.vehicle_classifier.draw_box
            if self.vehicle_classifier
            else False
        )
        annotated_frame = self.visualizer.draw(
            frame=frame,
            tracks=active_tracks,
            ocr_cache=self.ocr_cache,
            provisional_cache=self.provisional_cache,
            vehicle_tracks=self.vehicle_active_tracks if self.vehicle_classifier else None,
            vehicle_type_map=vehicle_type_map,
            draw_vehicle_box=draw_v_box,
            fps=self.fps,
            ocr_calls=self.ocr_calls_count,
        )

        frame_ms = (time.perf_counter() - t_frame_start) * 1000
        self.pipeline_times.append(frame_ms)

        return annotated_frame, active_tracks, self.ocr_cache

    def _handle_finished_sighting(self, sighting: FinalizedSighting):
        """Xử lý và lưu một lượt xe đã kết thúc vào cơ sở dữ liệu."""
        if sighting.lock_reason == "track_end_vote":
            self.track_end_locks_count += 1

        # Cập nhật cache hiển thị lần cuối
        if sighting.plate:
            self.ocr_cache[sighting.track_id] = sighting.plate

        # Trích xuất video clip nếu bật clip
        if self.clip_buffer.enabled and sighting.plate:
            now_dt = time.strftime("%Y%m%d_%H%M%S", time.localtime(sighting.t_start))
            clip_name = f"{now_dt}_t{sighting.track_id}_{sighting.plate}.mp4"
            clip_path = Path(self.config.storage.clips_dir) / clip_name
            self.clip_buffer.export_clip(sighting.t_start, sighting.t_end, clip_path)

        # Gửi sang thread lưu trữ không đồng bộ
        self.storage_writer.submit(sighting, self.source_id)

    def process_image(self, image_path: str) -> np.ndarray:
        """Xử lý ảnh tĩnh đơn lẻ."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc file ảnh: {image_path}")

        annotated_img, _, _ = self.process_frame(img, frame_idx=0)
        self.finish()
        return annotated_img

    def finish(self):
        """Kết thúc luồng, chốt toàn bộ các track đang mở và đóng CSDL an toàn."""
        print("[Pipeline] Đang hoàn tất và chốt các track còn lại...")
        remaining_sightings = self.track_manager.flush_all()
        for s in remaining_sightings:
            self._handle_finished_sighting(s)

        self.storage_writer.close(timeout=5.0)
        print("[Pipeline] Toàn bộ dữ liệu đã được lưu trữ an toàn.")

    def print_benchmark_report(self):
        """In bảng thống kê chi tiết benchmark (thời gian load model, plate detection, OCR, pipeline)."""
        load_det = getattr(self.detector, "load_time_ms", 0.0)
        load_ocr = getattr(self.ocr, "load_time_ms", 0.0)
        total_init = load_det + load_ocr + getattr(self, "warmup_time_ms", 0.0)

        print("\n" + "=" * 65)
        print("          BÁO CÁO BENCHMARK CHI TIẾT HỆ THỐNG ALPR")
        print("=" * 65)

        print("1. THỜI GIAN NẠP & WARM-UP MÔ HÌNH (MODEL LOADING):")
        print(f"   * YOLO NCNN Plate Detector : {load_det:8.2f} ms")
        print(f"   * PaddleOCR ONNX Runtime   : {load_ocr:8.2f} ms")
        print(f"   * Warm-up Models           : {self.warmup_time_ms:8.2f} ms")
        print(f"   -> Tổng thời gian khởi tạo : {total_init:8.2f} ms")
        print("-" * 65)

        print("2. THỜI GIAN SUY LUẬN TỪNG MODULE (INFERENCE TIME):")
        if self.det_times:
            det_avg = float(np.mean(self.det_times))
            det_min = float(np.min(self.det_times))
            det_max = float(np.max(self.det_times))
            det_fps = 1000.0 / det_avg if det_avg > 0 else 0.0
            print(f"   * Plate Detection (YOLO NCNN):")
            print(f"       - Số lần chạy      : {len(self.det_times)} lần")
            print(f"       - Trung bình / lần : {det_avg:6.2f} ms  (tương đương ~{det_fps:.1f} FPS)")
            print(f"       - Nhanh nhất / Chậm: {det_min:6.2f} ms / {det_max:6.2f} ms")
        else:
            print("   * Plate Detection: Không có frame nào được chạy detector.")

        if self.ocr_times:
            ocr_avg = float(np.mean(self.ocr_times))
            ocr_min = float(np.min(self.ocr_times))
            ocr_max = float(np.max(self.ocr_times))
            print(f"   * Character OCR (PaddleOCR ONNX Runtime):")
            print(f"       - Số lần gọi OCR   : {len(self.ocr_times)} lần")
            print(f"       - Trung bình / lần : {ocr_avg:6.2f} ms")
            print(f"       - Nhanh nhất / Chậm: {ocr_min:6.2f} ms / {ocr_max:6.2f} ms")
        else:
            print("   * Character OCR: Không có lần gọi OCR nào.")

        if self.vc_times:
            vc_avg = float(np.mean(self.vc_times))
            vc_min = float(np.min(self.vc_times))
            vc_max = float(np.max(self.vc_times))
            vc_fps = 1000.0 / vc_avg if vc_avg > 0 else 0.0
            print(f"   * Vehicle Classification (YOLO NCNN 14 Classes):")
            print(f"       - Số lần chạy      : {len(self.vc_times)} lần")
            print(f"       - Trung bình / lần : {vc_avg:6.2f} ms  (tương đương ~{vc_fps:.1f} FPS)")
            print(f"       - Nhanh nhất / Chậm: {vc_min:6.2f} ms / {vc_max:6.2f} ms")
        print("-" * 65)

        print("3. HIỆU NĂNG TOÀN BỘ PIPELINE & ĐỘ TIN CẬY:")
        det_avg_val = float(np.mean(self.det_times)) if self.det_times else 0.0
        ocr_avg_val = float(np.mean(self.ocr_times)) if self.ocr_times else 0.0
        core_time = det_avg_val + ocr_avg_val

        print(f"   * Thời gian Pipeline thuần (Plate Detection + OCR):")
        print(f"       = {det_avg_val:.2f} ms (Detection) + {ocr_avg_val:.2f} ms (OCR) = {core_time:.2f} ms")

        if self.pipeline_times:
            pipe_avg = float(np.mean(self.pipeline_times))
            pipe_min = float(np.min(self.pipeline_times))
            pipe_max = float(np.max(self.pipeline_times))
            pipe_fps = 1000.0 / pipe_avg if pipe_avg > 0 else 0.0
            total_time_s = sum(self.pipeline_times) / 1000.0

            print(f"   * Toàn bộ Pipeline thực tế (Detection + Track + OCR + Draw):")
            print(f"       - Tổng số frames xử lý : {len(self.pipeline_times)}")
            print(f"       - Trung bình mỗi frame : {pipe_avg:6.2f} ms (~{pipe_fps:.1f} FPS lý thuyết xử lý)")
            print(f"         ⚠ Lưu ý: FPS thực tế thấp hơn do bao gồm thời gian đọc frame I/O")
            print(f"       - Nhanh nhất / Chậm    : {pipe_min:6.2f} ms / {pipe_max:6.2f} ms")
            print(f"       - Tổng thời gian chạy  : {total_time_s:6.2f} s")
            print(f"   * Thống kê chốt kết quả (Locking Stats):")
            print(f"       - Chốt sớm (Early-lock): {self.early_locks_count} lượt")
            print(f"       - Chốt khi hết track  : {self.track_end_locks_count} lượt")
            if getattr(self.storage_writer, "enabled", True):
                print(f"       - Đã lưu vào CSDL      : {self.storage_writer.written_count} lượt (Rơi: {self.storage_writer.dropped_count})")
            else:
                print(f"       - Đã lưu vào CSDL      : ĐÃ TẮT (--no-db / --no-store)")
        print("=" * 65 + "\n")

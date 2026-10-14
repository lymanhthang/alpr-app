"""
Pipeline điều phối toàn bộ luồng ALPR:
Frame -> Skip-frame / YOLO NCNN -> ByteTrack -> One-time OCR Cache -> Visualization
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import AppConfig
from src.detector import PlateDetector
from src.tracker import PlateTracker
from src.ocr import PlateOCR
from src.visualizer import Visualizer


class ALPRPipeline:
    """Hệ thống ALPR hoàn chỉnh cho video và ảnh."""

    def __init__(self, config: AppConfig, benchmark: bool = False):
        self.config = config
        self.benchmark = benchmark

        print("\n" + "=" * 50)
        print("KHỞI TẠO HỆ THỐNG ALPR CHO RASPBERRY PI 5")
        print("=" * 50)

        # 1. Khởi tạo các module
        self.detector = PlateDetector(config.detector)
        self.tracker = PlateTracker(config.tracker)
        self.ocr = PlateOCR(config.ocr)
        self.visualizer = Visualizer()

        # 2. Warm-up các mô hình với dummy input để nạp bộ nhớ đệm trước
        print("[Pipeline] Đang warm-up mô hình YOLO NCNN & PaddleOCR...")
        t_warmup = time.perf_counter()
        self.detector.warmup(imgsz=config.detector.imgsz)
        self.ocr.warmup()
        self.warmup_time_ms = (time.perf_counter() - t_warmup) * 1000
        print(f"[Pipeline] Warm-up hoàn tất trong {self.warmup_time_ms:.2f} ms.")

        # 3. Quản lý trạng thái và bộ nhớ đệm OCR (Cache)
        self.ocr_cache: Dict[int, str] = {}           # {track_id: plate_text} chỉ lưu biển số VN hợp lệ
        self.ocr_last_frame: Dict[int, int] = {}      # {track_id: frame_idx} thời điểm gần nhất chạy OCR
        self.ocr_calls_count: int = 0                 # Tổng số lần gọi OCR
        self.skip_frames: int = max(1, config.tracker.skip_frames)
        self.retry_interval_frames: int = getattr(config.ocr, "retry_interval_frames", 2)

        # 4. Thống kê benchmark
        self.det_times: List[float] = []              # Thời gian mỗi lần chạy YOLO (ms)
        self.ocr_times: List[float] = []              # Thời gian mỗi lần chạy OCR (ms)
        self.pipeline_times: List[float] = []         # Thời gian toàn bộ pipeline mỗi frame (ms)

        # 5. Tính toán FPS
        self.prev_time: float = time.time()
        self.fps: float = 0.0

        print(f"[Pipeline] Sẵn sàng. Tỷ lệ Skip-frame: mỗi {self.skip_frames} frame.")
        if self.benchmark:
            print("[Pipeline] Chế độ Benchmark CHI TIẾT: ĐÃ BẬT.")
        print("=" * 50 + "\n")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray, float]], Dict[int, str]]:
        """
        Xử lý 1 khung hình qua toàn bộ pipeline.

        Args:
            frame: Ảnh BGR.
            frame_idx: Chỉ số frame hiện tại (0-indexed).

        Returns:
            Tuple gồm:
                - annotated_frame: Ảnh đã được vẽ box và thông tin.
                - active_tracks: Danh sách track đang active [(track_id, box, conf)].
                - ocr_cache: Toàn bộ bảng cache biển số đã nhận diện.
        """
        t_frame_start = time.perf_counter()

        # 1. Tính toán FPS tức thời
        curr_time = time.time()
        dt = curr_time - self.prev_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = (self.fps * 0.9 + instant_fps * 0.1) if self.fps > 0 else instant_fps
        self.prev_time = curr_time

        # 2. Quyết định chạy Detector hay Skip
        # Frame được chọn chạy YOLO khi frame_idx chia hết cho skip_frames
        is_detection_frame = (frame_idx % self.skip_frames == 0)

        if is_detection_frame:
            # Chạy YOLO NCNN phát hiện biển số (đo thời gian nếu bật benchmark)
            t_det = time.perf_counter()
            xyxy, confidences, class_ids = self.detector.detect(frame)
            det_ms = (time.perf_counter() - t_det) * 1000
            self.det_times.append(det_ms)

            # Cập nhật ByteTrack với các box mới
            active_tracks = self.tracker.update(xyxy, confidences, class_ids)
        else:
            # Frame bị skip: tái sử dụng track từ ByteTrack để giữ bounding box ổn định
            active_tracks = self.tracker.get_tracks_for_skipped_frame()

        # 3. Duyệt qua các track và áp dụng cơ chế OCR 1-LẦN (Cache)
        h, w = frame.shape[:2]

        for track_id, box, conf in active_tracks:
            # TRƯỜNG HỢP 1: Track ID này ĐÃ CÓ trong cache (đã nhận diện biển VN hợp lệ)
            if track_id in self.ocr_cache:
                continue

            # TRƯỜNG HỢP 2: Track ID chưa có biển số hợp lệ
            # Kiểm tra thời điểm gọi OCR gần nhất của track này
            last_ocr_f = self.ocr_last_frame.get(track_id, -999)
            if (frame_idx - last_ocr_f) <= self.retry_interval_frames:
                # Bỏ qua không OCR đối tượng này trong 2 frame tiếp theo
                continue

            # Cắt ảnh vùng biển số khi lần đầu xuất hiện hoặc đã qua 2 frame tiếp theo
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                plate_crop = frame[y1:y2, x1:x2]

                # Ghi nhận frame này đã chạy OCR cho track_id
                self.ocr_last_frame[track_id] = frame_idx
                self.ocr_calls_count += 1
                t_ocr = time.perf_counter()
                plate_text, score = self.ocr.recognize_plate(plate_crop)
                ocr_ms = (time.perf_counter() - t_ocr) * 1000
                self.ocr_times.append(ocr_ms)

                if plate_text and self.ocr.is_valid_vn_plate(plate_text):
                    # Đạt chuẩn biển số Việt Nam -> Khóa vĩnh viễn vào Cache
                    self.ocr_cache[track_id] = plate_text
                    print(
                        f"[ALPR] Đã nhận diện biển số hợp lệ: "
                        f"[Track #{track_id}] -> '{plate_text}' (conf: {score:.2f}) "
                        f"ở frame {frame_idx} (OCR: {ocr_ms:.2f} ms)"
                    )
                else:
                    # Chưa đạt chuẩn: không lưu cache, sẽ được bỏ qua trong 2 frame tiếp theo và tiếp tục thử lại
                    raw_str = plate_text if plate_text else "None"
                    print(
                        f"[ALPR Info] Track #{track_id} (frame {frame_idx}): OCR ra '{raw_str}' "
                        f"(chưa đạt chuẩn biển VN, sẽ thử lại sau {self.retry_interval_frames} frame)"
                    )

        # 4. Vẽ kết quả lên ảnh
        annotated_frame = self.visualizer.draw(
            frame=frame,
            tracks=active_tracks,
            ocr_cache=self.ocr_cache,
            fps=self.fps,
            ocr_calls=self.ocr_calls_count,
        )

        frame_ms = (time.perf_counter() - t_frame_start) * 1000
        self.pipeline_times.append(frame_ms)

        return annotated_frame, active_tracks, self.ocr_cache

    def process_image(self, image_path: str) -> np.ndarray:
        """Xử lý ảnh tĩnh đơn lẻ."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc file ảnh: {image_path}")

        annotated_img, _, _ = self.process_frame(img, frame_idx=0)
        return annotated_img

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
            print(f"       - Số lần gọi OCR   : {len(self.ocr_times)} lần (áp dụng Cache 1-lần)")
            print(f"       - Trung bình / lần : {ocr_avg:6.2f} ms")
            print(f"       - Nhanh nhất / Chậm: {ocr_min:6.2f} ms / {ocr_max:6.2f} ms")
        else:
            print("   * Character OCR: Không có lần gọi OCR nào (hoặc biển số đã được cache).")
        print("-" * 65)

        print("3. HIỆU NĂNG TOÀN BỘ PIPELINE (PIPELINE PERFORMANCE):")
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
            print(f"       - Trung bình mỗi frame : {pipe_avg:6.2f} ms")
            print(f"       - Tốc độ thực tế       : ~{pipe_fps:.1f} FPS")
            print(f"       - Nhanh nhất / Chậm    : {pipe_min:6.2f} ms / {pipe_max:6.2f} ms")
            print(f"       - Tổng thời gian chạy  : {total_time_s:6.2f} s")
        print("=" * 65 + "\n")

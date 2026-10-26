"""
Điểm khởi chạy chính của ứng dụng ALPR trên Raspberry Pi 5.
Hỗ trợ đầu vào: Ảnh (.jpg, .png), Video (.mp4, .avi), Webcam hoặc luồng RTSP.
Tích hợp lưu trữ SQLite WAL, cơ chế voting đa frame và tra cứu lịch sử xe.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import cv2

from src.config import load_config
from src.pipeline import ALPRPipeline
from src.storage.db import DatabaseManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hệ thống ALPR (Nhận diện biển số xe) tối ưu cho Raspberry Pi 5"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Đường dẫn file ảnh, video, hoặc số thứ tự webcam (VD: 0) hoặc URL RTSP",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Tên định danh nguồn (VD: 'Cổng chính', 'Cam_01', 'Demo_Video'). Tự suy luận nếu bỏ trống.",
    )
    parser.add_argument(
        "--source-kind",
        type=str,
        choices=["camera", "video"],
        default=None,
        help="Loại nguồn: 'camera' hoặc 'video'. Tự động nhận diện nếu bỏ trống.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["voting", "lock_first"],
        default=None,
        help="Chiến lược OCR: 'voting' (bỏ phiếu đa frame, khuyến nghị) hoặc 'lock_first' (khóa lần đầu).",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Mốc thời gian bắt đầu (timestamp số hoặc chuỗi ISO 'YYYY-MM-DDTHH:MM:SS'). Mặc định: thời gian hiện tại.",
    )
    parser.add_argument(
        "--no-store",
        "--no-db",
        dest="no_store",
        action="store_true",
        help="Không lưu dữ liệu lượt xe và ảnh crop vào SQLite (phục vụ test và benchmark).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Đường dẫn file cấu hình YAML (mặc định: config.yaml)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="Ghi đè số frame bỏ qua giữa các lần chạy YOLO (VD: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file lưu kết quả ảnh hoặc video",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Bật cửa sổ hiển thị trực tiếp (yêu cầu Raspberry Pi 5 có màn hình)",
    )
    parser.add_argument(
        "--stream",
        "--web-stream",
        dest="web_stream",
        action="store_true",
        help="Bắn luồng video đã xử lý qua Web HTTP (MJPEG) để xem trực tiếp trên trình duyệt",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=None,
        help="Cổng phát luồng Web (mặc định lấy từ config hoặc 8080)",
    )
    parser.add_argument(
        "--stream-host",
        type=str,
        default=None,
        help="Địa chỉ IP host phát luồng Web (mặc định: 0.0.0.0)",
    )
    parser.add_argument(
        "--classify-vehicle",
        "--vehicle-classify",
        dest="classify_vehicle",
        action="store_true",
        help="Bật chức năng phát hiện và phân loại 14 loại xe 2 bánh Việt Nam (NCNN)",
    )
    parser.add_argument(
        "--vehicle-model",
        type=str,
        default=None,
        help="Đường dẫn thư mục mô hình phân loại xe NCNN (mặc định: models/vehicle_classify/384x640_fp16_ncnn_model)",
    )
    parser.add_argument(
        "--vehicle-strategy",
        type=str,
        choices=["voting", "lock_first"],
        default=None,
        help="Chiến lược phân loại xe: 'voting' (bỏ phiếu đa frame) hoặc 'lock_first' (khóa loại xe ngay lần đầu)",
    )
    parser.add_argument(
        "--no-draw-vehicle-box",
        dest="no_draw_vehicle_box",
        action="store_true",
        help="Ẩn bounding box thân xe trên khung hình (chỉ hiển thị loại xe trên nhãn biển số)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Bật chế độ đo benchmark chi tiết (thời gian load model, plate detection, ocr, pipeline)",
    )
    return parser.parse_args()


def is_image_file(source: str) -> bool:
    ext = Path(source).suffix.lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_start_time(time_str: str) -> float:
    """Chuyển đổi chuỗi thời gian thành unix timestamp."""
    try:
        return float(time_str)
    except ValueError:
        pass

    try:
        dt = datetime.fromisoformat(time_str)
        return dt.timestamp()
    except Exception:
        print(f"[Cảnh báo] Không thể đọc định dạng --start-time '{time_str}', sử dụng thời gian hiện tại.")
        return time.time()


def process_image(
    source: str,
    pipeline: ALPRPipeline,
    output_path: str,
    show: bool,
    web_stream: bool = False,
    stream_host: str = "0.0.0.0",
    stream_port: int = 8080,
    stream_quality: int = 80,
    source_name: str = "Ảnh ALPR",
):
    print(f"\n[ALPR] Đang xử lý ảnh: {source}...")
    start_t = time.time()

    annotated = pipeline.process_image(source)
    duration = time.time() - start_t

    print(f"[ALPR] Hoàn tất sau {duration:.2f} giây.")
    print(f"[ALPR] Danh sách biển số nhận diện được: {pipeline.ocr_cache}")

    if pipeline.benchmark:
        pipeline.print_benchmark_report()

    # Lưu kết quả
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), annotated)
        print(f"[ALPR] Đã lưu ảnh kết quả tại: {out_p.resolve()}")

    if web_stream:
        from src.streamer import WebStreamer
        streamer = WebStreamer(
            host=stream_host,
            port=stream_port,
            source_name=source_name,
            jpeg_quality=stream_quality,
        )
        streamer.start()
        streamer.update_frame(
            frame=annotated,
            fps=0.0,
            active_tracks_count=len(pipeline.ocr_cache),
            total_locked_count=len(pipeline.ocr_cache),
        )
        print("[ALPR] Đang phát ảnh kết quả qua Web. Nhấn Ctrl+C để dừng...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            streamer.stop()

    if show:
        cv2.imshow("ALPR Result", annotated)
        print("[ALPR] Nhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video_or_camera(
    source: str,
    pipeline: ALPRPipeline,
    output_path: str,
    show: bool,
    is_live: bool,
    web_stream: bool = False,
    stream_host: str = "0.0.0.0",
    stream_port: int = 8080,
    stream_quality: int = 80,
    source_name: str = "Camera ALPR",
):
    if source.isdigit():
        cap_source = int(source)
        print(f"\n[ALPR] Đang mở Camera thiết bị số: {cap_source}...")
    else:
        cap_source = source
        print(f"\n[ALPR] Đang mở Video / Stream: {cap_source}...")

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[Lỗi] Không thể mở nguồn video: {source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pipeline.fps_in = fps_in
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0

    writer = None
    if output_path and not is_live:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps_in, (width, height))
        print(f"[ALPR] Video kết quả sẽ được ghi vào: {out_p.resolve()}")

    # Khởi động Web Streamer nếu được kích hoạt
    streamer = None
    if web_stream:
        from src.streamer import WebStreamer
        streamer = WebStreamer(
            host=stream_host,
            port=stream_port,
            source_name=source_name,
            jpeg_quality=stream_quality,
        )
        streamer.start()

    frame_idx = 0
    start_wall_time = time.time()

    print("[ALPR] Bắt đầu luồng xử lý... Nhấn Ctrl+C (hoặc 'q' nếu bật màn hình) để dừng.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, active_tracks, ocr_cache = pipeline.process_frame(
                frame, frame_idx=frame_idx, is_live=is_live
            )

            # Bắn khung hình đã vẽ thông tin tới máy chủ Web Streamer
            if streamer:
                streamer.update_frame(
                    frame=annotated_frame,
                    fps=pipeline.fps,
                    active_tracks_count=len(active_tracks),
                    total_locked_count=len(ocr_cache),
                )

            if writer:
                writer.write(annotated_frame)

            if show:
                cv2.imshow("ALPR Live - Raspberry Pi 5", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[ALPR] Đã nhận lệnh dừng từ người dùng.")
                    break

            frame_idx += 1
            if not is_live and total_frames > 0 and frame_idx % 30 == 0:
                percent = (frame_idx / total_frames) * 100
                print(
                    f"  -> Tiến độ: {frame_idx}/{total_frames} frames ({percent:.1f}%) | "
                    f"Biển số đã chốt: {len(ocr_cache)}"
                )

    except KeyboardInterrupt:
        print("\n[ALPR] Đã dừng theo yêu cầu (KeyboardInterrupt).")

    finally:
        total_time = time.time() - start_wall_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0.0

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        if streamer:
            streamer.stop()

        # Hoàn tất pipeline và chốt các track còn lại
        pipeline.finish()

        print("\n" + "=" * 50)
        print("TỔNG KẾT KẾT QUẢ XỬ LÝ:")
        print(f" - Tổng số frame đã xử lý : {frame_idx}")
        print(f" - Thời gian thực thi     : {total_time:.2f}s")
        print(f" - Tốc độ trung bình      : {avg_fps:.2f} FPS")
        print(f" - Tổng số lần gọi OCR    : {pipeline.ocr_calls_count}")
        print(f" - Danh sách biển số thu thập được:")
        for tid, text in pipeline.ocr_cache.items():
            print(f"    * Track #{tid}: {text}")
        if output_path and writer:
            print(f" - File video đã lưu tại   : {output_path}")
        print("=" * 50 + "\n")

        if pipeline.benchmark:
            pipeline.print_benchmark_report()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Ghi đè cấu hình từ CLI
    if args.skip_frames is not None:
        config.tracker.skip_frames = args.skip_frames
    if args.strategy is not None:
        config.ocr.strategy = args.strategy

    # Ghi đè cấu hình phân loại xe
    if args.classify_vehicle:
        config.vehicle_classifier.enabled = True
    if args.vehicle_model is not None:
        config.vehicle_classifier.model_path = args.vehicle_model
    if args.vehicle_strategy is not None:
        config.vehicle_classifier.strategy = args.vehicle_strategy
    if args.no_draw_vehicle_box:
        config.vehicle_classifier.draw_box = False

    # Tùy chọn phát luồng web
    web_stream = args.web_stream or config.output.web_stream
    stream_host = args.stream_host or config.output.stream_host or "0.0.0.0"
    stream_port = args.stream_port or config.output.stream_port or 8080
    stream_quality = getattr(config.output, "stream_quality", 80)

    # Tự động nhận diện loại nguồn và tên nguồn
    is_live = False
    source_kind = args.source_kind
    if source_kind is None:
        if args.source.isdigit() or args.source.startswith("rtsp://") or args.source.startswith("http://"):
            source_kind = "camera"
            is_live = True
        else:
            source_kind = "video"
            is_live = False

    source_name = args.source_name
    if source_name is None:
        if args.source.isdigit():
            source_name = f"Camera_{args.source}"
        else:
            source_name = Path(args.source).stem

    # Mốc thời gian bắt đầu
    start_ts = parse_start_time(args.start_time) if args.start_time else time.time()

    # Đăng ký hoặc lấy ID nguồn từ SQLite DB
    source_id = 1
    if not args.no_store:
        try:
            db = DatabaseManager(config.storage.db_path)
            source_id = db.get_or_create_source(
                name=source_name, kind=source_kind, uri=str(args.source)
            )
            print(f"[ALPR] Đã gán Source: '{source_name}' (ID: {source_id}, Loại: {source_kind})")
        except Exception as e:
            print(f"[ALPR Warning] Không thể kết nối DB để lấy source_id: {e}")
    else:
        print("[ALPR] ⚠️  Chế độ TEST / BENCHMARK (--no-db / --no-store): Đã TẮT lưu CSDL SQLite và lưu ảnh crop.")

    # Đường dẫn output
    output_path = args.output
    if output_path is None and config.output.save_result:
        ext = ".jpg" if is_image_file(args.source) else ".mp4"
        output_path = f"{config.output.output_dir}/output_{source_name}{ext}"

    show = args.show or config.output.show_live

    # Khởi tạo pipeline
    pipeline = ALPRPipeline(
        config=config,
        benchmark=args.benchmark,
        source_id=source_id,
        fps_in=25.0,
        start_time=start_ts,
        no_store=args.no_store,
    )

    if is_image_file(args.source):
        process_image(
            source=args.source,
            pipeline=pipeline,
            output_path=output_path,
            show=show,
            web_stream=web_stream,
            stream_host=stream_host,
            stream_port=stream_port,
            stream_quality=stream_quality,
            source_name=source_name,
        )
    else:
        process_video_or_camera(
            source=args.source,
            pipeline=pipeline,
            output_path=output_path,
            show=show,
            is_live=is_live,
            web_stream=web_stream,
            stream_host=stream_host,
            stream_port=stream_port,
            stream_quality=stream_quality,
            source_name=source_name,
        )


if __name__ == "__main__":
    main()

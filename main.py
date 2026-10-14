"""
Điểm khởi chạy chính của ứng dụng ALPR trên Raspberry Pi 5.
Hỗ trợ đầu vào: Ảnh (.jpg, .png), Video (.mp4, .avi), Webcam hoặc luồng RTSP.
"""

import argparse
from pathlib import Path
import sys
import time
import cv2

from src.config import load_config
from src.pipeline import ALPRPipeline


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
        "--benchmark",
        action="store_true",
        help="Bật chế độ đo benchmark chi tiết (thời gian load model, plate detection, ocr, pipeline)",
    )
    return parser.parse_args()


def is_image_file(source: str) -> bool:
    ext = Path(source).suffix.lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def process_image(source: str, pipeline: ALPRPipeline, output_path: str, show: bool):
    print(f"\n[ALPR] Đang xử lý ảnh: {source}...")
    start_t = time.time()

    annotated = pipeline.process_image(source)
    duration = time.time() - start_t

    print(f"[ALPR] Hoàn tất sau {duration:.2f} giây.")
    print(f"[ALPR] Danh sách biển số nhận diện được: {pipeline.ocr_cache}")

    # In benchmark nếu được kích hoạt
    if pipeline.benchmark:
        pipeline.print_benchmark_report()

    # Lưu kết quả
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), annotated)
        print(f"[ALPR] Đã lưu ảnh kết quả tại: {out_p.resolve()}")

    if show:
        cv2.imshow("ALPR Result", annotated)
        print("[ALPR] Nhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video_or_camera(
    source: str, pipeline: ALPRPipeline, output_path: str, show: bool
):
    # Kiểm tra nguồn là camera index hay file/RTSP
    if source.isdigit():
        cap_source = int(source)
        is_live = True
        print(f"\n[ALPR] Đang mở Camera thiết bị số: {cap_source}...")
    else:
        cap_source = source
        is_live = False
        print(f"\n[ALPR] Đang mở Video / Stream: {cap_source}...")

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[Lỗi] Không thể mở nguồn video: {source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0

    writer = None
    if output_path and not is_live:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps_in, (width, height))
        print(f"[ALPR] Video kết quả sẽ được ghi vào: {out_p.resolve()}")

    frame_idx = 0
    start_time = time.time()

    print("[ALPR] Bắt đầu luồng xử lý... Nhấn Ctrl+C (hoặc 'q' nếu bật màn hình) để dừng.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, active_tracks, ocr_cache = pipeline.process_frame(
                frame, frame_idx=frame_idx
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
                    f"Biển số đã đọc: {len(ocr_cache)}"
                )

    except KeyboardInterrupt:
        print("\n[ALPR] Đã dừng theo yêu cầu (KeyboardInterrupt).")

    finally:
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0.0

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        print("\n" + "=" * 50)
        print("TỔNG KẾT KẾT QUẢ XỬ LÝ:")
        print(f" - Tổng số frame đã xử lý: {frame_idx}")
        print(f" - Thời gian thực thi: {total_time:.2f}s")
        print(f" - Tốc độ trung bình: {avg_fps:.2f} FPS")
        print(f" - Tổng số lần gọi OCR: {pipeline.ocr_calls_count}")
        print(f" - Danh sách biển số thu thập được:")
        for tid, text in pipeline.ocr_cache.items():
            print(f"    * Track #{tid}: {text}")
        if output_path and writer:
            print(f" - File video đã lưu tại: {output_path}")
        print("=" * 50 + "\n")

        # In báo cáo benchmark chi tiết nếu được kích hoạt
        if pipeline.benchmark:
            pipeline.print_benchmark_report()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Ghi đè skip-frames từ tham số dòng lệnh nếu có
    if args.skip_frames is not None:
        config.tracker.skip_frames = args.skip_frames

    # Xác định đường dẫn file output mặc định nếu cần lưu
    output_path = args.output
    if output_path is None and config.output.save_result:
        source_name = Path(args.source).stem if not args.source.isdigit() else "camera"
        ext = ".jpg" if is_image_file(args.source) else ".mp4"
        output_path = f"{config.output.output_dir}/output_{source_name}{ext}"

    show = args.show or config.output.show_live

    # Khởi tạo pipeline (có hỗ trợ chế độ benchmark)
    pipeline = ALPRPipeline(config, benchmark=args.benchmark)

    if is_image_file(args.source):
        process_image(args.source, pipeline, output_path, show)
    else:
        process_video_or_camera(args.source, pipeline, output_path, show)


if __name__ == "__main__":
    main()

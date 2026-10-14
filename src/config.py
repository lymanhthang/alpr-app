"""
Cấu hình tập trung cho ứng dụng ALPR.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DetectorConfig:
    model_path: str = "models/lp_det/best_ncnn_model"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640


@dataclass
class TrackerConfig:
    track_activation_threshold: float = 0.3
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    skip_frames: int = 3


@dataclass
class OCRConfig:
    engine: str = "onnxruntime"
    lang: str = "en"
    det_model_name: str = "PP-OCRv6_tiny_det"
    rec_model_name: str = "PP-OCRv6_tiny_rec"
    det_model_dir: Optional[str] = "models/ch_det"
    rec_model_dir: Optional[str] = "models/ch_rec"
    use_angle_cls: bool = False
    min_confidence: float = 0.5
    retry_interval_frames: int = 2  # Bỏ qua 2 frame tiếp theo nếu chưa đạt chuẩn biển số VN


@dataclass
class OutputConfig:
    save_result: bool = True
    output_dir: str = "outputs"
    show_live: bool = False


@dataclass
class AppConfig:
    device: str = "cpu"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Đọc cấu hình từ file YAML, có fallback về giá trị mặc định."""
    path = Path(config_path)
    if not path.is_file():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    det_data = data.get("detector", {})
    tracker_data = data.get("tracker", {})
    ocr_data = data.get("ocr", {})
    out_data = data.get("output", {})

    return AppConfig(
        device=data.get("device", "cpu"),
        detector=DetectorConfig(
            model_path=det_data.get("model_path", "models/lp_det/best_ncnn_model"),
            conf_threshold=float(det_data.get("conf_threshold", 0.35)),
            iou_threshold=float(det_data.get("iou_threshold", 0.45)),
            imgsz=int(det_data.get("imgsz", 640)),
        ),
        tracker=TrackerConfig(
            track_activation_threshold=float(tracker_data.get("track_activation_threshold", 0.3)),
            lost_track_buffer=int(tracker_data.get("lost_track_buffer", 30)),
            minimum_matching_threshold=float(tracker_data.get("minimum_matching_threshold", 0.8)),
            skip_frames=int(tracker_data.get("skip_frames", 3)),
        ),
        ocr=OCRConfig(
            engine=ocr_data.get("engine", "onnxruntime"),
            lang=ocr_data.get("lang", "en"),
            det_model_name=ocr_data.get("det_model_name", "PP-OCRv6_tiny_det"),
            rec_model_name=ocr_data.get("rec_model_name", "PP-OCRv6_tiny_rec"),
            det_model_dir=ocr_data.get("det_model_dir"),
            rec_model_dir=ocr_data.get("rec_model_dir"),
            use_angle_cls=bool(ocr_data.get("use_angle_cls", False)),
            min_confidence=float(ocr_data.get("min_confidence", 0.5)),
            retry_interval_frames=int(ocr_data.get("retry_interval_frames", 2)),
        ),
        output=OutputConfig(
            save_result=bool(out_data.get("save_result", True)),
            output_dir=out_data.get("output_dir", "outputs"),
            show_live=bool(out_data.get("show_live", False)),
        ),
    )

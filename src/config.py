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

    # Nâng cao độ tin cậy
    strategy: str = "voting"  # "voting" hoặc "lock_first" (chế độ cũ)
    use_regex_filter: bool = True
    use_quality_selection: bool = True
    use_char_correction: bool = True
    quality_min_score: float = 0.35
    max_ocr_per_track: int = 8
    early_lock_enabled: bool = True
    early_lock_votes: int = 3
    early_lock_conf: float = 0.85


@dataclass
class StorageConfig:
    enabled: bool = True
    db_path: str = "data/alpr.db"
    crops_dir: str = "data/crops"
    contexts_dir: str = "data/contexts"
    clips_dir: str = "data/clips"
    retention_days: int = 30
    max_disk_gb: float = 20.0
    queue_maxsize: int = 256


@dataclass
class ClipConfig:
    enabled: bool = False
    pre_seconds: float = 3.0
    post_seconds: float = 2.0
    scale: float = 0.5


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000


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
    storage: StorageConfig = field(default_factory=StorageConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
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
    storage_data = data.get("storage", {})
    clip_data = data.get("clip", {})
    api_data = data.get("api", {})
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
            strategy=str(ocr_data.get("strategy", "voting")),
            use_regex_filter=bool(ocr_data.get("use_regex_filter", True)),
            use_quality_selection=bool(ocr_data.get("use_quality_selection", True)),
            use_char_correction=bool(ocr_data.get("use_char_correction", True)),
            quality_min_score=float(ocr_data.get("quality_min_score", 0.35)),
            max_ocr_per_track=int(ocr_data.get("max_ocr_per_track", 8)),
            early_lock_enabled=bool(ocr_data.get("early_lock_enabled", True)),
            early_lock_votes=int(ocr_data.get("early_lock_votes", 3)),
            early_lock_conf=float(ocr_data.get("early_lock_conf", 0.85)),
        ),
        storage=StorageConfig(
            enabled=bool(storage_data.get("enabled", True)),
            db_path=str(storage_data.get("db_path", "data/alpr.db")),
            crops_dir=str(storage_data.get("crops_dir", "data/crops")),
            contexts_dir=str(storage_data.get("contexts_dir", "data/contexts")),
            clips_dir=str(storage_data.get("clips_dir", "data/clips")),
            retention_days=int(storage_data.get("retention_days", 30)),
            max_disk_gb=float(storage_data.get("max_disk_gb", 20.0)),
            queue_maxsize=int(storage_data.get("queue_maxsize", 256)),
        ),
        clip=ClipConfig(
            enabled=bool(clip_data.get("enabled", False)),
            pre_seconds=float(clip_data.get("pre_seconds", 3.0)),
            post_seconds=float(clip_data.get("post_seconds", 2.0)),
            scale=float(clip_data.get("scale", 0.5)),
        ),
        api=ApiConfig(
            host=str(api_data.get("host", "0.0.0.0")),
            port=int(api_data.get("port", 8000)),
        ),
        output=OutputConfig(
            save_result=bool(out_data.get("save_result", True)),
            output_dir=out_data.get("output_dir", "outputs"),
            show_live=bool(out_data.get("show_live", False)),
        ),
    )

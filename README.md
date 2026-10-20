# 🚗 Automatic License Plate Recognition (ALPR) System

### _Tối ưu hóa hiệu năng cao cho Raspberry Pi 5 (ARM64 - Python 3.10)_

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![Detector](https://img.shields.io/badge/YOLO-NCNN%20ARM64-00FFFF)](https://github.com/ultralytics/ultralytics)
[![OCR](https://img.shields.io/badge/PaddleOCR-PP--OCRv6__tiny%20ONNX-blue)](https://github.com/PaddlePaddle/PaddleOCR)
[![Tracker](<https://img.shields.io/badge/Tracking-ByteTrack%20(supervision)-brightgreen>)](https://github.com/roboflow/supervision)

Hệ thống nhận diện biển số xe tự động (ALPR - Automatic License Plate Recognition) hoàn chỉnh, được thiết kế chuyên biệt để hoạt động trên thiết bị nhúng **Raspberry Pi 5**.

---

## 🌟 Tính Năng Nổi Bật

- ⚡ **YOLO NCNN Plate Detector**: Tận dụng định dạng **NCNN** tối ưu hóa tập lệnh SIMD NEON trên CPU ARM64 của Raspberry Pi 5, nạp trực tiếp qua Ultralytics.
- 🎯 **ByteTrack + Skip-Frame**:
  - Tích hợp thuật toán **ByteTrack** từ thư viện `supervision` (thuần NumPy, không cần build C++ extension, tránh lỗi trên Linux ARM64).
  - Kỹ thuật **Skip-frame** thông minh: chạy detector cách quãng (VD: mỗi 3 frame) và duy trì track mượt mà ở các frame bị bỏ qua, giảm tới 60-70% tải CPU.
- 🔤 **PaddleOCR PP-OCRv6 Tiny (ONNX Runtime)**:
  - Nạp trực tiếp cặp mô hình cục bộ `PP-OCRv6_tiny_det` và `PP-OCRv6_tiny_rec` qua ONNX Runtime.
  - Tắt hoàn toàn các module nặng không cần thiết (như document unwarping hay full-page orientation) giúp OCR chạy siêu tốc.
- 🧠 **Cơ Chế OCR Cooldown & Khóa Cache Biển Số Việt Nam**:
  - Khi xe mới xuất hiện, hệ thống chạy OCR ngay lập tức.
  - Nếu kết quả đạt chuẩn biển số Việt Nam (`is_valid_vn_plate`), biển số được **khóa vĩnh viễn vào Cache**, các frame tiếp theo không tốn thêm tài nguyên OCR.
  - Nếu chưa đạt chuẩn (xe ở xa, rung lắc hoặc mờ), hệ thống **bỏ qua không OCR trong 2 frame tiếp theo** (cooldown) rồi thử lại ở frame kế tiếp. Quá trình lặp lại cho đến khi nhận diện thành công hoặc xe rời khỏi khung hình.
- 🛡️ **Bộ Lọc Ký Tự & Cấu Trúc Biển Số VN**: Chỉ chấp nhận ký tự in hoa `[A-Z0-9]`, chỉ vẽ lên khung hình đối với các biển số đạt cấu trúc chuẩn Việt Nam (mã tỉnh 11-99: `^(?:1[1-9]|[2-9][0-9])[A-Z][A-Z0-9]?[0-9]{4,5}$`).
- ⏱️ **Công Cụ Benchmark Tích Hợp**: Đo lường và báo cáo chi tiết thời gian load model, latency detector, latency OCR và tốc độ pipeline thực tế.

---

## 🏗️ Luồng Hoạt Động (Architecture Pipeline)

```mermaid
flowchart TD
    Input[Input: Image / Video / RTSP / Camera] --> FrameRead[Frame Reader]
    FrameRead --> SkipCheck{Đến frame nhận diện?<br>frame_idx % skip_frames == 0}

    %% Nhánh chạy Detector
    SkipCheck -- "Có (VD: mỗi 3 frames)" --> YOLO[YOLO NCNN Plate Detector]
    YOLO --> Tracker[ByteTrack Update Tracks]

    %% Nhánh Skip
    SkipCheck -- "Không (Frame bị skip)" --> TrackerPredict[Giữ nguyên vị trí Active Tracks]

    Tracker --> TrackLoop[Duyệt qua từng Track ID]
    TrackerPredict --> TrackLoop

    %% Xử lý OCR Cache
    TrackLoop --> OCRCheck{Track ID đã có trong OCR Cache?}
    OCRCheck -- "Đã có" --> ReuseText[Lấy biển số từ Cache<br>KHÔNG gọi OCR]
    OCRCheck -- "Chưa có (Xe mới)" --> CropPlate[Crop vùng biển số xe]
    CropPlate --> PaddleOCR[PaddleOCR ONNX Runtime<br>PP-OCRv6_tiny]
    PaddleOCR --> FilterText[Bộ lọc nghiêm ngặt<br>Chỉ giữ A-Z và 0-9]
    FilterText --> SaveCache[Lưu vào ocr_cache<br>Khóa vĩnh viễn Track ID]

    ReuseText --> Annotate[Vẽ Bounding Box + Track ID + Biển số]
    SaveCache --> Annotate
    Annotate --> Output[Output: Hiển thị / Xuất Video / Benchmark]
```

---

## 📁 Cấu Trúc Thư Mục

```text
ALPR_app/
├── config.yaml               # File cấu hình tập trung (paths, thresholds, skip-frames)
├── requirements.txt          # Danh sách thư viện tương thích Raspberry Pi 5
├── .gitignore                # Bỏ qua test_data/ và outputs/
├── models/                   # Thư mục chứa các mô hình đã tối ưu
│   ├── lp_det/               # Mô hình YOLO phát hiện biển số
│   │   ├── best_ncnn_model/  # YOLO NCNN model (model.ncnn.bin, model.ncnn.param)
│   │   └── best_lp_det.pt    # PyTorch checkpoint gốc
│   ├── ch_det/               # PaddleOCR PP-OCRv6_tiny_det (ONNX)
│   │   ├── inference.onnx
│   │   └── inference.yml
│   └── ch_rec/               # PaddleOCR PP-OCRv6_tiny_rec (ONNX)
│       ├── inference.onnx
│       └── inference.yml
├── src/
│   ├── __init__.py
│   ├── config.py             # Đọc và xác thực cấu hình config.yaml
│   ├── detector.py           # Wrapper YOLO NCNN qua ultralytics + Warmup
│   ├── tracker.py            # Wrapper ByteTrack (supervision) hỗ trợ skip-frame
│   ├── ocr.py                # Wrapper PaddleOCR ONNX Runtime + Bộ lọc [A-Z0-9]
│   ├── visualizer.py         # Vẽ Bounding Box, badge nhãn biển số và panel FPS
│   └── pipeline.py           # Bộ điều phối luồng ALPR và quản lý OCR Cache
├── scripts/
│   └── test_paddleocr.py     # Script kiểm tra độc lập module OCR
└── main.py                   # Điểm khởi chạy chính (CLI: Image, Video, Camera, Benchmark)
```

---

## ⚙️ Cài Đặt (Installation)

### 1. Chuẩn bị môi trường trên Raspberry Pi 5

Hệ điều hành khuyến nghị: **Raspberry Pi OS (64-bit)** với **Python 3.10**.

Khởi tạo và kích hoạt môi trường ảo (Virtual Environment hoặc Conda/Miniforge):

```bash
# Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Hoặc nếu dùng Conda:
conda create -n alpr python=3.10 -y
conda activate alpr
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Sử Dụng (Usage)

### 1. Kiểm tra độc lập Module OCR

Dùng để kiểm tra nhanh việc nhận diện chữ trên ảnh cắt biển số:

```bash
python scripts/test_paddleocr.py --image test_data/imgs/carlong_0004.png
```

### 2. Nhận diện trên ảnh tĩnh (Image Input)

```bash
python main.py --source test_data/imgs/carlong_0004.png
```

_Kết quả ảnh khoanh vùng biển số sẽ được tự động lưu vào thư mục `outputs/`._

### 3. Nhận diện trên Video (Video Input)

Áp dụng **ByteTrack** + **Skip 3 frames** + **OCR Cache 1-Lần**:

```bash
python main.py --source test_data/videos/demo.mp4 --skip-frames 3
```

- Tiến độ `%` và số lượng biển số thu thập được sẽ cập nhật thời gian thực.
- Video kết quả kèm BBox, Track ID và Text biển số sẽ được xuất ra `outputs/`.

### 4. Nhận diện trực tiếp qua Camera / Webcam / RTSP

- Với Webcam USB hoặc Camera CSI (thiết bị 0):
  ```bash
  python main.py --source 0 --source-name "Cổng Chính"
  ```
- Với Camera IP qua luồng RTSP:
  ```bash
  python main.py --source "rtsp://admin:password@192.168.1.100:554/stream1" --source-name "Camera Tuyến 1"
  ```
- Nếu Pi 5 có kết nối màn hình HDMI trực tiếp, thêm cờ `--show`:
  ```bash
  python main.py --source 0 --show
  ```

---

## 🌐 Khởi Chạy Web Dashboard Tra Cứu Lịch Sử Xe (`run_api.py`)

Hệ thống cung cấp sẵn máy chủ FastAPI tích hợp giao diện Web tĩnh hiện đại, hỗ trợ:
- Tìm kiếm theo biển số xe (hỗ trợ **Tìm kiếm mờ - Fuzzy search** với `rapidfuzz` xử lý lỗi OCR).
- Lọc theo khoảng thời gian, nguồn camera / video.
- Xem ảnh crop cận cảnh biển số, đối chiếu **chuỗi OCR thô** và **chuỗi sau khi sửa lỗi**.
- Xem khung cảnh và **vẽ quỹ đạo di chuyển (trajectory)** của phương tiện trên khung hình.

Khởi chạy máy chủ:

```bash
python scripts/run_api.py --port 8000
```

- Mở trình duyệt truy cập: `http://localhost:8000` (hoặc `http://<IP_Raspberry_Pi>:8000` từ thiết bị cùng mạng LAN).


---

## 🧹 Dọn Dẹp / Xóa Dữ Liệu Trong CSDL (`clean_db.py`)

Khi bạn muốn xóa toàn bộ lịch sử các lượt xe để làm mới hệ thống hoặc test lại từ đầu (nhưng vẫn giữ nguyên cấu trúc bảng và index):

```bash
# Xóa sạch toàn bộ dữ liệu lượt xe và ảnh crop (có hộp thoại hỏi xác nhận):
python scripts/clean_db.py

# Xóa nhanh không cần gõ xác nhận:
python scripts/clean_db.py -y

# Chỉ xóa bảng sightings, giữ lại danh sách Camera/Video đã đăng ký:
python scripts/clean_db.py --keep-sources
```

---

## 📊 Chế Độ Đo Benchmark Chi Tiết (`--benchmark`) & Tắt Lưu Trữ (`--no-db`)

Để kiểm tra chính xác thời gian load model, latency từng module và FPS của hệ thống:

```bash
# Đo benchmark và tắt hoàn toàn việc ghi CSDL SQLite / lưu ảnh crop:
python main.py --source test_data/videos/Recording2026-10-14-211324.mp4 --benchmark --no-db
```

> [!TIP]
> Cờ `--no-db` (hoặc `--no-store`) sẽ tắt toàn bộ luồng ghi ngầm, không tạo kết nối SQLite và không ghi ảnh xuống đĩa. Rất thích hợp khi bạn cần chạy thử nghiệm lặp lại, đo FPS tối đa hoặc test các thuật toán mà không làm phình dung lượng CSDL.

**Mẫu kết quả báo cáo benchmark:**

```text
=================================================================
          BÁO CÁO BENCHMARK CHI TIẾT HỆ THỐNG ALPR
=================================================================
1. THỜI GIAN NẠP & WARM-UP MÔ HÌNH (MODEL LOADING):
   * YOLO NCNN Plate Detector :   142.10 ms
   * PaddleOCR ONNX Runtime   :   480.35 ms
   * Warm-up Models           :   265.80 ms
   -> Tổng thời gian khởi tạo :   888.25 ms
-----------------------------------------------------------------
2. THỜI GIAN SUY LUẬN TỪNG MODULE (INFERENCE TIME):
   * Plate Detection (YOLO NCNN):
       - Số lần chạy      : 34 lần
       - Trung bình / lần :  28.50 ms  (tương đương ~35.1 FPS)
       - Nhanh nhất / Chậm:  24.20 ms /  39.10 ms
   * Character OCR (PaddleOCR ONNX Runtime):
       - Số lần gọi OCR   : 2 lần (áp dụng Cache 1-lần)
       - Trung bình / lần :  61.40 ms
       - Nhanh nhất / Chậm:  59.80 ms /  63.00 ms
-----------------------------------------------------------------
3. HIỆU NĂNG TOÀN BỘ PIPELINE (PIPELINE PERFORMANCE):
   * Thời gian Pipeline thuần (Plate Detection + OCR):
       = 28.50 ms (Detection) + 61.40 ms (OCR) = 89.90 ms
   * Toàn bộ Pipeline thực tế (Detection + Track + OCR + Draw):
       - Tổng số frames xử lý : 100
       - Trung bình mỗi frame :  18.60 ms
       - Tốc độ thực tế       : ~53.8 FPS
       - Nhanh nhất / Chậm    :  11.20 ms /  95.40 ms
       - Tổng thời gian chạy  :   1.86 s
=================================================================
```

---

## 🛠️ Tùy Biến Cấu Hình (`config.yaml`)

| Nhóm         | Tham số               | Giá trị mặc định                | Giải thích                                        |
| ------------ | --------------------- | ------------------------------- | ------------------------------------------------- |
| **detector** | `model_path`          | `models/lp_det/best_ncnn_model` | Đường dẫn thư mục model YOLO NCNN                 |
|              | `conf_threshold`      | `0.35`                          | Ngưỡng tin cậy phát hiện biển số                  |
|              | `imgsz`               | `640`                           | Kích thước ảnh đầu vào detector                   |
| **tracker**  | `skip_frames`         | `3`                             | Số frame bỏ qua giữa các lần chạy YOLO            |
|              | `lost_track_buffer`   | `30`                            | Số frame giữ track khi xe tạm thời bị che khuất   |
| **ocr**      | `engine`              | `onnxruntime`                   | Backend suy luận OCR                              |
|              | `det_model_dir`       | `models/ch_det`                 | Thư mục model text detection ONNX                 |
|              | `retry_interval_frames`| `2`                            | Số frame bỏ qua giữa các lần thử nếu chưa đạt chuẩn VN |
| **output**   | `save_result`         | `true`                          | Tự động lưu video/ảnh kết quả                     |
|              | `output_dir`          | `outputs`                       | Thư mục lưu kết quả                               |

---

## 📜 Giấy Phép (License)

Dự án được phân phối dưới giấy phép **MIT License**. Bạn được tự do sử dụng, chỉnh sửa.

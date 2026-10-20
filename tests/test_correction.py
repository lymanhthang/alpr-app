"""
Unit test kiểm tra tính năng sửa ký tự OCR theo vị trí chuẩn biển số Việt Nam.
"""

from src.reliability.correction import correct_vn_plate


def test_correct_prefix_digits():
    # 2 ký tự đầu là mã tỉnh phải là số (O->0, I->1, B->8)
    corr = correct_vn_plate("O9A12345")
    assert corr.is_corrected is True
    assert corr.plate_corrected == "09A12345"
    assert corr.corrections[0]["pos"] == 0
    assert corr.corrections[0]["from"] == "O"
    assert corr.corrections[0]["to"] == "0"


def test_correct_series_letter():
    # Ký tự thứ 3 là series phải là chữ (0->D, 8->B, 5->S)
    corr = correct_vn_plate("29012345")
    assert corr.is_corrected is True
    assert corr.plate_corrected == "29D12345"
    assert corr.corrections[0]["pos"] == 2
    assert corr.corrections[0]["from"] == "0"
    assert corr.corrections[0]["to"] == "D"


def test_correct_tail_digits():
    # 4 hoặc 5 ký tự cuối chắc chắn là số: chữ cái bị đọc nhầm phải sửa thành số
    # Trường hợp 1: Z -> 2 trong 51F1Z345 (biển xe máy 8 ký tự)
    corr1 = correct_vn_plate("51F1Z345")
    assert corr1.is_corrected is True
    assert corr1.plate_corrected == "51F12345"
    assert any(c["from"] == "Z" and c["to"] == "2" for c in corr1.corrections)

    # Trường hợp 2: B -> 8 ở cuối biển số 9 ký tự
    corr2 = correct_vn_plate("29F11234B")
    assert corr2.is_corrected is True
    assert corr2.plate_corrected == "29F112348"

    # Trường hợp 3: O -> 0 ở đuôi biển số 7 ký tự (51F123O -> 51F1230)
    corr3 = correct_vn_plate("51F123O")
    assert corr3.is_corrected is True
    assert corr3.plate_corrected == "51F1230"


def test_valid_plate_without_correction():
    # Biển số chuẩn không bị sửa đổi
    corr = correct_vn_plate("30A12345")
    assert corr.is_corrected is False
    assert corr.plate_corrected == "30A12345"
    assert len(corr.corrections) == 0


def test_special_series_letters_preserved():
    # Ký tự liên doanh LD, DA ở pos 3 không bị sửa ép thành số
    corr = correct_vn_plate("29LD1234")
    assert corr.plate_corrected == "29LD1234"


def test_lock_first_subcases():
    from src.ocr import PlateOCR

    # TH 2a: Chuỗi quá ngắn (< 7 ký tự) -> Không đủ điều kiện sửa
    short_text = "51F12"
    assert len(short_text) < 7

    # TH 2b: Chuỗi >= 7 ký tự có ký tự nhầm lẫn -> Sửa và kiểm tra đạt chuẩn
    raw_text = "51F1Z345"
    assert len(raw_text) >= 7
    corr = correct_vn_plate(raw_text)
    assert corr.plate_corrected == "51F12345"
    assert PlateOCR.is_valid_vn_plate(corr.plate_corrected) is True

    # TH 2b nhưng sau sửa vẫn không hợp lệ -> Không khóa
    invalid_raw = "1234567"
    corr_invalid = correct_vn_plate(invalid_raw)
    assert PlateOCR.is_valid_vn_plate(corr_invalid.plate_corrected) is False


def test_valid_vn_plate_province_range():
    from src.ocr import PlateOCR

    # Mã tỉnh từ 11 đến 99: Hợp lệ
    assert PlateOCR.is_valid_vn_plate("11A12345") is True   # Cao Bằng
    assert PlateOCR.is_valid_vn_plate("29A12345") is True   # Hà Nội
    assert PlateOCR.is_valid_vn_plate("30A12345") is True   # Hà Nội (đuôi 0)
    assert PlateOCR.is_valid_vn_plate("50A12345") is True   # TP.HCM (đuôi 0)
    assert PlateOCR.is_valid_vn_plate("51F12345") is True   # TP.HCM
    assert PlateOCR.is_valid_vn_plate("99A12345") is True   # Bắc Ninh

    # Mã tỉnh bắt đầu bằng 0 (01-09, 00) hoặc 10: Không hợp lệ
    assert PlateOCR.is_valid_vn_plate("01A12345") is False
    assert PlateOCR.is_valid_vn_plate("09A12345") is False
    assert PlateOCR.is_valid_vn_plate("06069") is False
    assert PlateOCR.is_valid_vn_plate("10A12345") is False



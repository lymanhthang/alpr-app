"""
Module sửa lỗi ký tự OCR theo vị trí chuẩn biển số xe Việt Nam.
Tự động đối chiếu cấu trúc biển số (7, 8, 9 ký tự) và ánh xạ các ký tự bị OCR đọc nhầm:
- Vị trí mã tỉnh (2 ký tự đầu): Chắc chắn là số (O->0, I->1, B->8, Z->2, S->5,...)
- Vị trí series (ký tự thứ 3): Chắc chắn là chữ (0->D, 8->B, 5->S, 1->I, 2->Z,...)
- 4 hoặc 5 ký tự cuối: Chắc chắn là số đăng ký. Mọi chữ cái bị đọc nhầm ở phần đuôi đều được chuyển thành số.
- Đánh dấu chi tiết từng ký tự đã sửa để lưu DB và hiển thị đối chiếu cho người dùng.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Ánh xạ chữ cái bị OCR nhầm sang chữ số
LETTER_TO_DIGIT: Dict[str, str] = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "J": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "B": "8",
    "T": "7",
}

# Ánh xạ chữ số bị OCR nhầm sang chữ cái
DIGIT_TO_LETTER: Dict[str, str] = {
    "0": "D",  # 0 thường nhầm với D (hoặc O) trong series biển số VN
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}


@dataclass
class CorrectionDetail:
    plate_raw: str
    plate_corrected: str
    is_corrected: bool
    corrections: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.corrections, ensure_ascii=False)


def correct_vn_plate(raw_text: Optional[str]) -> CorrectionDetail:
    """
    Sửa ký tự theo vị trí chuẩn của biển số xe Việt Nam (độ dài 7, 8, 9).

    Args:
        raw_text: Chuỗi ký tự biển số thô nhận diện từ OCR (chỉ gồm A-Z0-9).

    Returns:
        CorrectionDetail chứa:
            - plate_raw: Chuỗi gốc
            - plate_corrected: Chuỗi sau hiệu chỉnh
            - is_corrected: True nếu có ít nhất 1 ký tự được sửa
            - corrections: Danh sách chi tiết [{index, from, to, rule}]
    """
    if not raw_text:
        return CorrectionDetail(plate_raw="", plate_corrected="", is_corrected=False)

    # Làm sạch, chỉ giữ ký tự in hoa A-Z và chữ số 0-9
    clean = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    n = len(clean)

    # Biển số chuẩn Việt Nam có độ dài từ 7 đến 9 ký tự
    if n < 7 or n > 9:
        return CorrectionDetail(
            plate_raw=clean,
            plate_corrected=clean,
            is_corrected=False,
            corrections=[],
        )

    chars = list(clean)
    corrections: List[Dict[str, Any]] = []

    # 1. Hai ký tự đầu (Mã tỉnh thành): Bắt buộc là số [0-9]
    for idx in (0, 1):
        ch = chars[idx]
        if ch.isalpha():
            if ch in LETTER_TO_DIGIT:
                target = LETTER_TO_DIGIT[ch]
                chars[idx] = target
                corrections.append({
                    "pos": idx,
                    "from": ch,
                    "to": target,
                    "rule": "mã_tỉnh_phải_là_số",
                })

    # 2. Ký tự thứ 3 (Series đăng ký): Bắt buộc là chữ [A-Z]
    ch_idx2 = chars[2]
    if ch_idx2.isdigit():
        if ch_idx2 in DIGIT_TO_LETTER:
            target = DIGIT_TO_LETTER[ch_idx2]
            chars[2] = target
            corrections.append({
                "pos": 2,
                "from": ch_idx2,
                "to": target,
                "rule": "series_phải_là_chữ",
            })

    # 3. Ký tự phần đuôi (Số đăng ký xe):
    # - Nếu n == 7 (VD: 51F-1234): 4 ký tự cuối (pos 3..6) bắt buộc là số.
    # - Nếu n == 8:
    #     + Nếu pos 3 là số -> biển 5 số ô tô (30A-12345) hoặc 4 số xe máy (29F1-2345). 4 hoặc 5 ký tự cuối là số.
    #     + Trong mọi trường hợp n==8, 4 ký tự cuối (pos 4..7) CHẮC CHẮN là số.
    # - Nếu n == 9 (VD: 29F1-12345): 5 ký tự cuối (pos 4..8) CHẮC CHẮN là số.
    digit_start_idx = 3 if n == 7 else 4

    for idx in range(digit_start_idx, n):
        ch = chars[idx]
        if ch.isalpha():
            if ch in LETTER_TO_DIGIT:
                target = LETTER_TO_DIGIT[ch]
                chars[idx] = target
                corrections.append({
                    "pos": idx,
                    "from": ch,
                    "to": target,
                    "rule": "đuôi_biển_số_phải_là_số",
                })

    # Xử lý trường hợp riêng cho n == 8:
    # Nếu pos 3 là chữ (như O) nhưng pos 4..7 đều là số, và pos 3 đổi thành 0 thì tạo thành biển 5 số hợp lệ
    # (VD: 30AO1234 -> 30A01234)
    if n == 8 and chars[3].isalpha():
        # Nếu là các ký tự series đặc biệt hợp lệ (LD, KT, DA, NG, NN...), giữ nguyên chữ
        # Ngược lại, nếu pos 3 là chữ hay bị nhầm với số (O, I, Z, S...), chuyển thành số nếu đuôi là số
        if chars[3] in ("O", "I", "Z", "S", "B", "D", "Q"):
            old_c = chars[3]
            target = LETTER_TO_DIGIT[old_c]
            chars[3] = target
            corrections.append({
                "pos": 3,
                "from": old_c,
                "to": target,
                "rule": "ký_tự_thứ_4_sửa_thành_số",
            })

    corrected_str = "".join(chars)
    is_corrected = len(corrections) > 0

    return CorrectionDetail(
        plate_raw=clean,
        plate_corrected=corrected_str,
        is_corrected=is_corrected,
        corrections=corrections,
    )

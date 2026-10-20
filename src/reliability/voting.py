"""
Module bỏ phiếu đa frame có trọng số (Weighted Multi-Frame Voting).
Kết hợp bỏ phiếu theo độ dài chuỗi và theo từng vị trí ký tự để chọn ra kết quả biển số chuẩn nhất.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Observation:
    text_raw: str
    text_corrected: str
    is_corrected: bool
    corrections: List[Dict[str, Any]]
    conf: float
    quality: float
    frame_idx: int


@dataclass
class VoteResult:
    plate: str
    plate_raw: str
    is_corrected: bool
    corrections_detail: List[Dict[str, Any]]
    confidence: float
    is_valid: bool
    vote_count: int
    should_early_lock: bool
    lock_reason: str


class PlateVoter:
    """Bộ điều phối bỏ phiếu đa frame cho từng track."""

    @staticmethod
    def is_valid_vn_plate(plate: str) -> bool:
        """Kiểm tra cấu trúc biển số xe Việt Nam hợp lệ (7 - 9 ký tự, mã tỉnh 11-99)."""
        if not plate:
            return False
        pattern = r"^(?:1[1-9]|[2-9][0-9])[A-Z][A-Z0-9]?[0-9]{4,5}$"
        return bool(re.match(pattern, plate))


    @classmethod
    def vote(
        cls,
        observations: List[Observation],
        use_quality: bool = True,
        early_lock_votes: int = 3,
        early_lock_conf: float = 0.85,
    ) -> VoteResult:
        """
        Bỏ phiếu có trọng số trên danh sách các lần quan sát OCR của một track.

        Args:
            observations: Danh sách các lần OCR ghi nhận được.
            use_quality: Có dùng điểm chất lượng crop làm trọng số không.
            early_lock_votes: Số lần đồng thuận tối thiểu để khóa sớm.
            early_lock_conf: Ngưỡng độ tin cậy để khóa sớm.

        Returns:
            VoteResult: Kết quả biển số chiến thắng và trạng thái chốt.
        """
        if not observations:
            return VoteResult(
                plate="",
                plate_raw="",
                is_corrected=False,
                corrections_detail=[],
                confidence=0.0,
                is_valid=False,
                vote_count=0,
                should_early_lock=False,
                lock_reason="no_observations",
            )

        # 1. Tính trọng số cho từng quan sát: w = conf * (quality nếu bật, ngược lại 1.0)
        weights: List[float] = []
        for obs in observations:
            q_factor = obs.quality if use_quality else 1.0
            # Tránh trọng số bằng 0 tuyệt đối
            w = max(0.01, obs.conf * max(0.1, q_factor))
            weights.append(w)

        total_weight = sum(weights)

        # 2. Bỏ phiếu theo cả chuỗi (Full string voting)
        string_weights: Dict[str, float] = defaultdict(float)
        string_obs_map: Dict[str, List[Observation]] = defaultdict(list)

        for obs, w in zip(observations, weights):
            target_str = obs.text_corrected
            string_weights[target_str] += w
            string_obs_map[target_str].append(obs)

        # 3. Bỏ phiếu theo độ dài chuỗi (Length voting)
        length_weights: Dict[int, float] = defaultdict(float)
        for obs, w in zip(observations, weights):
            length_weights[len(obs.text_corrected)] += w

        dominant_len = max(length_weights.keys(), key=lambda l: length_weights[l])

        # 4. Bỏ phiếu theo từng vị trí (Per-position character voting) cho các chuỗi có dominant_len
        char_votes: List[Dict[str, float]] = [defaultdict(float) for _ in range(dominant_len)]
        dom_obs = [obs for obs in observations if len(obs.text_corrected) == dominant_len]
        dom_weights = [w for obs, w in zip(observations, weights) if len(obs.text_corrected) == dominant_len]

        for obs, w in zip(dom_obs, dom_weights):
            for idx, ch in enumerate(obs.text_corrected):
                char_votes[idx][ch] += w

        constructed_chars: List[str] = []
        for idx in range(dominant_len):
            best_ch = max(char_votes[idx].keys(), key=lambda c: char_votes[idx][c])
            constructed_chars.append(best_ch)
        constructed_plate = "".join(constructed_chars)

        # 5. Quyết định chuỗi chiến thắng
        # Ưu tiên chuỗi hợp lệ chuẩn biển số Việt Nam
        best_candidate = ""
        best_cand_weight = -1.0

        # Kiểm tra xem constructed_plate có hợp lệ không
        if cls.is_valid_vn_plate(constructed_plate):
            best_candidate = constructed_plate
            # Tính trọng số gán cho constructed_plate
            best_cand_weight = string_weights.get(constructed_plate, sum(dom_weights) * 0.9)
        else:
            # Nếu chuỗi ghép theo vị trí chưa hợp lệ, tìm trong các chuỗi nguyên bản
            valid_candidates = [
                s for s in string_weights.keys() if cls.is_valid_vn_plate(s)
            ]
            if valid_candidates:
                best_candidate = max(valid_candidates, key=lambda s: string_weights[s])
                best_cand_weight = string_weights[best_candidate]
            else:
                # Nếu không có chuỗi nào hợp lệ, chọn chuỗi có trọng số lớn nhất
                best_candidate = max(string_weights.keys(), key=lambda s: string_weights[s])
                best_cand_weight = string_weights[best_candidate]

        # 6. Tính độ tin cậy chuẩn hóa và tìm thông tin gốc (raw & correction detail)
        conf_score = float(min(1.0, best_cand_weight / total_weight)) if total_weight > 0 else 0.0

        # Lấy quan sát tiêu biểu tương ứng với chuỗi chiến thắng
        matching_obs = string_obs_map.get(best_candidate, [])
        if matching_obs:
            # Chọn quan sát có quality hoặc conf cao nhất
            rep_obs = max(matching_obs, key=lambda o: o.conf * (o.quality if use_quality else 1.0))
            plate_raw = rep_obs.text_raw
            is_corrected = rep_obs.is_corrected
            corrections_detail = rep_obs.corrections
        else:
            # Nếu là constructed_plate ghép chéo từ nhiều obs
            plate_raw = constructed_plate
            is_corrected = False
            corrections_detail = []

        is_valid = cls.is_valid_vn_plate(best_candidate)
        votes_count = len(matching_obs)

        # 7. Kiểm tra điều kiện khóa sớm (Early-lock):
        # Đạt đủ số phiếu tối thiểu, độ tin cậy cao và phải là biển số VN hợp lệ
        should_early_lock = False
        lock_reason = "pending"

        if is_valid and votes_count >= early_lock_votes and conf_score >= early_lock_conf:
            should_early_lock = True
            lock_reason = f"early_lock_consensus_{votes_count}_votes"

        return VoteResult(
            plate=best_candidate,
            plate_raw=plate_raw,
            is_corrected=is_corrected,
            corrections_detail=corrections_detail,
            confidence=conf_score,
            is_valid=is_valid,
            vote_count=votes_count,
            should_early_lock=should_early_lock,
            lock_reason=lock_reason,
        )

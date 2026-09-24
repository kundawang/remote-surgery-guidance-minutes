"""CSA 农场会员会转写与说话人分离合并、说话人角色分类。"""

from typing import Any, Callable, Dict, List, Optional

UNKNOWN_ROLE = "未分类"
UNKNOWN_SPEAKER_TOKENS = {"", "unknown", "none", "null", "未知", "未识别"}


def normalize_speaker(speaker: Any) -> Optional[str]:
    """把说话人标识归一化：None、空串、"unknown" 等统一视为未知（返回 None）。"""
    if speaker is None:
        return None
    if not isinstance(speaker, str):
        speaker = str(speaker)
    token = speaker.strip()
    if token.lower() in UNKNOWN_SPEAKER_TOKENS:
        return None
    return token


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是数值，收到 {value!r}") from exc


def _get_bounds(segment: Dict[str, Any]) -> (float, float):
    start = segment.get("start", segment.get("start_time"))
    end = segment.get("end", segment.get("end_time"))
    start = _as_float(start, "start")
    end = _as_float(end, "end")
    if end < start:
        raise ValueError(f"end({end}) 不能早于 start({start})")
    return start, end


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _split_text(text: str, durations: List[float]) -> List[str]:
    """按各片段时长占比确定性地切分文本（按字符，余数按从左到右分配）。"""
    n = len(text)
    total = sum(durations)
    if not durations or total <= 0 or n == 0:
        return ["" for _ in durations]

    bounds = []
    cumulative = 0.0
    for duration in durations[:-1]:
        cumulative += duration
        bounds.append(int(round(cumulative / total * n)))
    bounds = [max(0, min(n, b)) for b in bounds]

    pieces = []
    cursor = 0
    for boundary in bounds:
        pieces.append(text[cursor:boundary])
        cursor = boundary
    pieces.append(text[cursor:])
    return pieces


def merge_transcript_and_diarization(
    transcript_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把转写段与说话人时间段做多对多归属。

    一条转写段横跨多个说话人时间段时，按重叠区间拆成多条，每条带自己的
    speaker 与对应文本；完全无重叠的转写段保留原文，speaker 为 None。
    归属取舍规则完全确定：重叠总时长优先，平局再按说话人段 (start, end,
    speaker) 升序。
    """
    if not transcript_segments:
        return []

    # 归一化说话人并按 (start, end, speaker) 排序，保证后续平局取舍稳定。
    speaker_windows = []
    for index, dia in enumerate(diarization_segments or []):
        d_start, d_end = _get_bounds(dia)
        speaker = normalize_speaker(dia.get("speaker"))
        speaker_windows.append(
            {
                "start": d_start,
                "end": d_end,
                "speaker": speaker,
                "order": index,
                "sort_key": (d_start, d_end, speaker or ""),
            }
        )
    speaker_windows.sort(key=lambda w: w["sort_key"])

    merged: List[Dict[str, Any]] = []

    for seg in transcript_segments:
        seg_start, seg_end = _get_bounds(seg)
        text = (seg.get("text") or "").strip()

        overlapping = [
            w
            for w in speaker_windows
            if _overlap(seg_start, seg_end, w["start"], w["end"]) > 0
        ]

        if not overlapping:
            merged.append(
                {"start": seg_start, "end": seg_end, "text": text, "speaker": None}
            )
            continue

        # 切片边界：转写段两端 + 落在段内的说话人段边界。
        cuts = {seg_start, seg_end}
        for w in overlapping:
            if seg_start < w["start"] < seg_end:
                cuts.add(w["start"])
            if seg_start < w["end"] < seg_end:
                cuts.add(w["end"])
        boundaries = sorted(cuts)

        # 每个子区间归属：先比该说话人段与整条转写段的重叠总时长，
        # 平局按说话人段的确定性排序键（start, end, speaker）。
        assigned: List[Dict[str, Any]] = []
        for piece_start, piece_end in zip(boundaries, boundaries[1:]):
            candidates = [
                w
                for w in overlapping
                if _overlap(piece_start, piece_end, w["start"], w["end"]) > 0
            ]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda w: (
                        -_overlap(seg_start, seg_end, w["start"], w["end"]),
                        w["sort_key"],
                    ),
                )
                speaker = chosen["speaker"]
            else:
                # 转写段只有部分被覆盖时，未覆盖的子区间保持未知。
                speaker = None
            assigned.append(
                {"start": piece_start, "end": piece_end, "speaker": speaker}
            )

        # 合并相邻且说话人相同（归一化后）的子区间。
        coalesced: List[Dict[str, Any]] = []
        for piece in assigned:
            if coalesced and normalize_speaker(coalesced[-1]["speaker"]) == normalize_speaker(piece["speaker"]):
                coalesced[-1]["end"] = piece["end"]
            else:
                coalesced.append(dict(piece))

        durations = [p["end"] - p["start"] for p in coalesced]
        texts = _split_text(text, durations)
        for piece, piece_text in zip(coalesced, texts):
            merged.append(
                {
                    "start": piece["start"],
                    "end": piece["end"],
                    "text": piece_text.strip(),
                    "speaker": piece["speaker"],
                }
            )

    return merged


def default_role_classifier(speaker: str, sample_text: str) -> str:
    """无 LLM 时的确定性兜底分类，避免静默漏分。"""
    farmer_hints = ("农", "菜", "田", "种植", "土壤", "收成", "采摘", "配送", "肥料")
    consumer_hints = ("会员", "订", "购买", "取", "价格", "套餐", "配送费", "续费")
    farmer_score = sum(1 for h in farmer_hints if h in sample_text)
    consumer_score = sum(1 for h in consumer_hints if h in sample_text)
    if farmer_score > consumer_score:
        return "农人"
    if consumer_score > farmer_score:
        return "消费者"
    return UNKNOWN_ROLE


def classify_speaker_roles(
    segments: List[Dict[str, Any]],
    classify_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, str]:
    """对已归属到具体说话人且文本非空的片段分类（农人/消费者）。

    speaker 为 None、空串、"unknown"（归一化后为未知）以及空文本的片段
    直接跳过，不会产生 None 键，也不会触发分类调用。
    """
    classify = classify_fn or default_role_classifier

    grouped: Dict[str, List[str]] = {}
    for seg in segments or []:
        speaker = normalize_speaker(seg.get("speaker"))
        if speaker is None:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        grouped.setdefault(speaker, []).append(text)

    roles: Dict[str, str] = {}
    for speaker in sorted(grouped):
        roles[speaker] = classify(speaker, " ".join(grouped[speaker]))
    return roles

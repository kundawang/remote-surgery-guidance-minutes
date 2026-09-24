from typing import Iterable, List, Tuple


def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """合并重叠或相邻的时间区间，重叠部分只保留一次。"""
    normalized = sorted(
        (float(start), float(end))
        for start, end in intervals
        if end > start
    )
    if not normalized:
        return []

    merged = [list(normalized[0])]
    for start, end in normalized[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def total_duration(
    intervals: Iterable[Tuple[float, float]],
    audio_duration: float = None,
) -> float:
    """按区间并集统计总时长；提供音频实际长度时结果不超过该长度。"""
    merged = merge_intervals(intervals)
    total = sum(end - start for start, end in merged)
    if audio_duration is not None:
        total = min(total, float(audio_duration))
    return total

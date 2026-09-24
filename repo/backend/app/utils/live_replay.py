"""直播复盘：订单与转写分段对齐、冲量检测及其它复盘指标。"""

from __future__ import annotations

import bisect
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_BURST_WINDOW = 5


def _to_seconds(value: Any) -> float:
    """把时间字段统一换算成秒，支持数值和 "HH:MM:SS" / "MM:SS" 字符串。"""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        numbers = [float(part) for part in text.split(":")]
    except ValueError:
        raise ValueError(f"无法解析时间字段: {value!r}")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def _order_seconds(order: Dict[str, Any]) -> float:
    for key in ("seconds", "time", "timestamp"):
        if order.get(key) is not None:
            return _to_seconds(order[key])
    raise ValueError(f"订单缺少时间字段: {order!r}")


def normalize_orders(orders: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """补全订单的 seconds 字段并按时间升序返回。"""
    normalized = []
    for order in orders:
        item = dict(order)
        item["seconds"] = _order_seconds(item)
        normalized.append(item)
    normalized.sort(key=lambda item: item["seconds"])
    return normalized


def normalize_segments(segments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """补全分段的 start/end（秒）字段并按起点升序返回。"""
    normalized = []
    for segment in segments:
        item = dict(segment)
        start = item.get("start", item.get("start_time"))
        end = item.get("end", item.get("end_time"))
        if start is None or end is None:
            raise ValueError(f"分段缺少 start/end: {segment!r}")
        item["start"] = _to_seconds(start)
        item["end"] = _to_seconds(end)
        if item["end"] < item["start"]:
            raise ValueError(f"分段结束时间早于开始时间: {segment!r}")
        normalized.append(item)
    normalized.sort(key=lambda item: item["start"])
    return normalized


def assign_orders_to_segments(
    orders: Sequence[Dict[str, Any]],
    segments: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """按左闭右开 [start, end) 把订单归属到分段。

    边界订单（t == 段起点）只归到以它为起点的那一段，不再被前一段
    重复计入，因此各段 order_count 之和与订单明细严格一致。
    不落进任何分段的订单放入 unassigned_orders，保证守恒：
    sum(各段 order_count) + len(unassigned_orders) == len(orders)。
    """
    normalized_orders = normalize_orders(orders)
    normalized_segments = normalize_segments(segments)

    buckets: List[List[Dict[str, Any]]] = [[] for _ in normalized_segments]
    unassigned: List[Dict[str, Any]] = []

    starts = [segment["start"] for segment in normalized_segments]
    for order in normalized_orders:
        seconds = order["seconds"]
        candidate = bisect.bisect_right(starts, seconds) - 1
        if (
            0 <= candidate < len(normalized_segments)
            and normalized_segments[candidate]["start"] <= seconds < normalized_segments[candidate]["end"]
        ):
            buckets[candidate].append(order)
        else:
            unassigned.append(order)

    results = []
    for segment, bucket in zip(normalized_segments, buckets):
        revenue = float(sum(float(order.get("amount", order.get("revenue", 0.0))) for order in bucket))
        results.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "order_count": len(bucket),
                "revenue": revenue,
                "orders": bucket,
            }
        )
    return {"segments": results, "unassigned_orders": unassigned}


def detect_bursts(
    points: Optional[Sequence[Any]],
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    window: int = DEFAULT_BURST_WINDOW,
) -> List[Dict[str, Any]]:
    """检测订单量冲量点，基线只取当前点之前 window 个点的均值/标准差。

    - 当前点不参与自己的基线，峰值不会再把基线拉高而漏报。
    - 首点没有历史数据，直接跳过，不产出虚假 z 值。
    - 历史标准差为 0 时不除零：当前点高于历史均值才报，z_score 记为 inf。
    - 返回字段：timestamp / seconds / order_count / z_score / expected。
    """
    if not points or len(points) <= 1:
        return []

    series = []
    for point in points:
        if isinstance(point, dict):
            timestamp = point.get("timestamp")
            seconds = point.get("seconds")
            if seconds is None and timestamp is not None:
                seconds = _to_seconds(timestamp)
            series.append(
                {
                    "timestamp": timestamp,
                    "seconds": seconds,
                    "order_count": float(point.get("order_count", point.get("count", 0))),
                }
            )
        else:
            series.append({"timestamp": None, "seconds": None, "order_count": float(point)})

    if window is None or window <= 0:
        window = len(series)

    alerts = []
    for index, current in enumerate(series):
        start = max(0, index - window)
        history = [item["order_count"] for item in series[start:index]]
        if not history:
            continue

        expected = statistics.fmean(history)
        deviation = statistics.pstdev(history)
        value = current["order_count"]

        if deviation > 0:
            z_score = (value - expected) / deviation
            if z_score < z_threshold:
                continue
        elif value > expected:
            z_score = float("inf")
        else:
            continue

        alerts.append(
            {
                "timestamp": current["timestamp"],
                "seconds": current["seconds"],
                "order_count": int(value) if value == int(value) else value,
                "z_score": z_score,
                "expected": expected,
            }
        )
    return alerts


def build_replay_summary(
    orders: Sequence[Dict[str, Any]],
    viewers: Optional[Sequence[Any]] = None,
    visitors: Optional[int] = None,
) -> Dict[str, Any]:
    """汇总其它复盘指标（总量、客单价、转化率、峰值观众，算法保持不变）。"""
    normalized_orders = normalize_orders(orders)
    total_orders = len(normalized_orders)
    total_revenue = float(
        sum(float(order.get("amount", order.get("revenue", 0.0))) for order in normalized_orders)
    )
    average_order_value = total_revenue / total_orders if total_orders else 0.0

    peak_viewers = 0
    if viewers:
        peak_viewers = max(
            item if isinstance(item, (int, float)) else int(item.get("viewers", item.get("viewer_count", 0)))
            for item in viewers
        )

    conversion_rate = (total_orders / visitors) if visitors else 0.0

    return {
        "total_orders": total_orders,
        "total_revenue": total_revenue,
        "average_order_value": average_order_value,
        "conversion_rate": conversion_rate,
        "peak_viewers": peak_viewers,
    }

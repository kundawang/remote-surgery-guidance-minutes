"""直播复盘回归测试：分段边界归属 + 冲量检测基线。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.live_replay import (
    DEFAULT_Z_THRESHOLD,
    DEFAULT_BURST_WINDOW,
    assign_orders_to_segments,
    build_replay_summary,
    detect_bursts,
)


def make_orders(times, amounts=None):
    amounts = amounts or [100.0] * len(times)
    return [{"timestamp": t, "amount": amount} for t, amount in zip(times, amounts)]


class SegmentBoundaryTests(unittest.TestCase):
    def test_boundary_order_counts_only_once(self):
        # 用户复现场景：[0,100) 与 [100,200)，t=100 只能归到第二段
        orders = make_orders([10, 50, 99, 100, 150])
        segments = [
            {"start": 0, "end": 100},
            {"start": 100, "end": 200},
        ]

        report = assign_orders_to_segments(orders, segments)

        counts = [segment["order_count"] for segment in report["segments"]]
        self.assertEqual(counts, [3, 2])
        self.assertEqual(sum(counts), len(orders))
        self.assertEqual(report["unassigned_orders"], [])
        boundary_order = report["segments"][1]["orders"][0]
        self.assertEqual(boundary_order["seconds"], 100.0)

    def test_segment_totals_equal_detail_totals(self):
        # 含多个边界点：0、100、200，以及刚好落在最后一段末尾 300 的订单
        times = [0, 25, 100, 100, 199, 200, 250, 300]
        orders = make_orders(times, amounts=range(1, len(times) + 1))
        segments = [
            {"start": 0, "end": 100},
            {"start": 100, "end": 200},
            {"start": 200, "end": 300},
        ]

        report = assign_orders_to_segments(orders, segments)

        counts = [segment["order_count"] for segment in report["segments"]]
        self.assertEqual(counts, [2, 3, 2])
        self.assertEqual(sum(counts), 7)
        self.assertEqual(sum(counts) + len(report["unassigned_orders"]), len(orders))

        segment_revenue = sum(segment["revenue"] for segment in report["segments"])
        unassigned_revenue = sum(float(order.get("amount", 0.0)) for order in report["unassigned_orders"])
        self.assertAlmostEqual(segment_revenue + unassigned_revenue, float(sum(range(1, 9))))

        # t=300 是所有分段的右端点，左闭右开下不归任何段，保证不重复计数
        self.assertEqual([order["seconds"] for order in report["unassigned_orders"]], [300.0])

    def test_invariant_random_orders(self):
        import random

        random.seed(7)
        segments = [
            {"start": 0, "end": 10},
            {"start": 10, "end": 25},
            {"start": 25, "end": 25},
            {"start": 30, "end": 40},
        ]
        times = [random.randint(0, 45) for _ in range(500)]
        report = assign_orders_to_segments(make_orders(times), segments)

        counted = sum(segment["order_count"] for segment in report["segments"])
        self.assertEqual(counted + len(report["unassigned_orders"]), len(times))
        self.assertLessEqual(counted, len(times))


class BurstDetectionTests(unittest.TestCase):
    def test_peak_does_not_pollute_its_own_baseline(self):
        # 用户复现场景：[20]*10 + [400]，基线只看当前点之前的数据
        points = [20] * 10 + [400]
        alerts = detect_bursts(points)

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(
            set(alert.keys()), {"timestamp", "seconds", "order_count", "z_score", "expected"}
        )
        self.assertEqual(alert["order_count"], 400)
        self.assertEqual(alert["expected"], 20.0)
        self.assertEqual(alert["z_score"], float("inf"))

    def test_first_point_skipped_without_history(self):
        self.assertEqual(detect_bursts([400]), [])
        self.assertEqual(detect_bursts([400, 1, 1]), [])

    def test_empty_and_single_point(self):
        self.assertEqual(detect_bursts([]), [])
        self.assertEqual(detect_bursts([10]), [])

    def test_constant_series_no_division_by_zero(self):
        self.assertEqual(detect_bursts([20] * 11), [])

    def test_zero_stddev_then_spike_reports_infinite_z(self):
        alerts = detect_bursts([5, 5, 5, 5, 9])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["order_count"], 9)
        self.assertEqual(alerts[0]["z_score"], float("inf"))
        self.assertEqual(alerts[0]["expected"], 5.0)

    def test_lookback_baseline_excludes_current_point(self):
        # 用一个有限标准差的序列验证基线确实不含当前点（前面各点都不达标）
        points = [18, 20, 22, 20, 60]
        alerts = detect_bursts(points, z_threshold=2.0, window=5)
        history = points[:-1]
        mean = sum(history) / len(history)
        variance = sum((value - mean) ** 2 for value in history) / len(history)
        expected_z = (60 - mean) / variance ** 0.5
        peak_alerts = [alert for alert in alerts if alert["order_count"] == 60]
        self.assertEqual(len(peak_alerts), 1)
        self.assertAlmostEqual(peak_alerts[0]["z_score"], expected_z)
        self.assertGreater(expected_z, 2.0)

    def test_threshold_semantics_unchanged(self):
        self.assertEqual(DEFAULT_Z_THRESHOLD, 2.0)
        self.assertEqual(DEFAULT_BURST_WINDOW, 5)
        # history=[4,8,4,8]：mean=6, pstdev=2；当前点 10 -> z=2（== 阈值仍报）
        at_threshold = [4, 8, 4, 8, 10]
        alerts = detect_bursts(at_threshold, z_threshold=2.0, window=5)
        boundary_alerts = [alert for alert in alerts if alert["order_count"] == 10]
        self.assertEqual(len(boundary_alerts), 1)
        self.assertAlmostEqual(boundary_alerts[0]["z_score"], 2.0)
        # 当前点 9 -> z=1.5，不报
        below_threshold = detect_bursts([4, 8, 4, 8, 9], z_threshold=2.0, window=5)
        self.assertNotIn(9, [alert["order_count"] for alert in below_threshold])

    def test_dict_points_keep_fields(self):
        points = [
            {"timestamp": "00:00:00", "seconds": 0, "order_count": 20},
            {"timestamp": "00:00:10", "seconds": 10, "order_count": 20},
            {"timestamp": "00:00:20", "seconds": 20, "order_count": 20},
            {"timestamp": "00:00:30", "seconds": 30, "order_count": 20},
            {"timestamp": "00:00:40", "seconds": 40, "order_count": 20},
            {"timestamp": "00:00:50", "seconds": 50, "order_count": 400},
        ]
        alerts = detect_bursts(points)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["timestamp"], "00:00:50")
        self.assertEqual(alerts[0]["seconds"], 50)
        self.assertEqual(alerts[0]["order_count"], 400)

    def test_window_only_uses_prior_points(self):
        # window=1 时，每个点只和前一个点比；常量序列末尾跳一次应立即报
        alerts = detect_bursts([1, 1, 1, 1, 1, 30], window=1)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["expected"], 1.0)
        self.assertEqual(alerts[0]["order_count"], 30)

    def test_no_exception_on_none_and_equal_edges(self):
        self.assertEqual(detect_bursts(None), [])
        self.assertEqual(detect_bursts([0, 0, 0]), [])
        self.assertEqual(detect_bursts([0, 0, 1])[0]["z_score"], float("inf"))


class OtherMetricsTests(unittest.TestCase):
    def test_summary_metrics_unchanged(self):
        orders = make_orders([1, 2, 3], amounts=[100.0, 200.0, 300.0])
        summary = build_replay_summary(
            orders,
            viewers=[{"seconds": 1, "viewers": 10}, {"seconds": 2, "viewers": 99}],
            visitors=6,
        )
        self.assertEqual(summary["total_orders"], 3)
        self.assertEqual(summary["total_revenue"], 600.0)
        self.assertEqual(summary["average_order_value"], 200.0)
        self.assertEqual(summary["conversion_rate"], 0.5)
        self.assertEqual(summary["peak_viewers"], 99)

    def test_summary_empty(self):
        summary = build_replay_summary([])
        self.assertEqual(summary["total_orders"], 0)
        self.assertEqual(summary["total_revenue"], 0.0)
        self.assertEqual(summary["average_order_value"], 0.0)
        self.assertEqual(summary["conversion_rate"], 0.0)
        self.assertEqual(summary["peak_viewers"], 0)


if __name__ == "__main__":
    unittest.main()

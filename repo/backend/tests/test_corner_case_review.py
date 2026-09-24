"""corner case 评审回归测试。

覆盖两类历史缺陷：
1. 仿真 id 使用内建 hash()，同一 corner case 跨进程生成不同 id；
2. 分析失败被包装成 severity=unknown 的假 corner case，污染报告统计。
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.corner_case_review import (  # noqa: E402
    CornerCaseReviewService,
    HttpResponse,
    generate_local_simulation_id,
    generate_simulation_id,
    is_local_simulation_id,
)

CASE = {
    "description": "左侧车辆强行加塞",
    "timestamp": "10:23:41",
    "category": "cut-in",
    "severity": "high",
}


def make_service(analyzer=None, http_client=None, platform_url="http://sim.local"):
    return CornerCaseReviewService(
        analyzer=analyzer,
        http_client=http_client,
        simulation_platform_url=platform_url,
    )


class StubHttpClient:
    """可编程的仿真平台 stub。"""

    def __init__(self, post_response=None, post_exc=None, get_response=None):
        self.post_response = post_response
        self.post_exc = post_exc
        self.get_response = get_response
        self.posts = []
        self.gets = []

    def post(self, url, payload):
        self.posts.append((url, payload))
        if self.post_exc:
            raise self.post_exc
        return self.post_response

    def get(self, url):
        self.gets.append(url)
        return self.get_response


class SimulationIdTests(unittest.TestCase):
    def test_same_case_twice_same_id(self):
        """同名场景两次调用 id 一致。"""
        id1 = generate_simulation_id(CASE["description"], CASE["timestamp"], CASE["category"])
        id2 = generate_simulation_id(CASE["description"], CASE["timestamp"], CASE["category"])
        self.assertEqual(id1, id2)
        self.assertRegex(id1, r"^SIM_1\.5_\d{4}$")

    def test_id_stable_across_processes(self):
        """不同 PYTHONHASHSEED 的子进程生成相同 id（回归内建 hash 缺陷）。"""
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from app.services.corner_case_review import generate_simulation_id;"
            "print(generate_simulation_id('左侧车辆强行加塞','10:23:41','cut-in'))"
            % os.path.join(os.path.dirname(__file__), "..")
        )
        ids = set()
        for seed in ("0", "1", "42", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            out = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, env=env, check=True,
            )
            ids.add(out.stdout.strip())
        self.assertEqual(len(ids), 1, f"跨进程 id 不一致: {ids}")

    def test_different_description_different_id(self):
        id_a = generate_simulation_id("左侧车辆强行加塞", "10:23:41", "cut-in")
        id_b = generate_simulation_id("右侧车辆强行加塞", "10:23:41", "cut-in")
        id_c = generate_simulation_id("左侧车辆强行加塞", "10:23:42", "cut-in")
        self.assertNotEqual(id_a, id_b)
        self.assertNotEqual(id_a, id_c)

    def test_local_and_remote_ids_do_not_collide(self):
        local_id = generate_local_simulation_id(CASE["description"], CASE["timestamp"], CASE["category"])
        remote_id = generate_simulation_id(CASE["description"], CASE["timestamp"], CASE["category"])
        self.assertTrue(local_id.startswith("LOCAL_SIM_1.5_"))
        self.assertTrue(is_local_simulation_id(local_id))
        self.assertFalse(is_local_simulation_id(remote_id))
        self.assertNotEqual(local_id, remote_id)


class AnalyzeFailureTests(unittest.TestCase):
    def test_analyzer_timeout_returns_failure_not_fake_case(self):
        """模型超时：返回失败状态与错误原因，不产生假 corner case。"""
        def timeout_analyzer(_text):
            raise TimeoutError("model request timed out")

        result = make_service(analyzer=timeout_analyzer).analyze_corner_cases("任意文本")
        self.assertFalse(result["success"])
        self.assertEqual(result["corner_cases"], [])
        self.assertEqual(result["count"], 0)
        self.assertIn("分析失败", result["error"])
        self.assertIn("timed out", result["error"])
        # 不存在被包装成 corner case 的报错
        for case in result["corner_cases"]:
            self.assertNotEqual(case.get("severity"), "unknown")
            self.assertNotIn("分析失败", case.get("description", ""))

    def test_malformed_model_output_returns_failure(self):
        """模型返回格式不对：失败而非假场景。"""
        for bad in ({"not": "a list"}, [{"description": "缺字段"}], ["string-item"]):
            result = make_service(analyzer=lambda _t, b=bad: b).analyze_corner_cases("文本")
            self.assertFalse(result["success"], bad)
            self.assertEqual(result["corner_cases"], [])
            self.assertEqual(result["count"], 0)
            self.assertIn("分析失败", result["error"])

    def test_successful_empty_result_is_not_failure(self):
        """分析成功但结果为空 ≠ 分析失败。"""
        result = make_service(analyzer=lambda _t: []).analyze_corner_cases("一切正常")
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["error"])

    def test_successful_cases_get_stable_ids(self):
        analyzer = lambda _t: [dict(CASE)]  # noqa: E731
        svc = make_service(analyzer=analyzer)
        r1 = svc.analyze_corner_cases("x")
        r2 = svc.analyze_corner_cases("x")
        self.assertTrue(r1["success"])
        self.assertEqual(
            r1["corner_cases"][0]["simulation_id"],
            r2["corner_cases"][0]["simulation_id"],
        )

    def test_report_counts_only_real_cases(self):
        """报告统计只含真实场景，失败信息单独提示。"""
        svc = make_service(analyzer=lambda _t: [dict(CASE)])
        ok = svc.analyze_corner_cases("x")
        failed = make_service(
            analyzer=lambda _t: (_ for _ in ()).throw(TimeoutError("boom"))
        ).analyze_corner_cases("x")

        report_ok = svc.generate_markdown_report(ok)
        self.assertIn("共识别到 1 个极端场景", report_ok)
        self.assertNotIn("分析告警", report_ok)

        report_failed = svc.generate_markdown_report(failed)
        self.assertIn("共识别到 0 个极端场景", report_failed)
        self.assertIn("分析告警", report_failed)
        self.assertIn("boom", report_failed)
        # 失败不应被统计成场景
        self.assertNotIn("severity=unknown", report_failed)
        summary = svc.generate_meeting_summary(failed)
        self.assertEqual(summary["total_corner_cases"], 0)
        self.assertFalse(summary["analysis_succeeded"])
        self.assertIsNotNone(summary["analysis_error"])


class SimulationPlatformTests(unittest.TestCase):
    def test_remote_success_uses_platform_id(self):
        http = StubHttpClient(
            post_response=HttpResponse(200, '{"simulation_id": "SIM_1.5_9001"}')
        )
        svc = make_service(http_client=http)
        result = svc.create_simulation_test(CASE)
        self.assertTrue(result["success"])
        self.assertTrue(result["submitted"])
        self.assertEqual(result["source"], "remote")
        self.assertEqual(result["simulation_id"], "SIM_1.5_9001")
        self.assertEqual(result["status_code"], 200)
        # 幂等重试凭据：携带稳定 client_reference_id
        self.assertEqual(
            http.posts[0][1]["client_reference_id"],
            generate_simulation_id(CASE["description"], CASE["timestamp"], CASE["category"]),
        )

    def test_remote_non_200_not_silent_local_success(self):
        """远端非 200：不静默降级，带状态码与错误信息。"""
        http = StubHttpClient(
            post_response=HttpResponse(500, '{"error": "internal"}')
        )
        result = make_service(http_client=http).create_simulation_test(CASE)
        self.assertFalse(result["success"])
        self.assertFalse(result["submitted"])
        self.assertEqual(result["status_code"], 500)
        self.assertIn("500", result["error"])
        self.assertIn("internal", result["error"])
        self.assertIsNone(result["simulation_id"])

    def test_platform_unreachable_marks_local_only(self):
        """平台不可达：本地兜底 id 带 LOCAL_ 前缀，报告可区分。"""
        http = StubHttpClient(post_exc=ConnectionError("connection refused"))
        svc = make_service(http_client=http)
        result = svc.create_simulation_test(CASE)
        self.assertTrue(result["success"])
        self.assertFalse(result["submitted"])
        self.assertEqual(result["source"], "local")
        self.assertTrue(result["simulation_id"].startswith("LOCAL_"))
        self.assertIn("仅本地创建", result["error"])

        report = svc.generate_markdown_report(
            {"success": True, "corner_cases": [dict(CASE)], "count": 1, "error": None},
            simulation_results=[result],
        )
        self.assertIn("仅本地创建: 1 个", report)
        self.assertIn("已提交到仿真平台: 0 个", report)

    def test_get_test_status_local_id_skips_remote(self):
        http = StubHttpClient()
        svc = make_service(http_client=http)
        local_id = generate_local_simulation_id("d", "t", "c")
        status = svc.get_test_status(local_id)
        self.assertTrue(status["success"])
        self.assertEqual(status["status"], "local_only")
        self.assertFalse(status["submitted"])
        self.assertEqual(http.gets, [])  # 本地 id 不请求远端

    def test_get_test_status_remote_non_200(self):
        http = StubHttpClient(get_response=HttpResponse(404, "not found"))
        status = make_service(http_client=http).get_test_status("SIM_1.5_9001")
        self.assertFalse(status["success"])
        self.assertEqual(status["status_code"], 404)
        self.assertIn("404", status["error"])


if __name__ == "__main__":
    unittest.main()

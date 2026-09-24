"""回归测试：重复命中去重、最长匹配优先、兜底摘要不含过期日期、结果确定性。"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.pipeline_conflict_extractor import (  # noqa: E402
    COUNTING_RULE,
    BurialDepthExtractor,
    PipelineConflictExtractor,
    PipelineTypeExtractor,
    StakeNumberExtractor,
    generate_fallback_summary,
)


class StakeNumberDedupTest(unittest.TestCase):
    def setUp(self):
        self.extractor = StakeNumberExtractor()

    def test_same_stake_matched_by_multiple_patterns_counts_once(self):
        # "桩号K12+300" 同时被带前缀正则和裸桩号正则命中，只能算一条
        results = self.extractor.extract("桩号K12+300处需迁改")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].value, "K12+300")
        self.assertEqual(results[0].position, 2)

    def test_stake_number_normalized_with_k_prefix_and_uppercase(self):
        results = self.extractor.extract("k12+300与K12+300.5两处")
        values = [r.value for r in results]
        self.assertEqual(values, ["K12+300", "K12+300.5"])

    def test_stake_number_missing_k_prefix_gets_normalized(self):
        results = self.extractor.extract("桩号：12+300")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].value, "K12+300")


class PipelineTypeDedupTest(unittest.TestCase):
    def setUp(self):
        self.extractor = PipelineTypeExtractor()

    def test_contained_keywords_count_once(self):
        # "燃气管" 命中后，"燃气" 不得重复计入；通用正则命中同一处也去重
        results = self.extractor.extract("燃气管与光缆交叉")
        values = [r.value for r in results]
        self.assertEqual(values, ["燃气", "通信"])

    def test_generic_pattern_does_not_double_count(self):
        results = self.extractor.extract("燃气管道需迁改")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].value, "燃气")

    def test_position_preserved(self):
        text = "给水管在燃气管北侧"
        results = self.extractor.extract(text)
        for entity in results:
            self.assertTrue(text.startswith(entity.raw, entity.position))


class BurialDepthDedupTest(unittest.TestCase):
    def test_same_depth_matched_once(self):
        results = BurialDepthExtractor().extract("埋深1.5米")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].value, "1.5m")


class ConflictExtractionTest(unittest.TestCase):
    def setUp(self):
        self.extractor = PipelineConflictExtractor()

    def test_output_schema_and_confidence(self):
        record = self.extractor.extract_conflict(
            "桩号K12+300处燃气管与给水管交叉，埋深1.5米，需燃气公司确认",
            speaker="张工",
            organization="施工单位",
            timestamp="2026-09-24T10:00:00",
        )
        self.assertEqual(
            set(record),
            {
                "description", "speaker", "organization", "stake_numbers",
                "burial_depths", "pipeline_types", "mentioned_organizations",
                "confidence", "timestamp",
            },
        )
        self.assertEqual(len(record["stake_numbers"]), 1)
        self.assertEqual(
            sorted(e["value"] for e in record["pipeline_types"]), ["燃气", "给水"]
        )
        self.assertEqual(record["confidence"], 0.9)  # 0.5 + 4 类实体 * 0.1
        self.assertEqual(record["timestamp"], "2026-09-24T10:00:00")

    def test_one_segment_multiple_types_counts_as_one_conflict(self):
        segments = [
            {"text": "K12+300处燃气管、给水管、电缆三线交叉", "speaker": "张工"},
            {"text": "今天天气不错", "speaker": "李工"},
        ]
        conflicts = self.extractor.extract_conflicts(segments)
        self.assertEqual(len(conflicts), 1)
        summary = self.extractor.summarize(conflicts)
        self.assertEqual(summary["total_conflicts"], 1)
        self.assertEqual(summary["counting_rule"], COUNTING_RULE)
        self.assertEqual(
            summary["pipeline_type_counts"], {"电力": 1, "燃气": 1, "给水": 1}
        )

    def test_deterministic_across_runs(self):
        segments = [
            {"text": "桩号k12+300燃气管迁改，埋深1.5米，燃气公司配合"},
            {"text": "K12+300给水管与通信光缆交叉"},
        ]
        first = self.extractor.extract_conflicts(segments)
        second = PipelineConflictExtractor().extract_conflicts(segments)
        self.assertEqual(json.dumps(first, ensure_ascii=False, sort_keys=True),
                         json.dumps(second, ensure_ascii=False, sort_keys=True))


class FallbackSummaryTest(unittest.TestCase):
    def test_no_hardcoded_date_when_meeting_date_missing(self):
        summary = generate_fallback_summary([], meeting_date=None)
        payload = json.dumps(summary, ensure_ascii=False)
        self.assertNotIn("2024-02-15", payload)
        self.assertNotIn("2024", payload)
        for item in summary["action_items"]:
            self.assertEqual(item["deadline"], "待确定")

    def test_deadline_derived_from_meeting_date(self):
        summary = generate_fallback_summary([], meeting_date="2026-09-24")
        for item in summary["action_items"]:
            self.assertEqual(item["deadline"], "2026-10-01")

    def test_unparseable_meeting_date_falls_back_to_pending(self):
        summary = generate_fallback_summary([], meeting_date="另行通知")
        for item in summary["action_items"]:
            self.assertEqual(item["deadline"], "待确定")


if __name__ == "__main__":
    unittest.main()

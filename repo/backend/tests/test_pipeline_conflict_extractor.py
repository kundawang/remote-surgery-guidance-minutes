"""地下管线迁改冲突抽取的回归测试。

覆盖：
- 重复命中去重（多正则/多关键词命中同一处文字只计一次）；
- 最长匹配优先（"燃气"/"燃气管" 不重复计数）；
- 桩号归一化（K 前缀、大小写一致）；
- 兜底摘要不含写死的过期日期；
- 统计口径随结果返回、同一份输入重复运行结果一致。
"""
import unittest

from app.services.pipeline_conflict_extractor import (
    COUNTING_RULE,
    BurialDepthExtractor,
    PipelineConflictExtractor,
    PipelineTypeExtractor,
    StakeNumberExtractor,
    derive_deadline,
    generate_fallback_summary,
    normalize_stake_number,
)

EXPECTED_KEYS = {
    "description",
    "speaker",
    "organization",
    "stake_numbers",
    "burial_depths",
    "pipeline_types",
    "mentioned_organizations",
    "confidence",
    "timestamp",
}


class StakeNumberDedupTest(unittest.TestCase):
    def setUp(self):
        self.extractor = StakeNumberExtractor()

    def test_same_stake_hit_by_multiple_patterns_counts_once(self):
        # "桩号：K12+300" 同时命中带前缀正则和裸桩号正则，只能算一条
        entities = self.extractor.extract("冲突点位于桩号：K12+300 处")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].value, "K12+300")

    def test_stake_number_normalized_with_k_prefix_and_uppercase(self):
        self.assertEqual(normalize_stake_number("k12+300"), "K12+300")
        self.assertEqual(normalize_stake_number("12+300"), "K12+300")
        self.assertEqual(normalize_stake_number(" K12 + 300 "), "K12+300")

    def test_same_position_same_stake_deduped_after_normalization(self):
        # 同一位置小写 k 命中，归一化后与大写一致，只保留一条
        entities = self.extractor.extract("k12+300")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].value, "K12+300")

    def test_position_is_preserved(self):
        text = "北侧K3+200与南侧K5+400"
        entities = self.extractor.extract(text)
        self.assertEqual([e.value for e in entities], ["K3+200", "K5+400"])
        for entity in entities:
            self.assertEqual(
                text[entity.position:entity.position + len(entity.raw)],
                entity.raw,
            )


class PipelineTypeDedupTest(unittest.TestCase):
    def setUp(self):
        self.extractor = PipelineTypeExtractor()

    def test_contained_keywords_longest_match_wins(self):
        # "燃气管" 命中后 "燃气" 不得重复计入；同一处冲突只出一条
        entities = self.extractor.extract("此处燃气管与新建基础冲突")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].value, "燃气")
        self.assertEqual(entities[0].raw, "燃气管")

    def test_keyword_and_generic_regex_not_double_counted(self):
        # 关键词 "燃气管道" 与通用 "XX管道" 正则命中同一段文字，只计一次
        entities = self.extractor.extract("需迁改燃气管道约 30 米")
        gas = [e for e in entities if e.value == "燃气"]
        self.assertEqual(len(gas), 1)

    def test_same_type_same_position_counted_once(self):
        entities = self.extractor.extract("燃气、燃气管、燃气管道均在此交汇")
        # 三处不同位置的提及各自保留（位置不同），但每处只算一条
        self.assertTrue(all(e.value == "燃气" for e in entities))
        positions = [e.position for e in entities]
        self.assertEqual(len(positions), len(set(positions)))
        self.assertEqual(len(entities), 3)

    def test_company_name_not_pipeline_type(self):
        entities = self.extractor.extract("请燃气公司确认管线资料")
        self.assertEqual([e for e in entities if e.value == "燃气"], [])

    def test_position_is_preserved(self):
        text = "给水管与电缆交叉"
        entities = self.extractor.extract(text)
        for entity in entities:
            self.assertEqual(
                text[entity.position:entity.position + len(entity.raw)],
                entity.raw,
            )


class BurialDepthDedupTest(unittest.TestCase):
    def test_same_depth_not_double_counted(self):
        entities = BurialDepthExtractor().extract("燃气管埋深1.5米")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].value, "1.5m")


class ConflictExtractionTest(unittest.TestCase):
    def setUp(self):
        self.extractor = PipelineConflictExtractor()

    def test_output_schema_unchanged(self):
        record = self.extractor.extract_conflict(
            description="K12+300 处燃气管与给水管冲突，埋深1.5米，请燃气公司核实",
            speaker="张工",
            organization="市政设计院",
            timestamp="00:12:30",
        )
        self.assertEqual(set(record.keys()), EXPECTED_KEYS)
        self.assertEqual(record["speaker"], "张工")
        self.assertEqual(record["organization"], "市政设计院")
        self.assertEqual(record["timestamp"], "00:12:30")
        for field in ("stake_numbers", "burial_depths", "pipeline_types", "mentioned_organizations"):
            for entity in record[field]:
                self.assertEqual(set(entity.keys()), {"value", "raw", "position"})

    def test_confidence_is_deterministic_and_bounded(self):
        text = "K12+300 处燃气管冲突，埋深1.5米，燃气公司核实"
        first = self.extractor.extract_conflict(description=text)
        second = self.extractor.extract_conflict(description=text)
        self.assertEqual(first["confidence"], second["confidence"])
        self.assertGreaterEqual(first["confidence"], 0.5)
        self.assertLessEqual(first["confidence"], 0.95)

    def test_one_segment_counts_as_one_conflict(self):
        segments = [
            {"text": "K12+300 处燃气管、给水管、电缆三线交汇冲突", "speaker": "张工"},
            {"text": "今天天气不错", "speaker": "李工"},
        ]
        conflicts = self.extractor.extract_conflicts(segments)
        # 同一段发言命中多种管线类型仍只计 1 处冲突；无命中段不计
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(conflicts[0]["pipeline_types"]), 3)

    def test_repeated_runs_give_identical_results(self):
        segments = [
            {"text": "桩号：K12+300 燃气管与给水管冲突，埋深1.5米", "speaker": "张工"},
            {"text": "K15+800 电缆迁改，请供电公司确认", "speaker": "李工"},
        ]
        first = self.extractor.extract_conflicts(segments)
        second = PipelineConflictExtractor().extract_conflicts(segments)
        self.assertEqual(first, second)

    def test_summary_states_counting_rule(self):
        segments = [
            {"text": "K12+300 燃气管与给水管冲突", "speaker": "张工"},
            {"text": "K15+800 电缆迁改", "speaker": "李工"},
        ]
        conflicts = self.extractor.extract_conflicts(segments)
        summary = self.extractor.summarize(conflicts)
        self.assertEqual(summary["total_conflicts"], 2)
        self.assertEqual(summary["counting_rule"], COUNTING_RULE)
        self.assertEqual(summary["unique_stake_numbers"], ["K12+300", "K15+800"])
        self.assertEqual(
            summary["pipeline_type_counts"],
            {"电力": 1, "燃气": 1, "给水": 1},
        )


class FallbackSummaryTest(unittest.TestCase):
    def setUp(self):
        self.extractor = PipelineConflictExtractor()
        self.conflicts = self.extractor.extract_conflicts(
            [{"text": "K12+300 燃气管冲突，请燃气公司核实", "speaker": "张工"}]
        )

    def test_no_hardcoded_expired_date(self):
        summary = generate_fallback_summary(self.conflicts)
        deadlines = [item["deadline"] for item in summary["action_items"]]
        self.assertNotIn("2024-02-15", deadlines)
        # 没有真实会议日期时一律 "待确定"
        self.assertTrue(all(d == "待确定" for d in deadlines))

    def test_deadline_derived_from_meeting_date(self):
        summary = generate_fallback_summary(self.conflicts, meeting_date="2026-09-24")
        deadlines = {item["deadline"] for item in summary["action_items"]}
        self.assertEqual(deadlines, {"2026-10-01"})

    def test_deadline_accepts_date_object_and_slash_format(self):
        import datetime as dt

        self.assertEqual(derive_deadline(dt.date(2026, 9, 24)), "2026-10-01")
        self.assertEqual(derive_deadline("2026/09/24"), "2026-10-01")
        self.assertEqual(derive_deadline(dt.datetime(2026, 9, 24, 10, 30)), "2026-10-01")

    def test_unparseable_date_falls_back_to_undetermined(self):
        self.assertEqual(derive_deadline("尽快"), "待确定")
        self.assertEqual(derive_deadline(None), "待确定")
        self.assertEqual(derive_deadline(""), "待确定")

    def test_fallback_summary_states_counting_rule(self):
        summary = generate_fallback_summary(self.conflicts)
        self.assertEqual(summary["counting_rule"], COUNTING_RULE)
        self.assertEqual(summary["total_conflicts"], len(self.conflicts))


if __name__ == "__main__":
    unittest.main()

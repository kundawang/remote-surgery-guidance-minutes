import unittest

from app.services.csa_meeting_transcript import (
    classify_speaker_roles,
    merge_transcript_and_diarization,
    normalize_speaker,
)


class TestMultiSpeakerOverlap(unittest.TestCase):
    """一条转写段横跨多个说话人时间段时，不能悄悄丢人。"""

    def test_half_half_overlap_keeps_both_speakers(self):
        transcript = [
            {"start": 0.0, "end": 4.0, "text": "甲说两句乙也插话"},
        ]
        diarization = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        ]

        merged = merge_transcript_and_diarization(transcript, diarization)

        speakers = {piece["speaker"] for piece in merged}
        self.assertEqual(speakers, {"SPEAKER_00", "SPEAKER_01"})
        self.assertEqual(len(merged), 2)

    def test_split_boundaries_inside_segment_and_non_overlapping(self):
        transcript = [
            {"start": 1.0, "end": 7.0, "text": "这段转写横跨三个说话人时间段"},
        ]
        diarization = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_02"},
        ]

        merged = merge_transcript_and_diarization(transcript, diarization)

        self.assertEqual(len(merged), 3)
        self.assertEqual([p["speaker"] for p in merged],
                         ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
        for piece in merged:
            self.assertGreaterEqual(piece["start"], 1.0)
            self.assertLessEqual(piece["end"], 7.0)
            self.assertLess(piece["start"], piece["end"])
        for prev, nxt in zip(merged, merged[1:]):
            self.assertLessEqual(prev["end"], nxt["start"])
        # 时间区间完整覆盖原转写段
        self.assertEqual(merged[0]["start"], 1.0)
        self.assertEqual(merged[-1]["end"], 7.0)
        # 文本被拆分且不丢字符（忽略边界空白）
        self.assertEqual(
            "".join(p["text"] for p in merged),
            "这段转写横跨三个说话人时间段",
        )

    def test_equal_overlap_tiebreak_is_deterministic(self):
        # 两个说话人段对同一转写段重叠时长相等：多次运行结果必须一致。
        transcript = [
            {"start": 0.0, "end": 4.0, "text": "平局场景"},
        ]
        diarization = [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_B"},
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_A"},
        ]

        first = merge_transcript_and_diarization(transcript, diarization)
        for _ in range(5):
            again = merge_transcript_and_diarization(transcript, diarization)
            self.assertEqual(first, again)
        # 平局按 (start, end, speaker) 升序取舍，SPEAKER_A 胜出
        self.assertEqual([p["speaker"] for p in first], ["SPEAKER_A"])

    def test_partial_overlap_keeps_uncovered_part_unknown(self):
        transcript = [
            {"start": 0.0, "end": 6.0, "text": "前半有人后半没人"},
        ]
        diarization = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        ]

        merged = merge_transcript_and_diarization(transcript, diarization)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertIsNone(merged[1]["speaker"])
        self.assertEqual(merged[1]["start"], 3.0)
        self.assertEqual(merged[1]["end"], 6.0)


class TestNoOverlap(unittest.TestCase):
    """完全没有重叠的转写段保留原文并标记说话人未知。"""

    def test_no_overlap_keeps_text_and_unknown_speaker(self):
        transcript = [
            {"start": 10.0, "end": 12.0, "text": "这段没有对应说话人"},
        ]
        diarization = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        ]

        merged = merge_transcript_and_diarization(transcript, diarization)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "这段没有对应说话人")
        self.assertIsNone(merged[0]["speaker"])
        self.assertEqual(merged[0]["start"], 10.0)
        self.assertEqual(merged[0]["end"], 12.0)

    def test_empty_diarization_marks_everything_unknown(self):
        transcript = [
            {"start": 0.0, "end": 1.0, "text": "第一句"},
            {"start": 1.0, "end": 2.0, "text": "第二句"},
        ]

        merged = merge_transcript_and_diarization(transcript, [])

        self.assertEqual(len(merged), 2)
        for piece in merged:
            self.assertIsNone(piece["speaker"])
        self.assertEqual([p["text"] for p in merged], ["第一句", "第二句"])


class TestNoneSpeakerClassification(unittest.TestCase):
    """角色分类跳过未识别说话人的片段，不产生 None 键、不空调用。"""

    def test_normalize_speaker_unifies_unknown_tokens(self):
        for token in (None, "", "  ", "unknown", "UNKNOWN", "Unknown"):
            self.assertIsNone(normalize_speaker(token))
        self.assertEqual(normalize_speaker(" SPEAKER_00 "), "SPEAKER_00")

    def test_none_and_unknown_segments_are_skipped(self):
        calls = []

        def fake_classify(speaker, text):
            calls.append((speaker, text))
            return "农人"

        segments = [
            {"start": 0.0, "end": 1.0, "text": "这周西红柿可以摘了", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "没识别出说话人的内容", "speaker": None},
            {"start": 2.0, "end": 3.0, "text": "unknown 说话人内容", "speaker": "unknown"},
            {"start": 3.0, "end": 4.0, "text": "空串说话人内容", "speaker": ""},
            {"start": 4.0, "end": 5.0, "text": "", "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 6.0, "text": "   ", "speaker": "SPEAKER_02"},
        ]

        roles = classify_speaker_roles(segments, classify_fn=fake_classify)

        self.assertNotIn(None, roles)
        self.assertEqual(set(roles), {"SPEAKER_00"})
        self.assertEqual(roles["SPEAKER_00"], "农人")
        # 只对 SPEAKER_00 调用了一次，None/unknown/空文本都没有触发调用
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "SPEAKER_00")

    def test_no_none_key_and_no_empty_llm_call(self):
        calls = []

        def fake_classify(speaker, text):
            calls.append((speaker, text))
            return "消费者"

        segments = [
            {"start": 0.0, "end": 1.0, "text": "我想续订下季套餐", "speaker": "SPEAKER_01"},
            {"start": 1.0, "end": 2.0, "text": "另一条会员发言", "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "text": "未知说话人", "speaker": None},
        ]

        roles = classify_speaker_roles(segments, classify_fn=fake_classify)

        self.assertEqual(roles, {"SPEAKER_01": "消费者"})
        self.assertEqual(len(calls), 1)
        # 同一说话人的多条文本合并后一次调用
        self.assertIn("续订", calls[0][1])
        self.assertIn("会员", calls[0][1])

    def test_all_unknown_segments_yield_empty_roles(self):
        def fail_classify(speaker, text):  # 不应被调用
            raise AssertionError("不应对未知说话人发起分类调用")

        segments = [
            {"start": 0.0, "end": 1.0, "text": "未知一", "speaker": None},
            {"start": 1.0, "end": 2.0, "text": "未知二", "speaker": "unknown"},
        ]

        self.assertEqual(classify_speaker_roles(segments, classify_fn=fail_classify), {})


if __name__ == "__main__":
    unittest.main()

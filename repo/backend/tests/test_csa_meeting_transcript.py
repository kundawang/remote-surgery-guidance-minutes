"""CSA 农场会员会转写合并 / 角色分类回归测试。

覆盖：
1. 多人重叠（一条转写段横跨两个说话人，且中段各占一半插话）——不得丢说话人；
2. 转写段与说话人时间段完全没有重叠——保留原文、speaker 为 None；
3. speaker 显式为 None / 空串 / "unknown" 的片段——不产生 None 键、不空调用。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.csa_meeting_transcript import (  # noqa: E402
    OVERLAP_SPEAKER,
    classify_speaker_roles,
    merge_transcript_and_diarization,
    normalize_speaker,
)


def transcript(start, end, text):
    return {"start": start, "end": end, "text": text}


def speaker_turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


class MultiSpeakerOverlapTests(unittest.TestCase):
    def test_segments_split_when_spanning_two_speakers(self):
        transcript_segments = [transcript(0.0, 10.0, "零一二三四五六七八九")]
        speaker_segments = [
            speaker_turn(0.0, 5.0, "SPEAKER_00"),
            speaker_turn(5.0, 10.0, "SPEAKER_01"),
        ]

        merged = merge_transcript_and_diarization(transcript_segments, speaker_segments)

        self.assertEqual([item["speaker"] for item in merged],
                         ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(merged[0]["start"], 0.0)
        self.assertEqual(merged[0]["end"], 5.0)
        self.assertEqual(merged[1]["start"], 5.0)
        self.assertEqual(merged[1]["end"], 10.0)
        self.assertEqual("".join(item["text"] for item in merged), "零一二三四五六七八九")

    def test_interruption_keeps_both_speakers_and_marks_overlap(self):
        transcript_segments = [transcript(0.0, 10.0, "零一二三四五六七八九")]
        # 两个说话人时间段各与转写段重叠一半，且彼此重叠（插话）。
        speaker_segments = [
            speaker_turn(2.0, 7.0, "SPEAKER_01"),
            speaker_turn(3.0, 8.0, "SPEAKER_00"),
        ]

        merged = merge_transcript_and_diarization(transcript_segments, speaker_segments)

        speakers = [item["speaker"] for item in merged]
        overlap_items = [
            item for item in merged
            if isinstance(item["speaker"], str) and item["speaker"].startswith(OVERLAP_SPEAKER)
        ]
        self.assertTrue(overlap_items, "插话区间必须显式标记为多人重叠")
        tagged = overlap_items[0]["speaker"]
        self.assertIn("SPEAKER_00", tagged)
        self.assertIn("SPEAKER_01", tagged)
        self.assertTrue(any(item["speaker"] == "SPEAKER_01" for item in merged))
        self.assertTrue(any(
            item["speaker"] == "SPEAKER_00"
            or ("SPEAKER_00" in str(item["speaker"]))
            for item in merged
        ))

        # 切片边界落在原区间内，且彼此不重叠、无缝覆盖原区间。
        self.assertEqual(merged[0]["start"], 0.0)
        self.assertEqual(merged[-1]["end"], 10.0)
        for earlier, later in zip(merged, merged[1:]):
            self.assertEqual(earlier["end"], later["start"])
            self.assertLessEqual(earlier["start"], earlier["end"])
        self.assertEqual("".join(item["text"] for item in merged), "零一二三四五六七八九")
        # 两个说话人都必须出现在角色分类输入里（不能再"消失"）。
        roles = classify_speaker_roles(merged)
        self.assertIn("SPEAKER_00", roles)
        self.assertIn("SPEAKER_01", roles)

    def test_equal_overlap_tie_breaking_is_deterministic(self):
        transcript_segments = [transcript(0.0, 10.0, "零一二三四五六七八九")]
        speaker_segments = [
            speaker_turn(0.0, 10.0, "SPEAKER_01"),
            speaker_turn(0.0, 10.0, "SPEAKER_00"),
        ]
        first = merge_transcript_and_diarization(transcript_segments, speaker_segments)
        second = merge_transcript_and_diarization(list(reversed(transcript_segments)),
                                                  list(reversed(speaker_segments)))
        self.assertEqual(first, second)
        self.assertEqual(first[0]["speaker"], f"{OVERLAP_SPEAKER}:SPEAKER_00,SPEAKER_01")


class NoOverlapTests(unittest.TestCase):
    def test_no_overlap_keeps_text_and_unknown_speaker(self):
        transcript_segments = [transcript(0.0, 4.0, "这句话没人认领")]
        speaker_segments = [
            speaker_turn(10.0, 12.0, "SPEAKER_00"),
            speaker_turn(20.0, 22.0, "SPEAKER_01"),
        ]

        merged = merge_transcript_and_diarization(transcript_segments, speaker_segments)

        self.assertEqual(len(merged), 1)
        result = merged[0]
        self.assertIsNone(result["speaker"])
        self.assertEqual(result["text"], "这句话没人认领")
        self.assertEqual((result["start"], result["end"]), (0.0, 4.0))

    def test_gap_inside_segment_is_not_assigned_to_anyone(self):
        transcript_segments = [transcript(0.0, 12.0, "零一二三四五六七八九十甲")]
        speaker_segments = [
            speaker_turn(0.0, 4.0, "SPEAKER_00"),
            speaker_turn(8.0, 12.0, "SPEAKER_01"),
        ]

        merged = merge_transcript_and_diarization(transcript_segments, speaker_segments)

        self.assertEqual([item["speaker"] for item in merged],
                         ["SPEAKER_00", None, "SPEAKER_01"])
        self.assertEqual(merged[1]["start"], 4.0)
        self.assertEqual(merged[1]["end"], 8.0)
        self.assertEqual("".join(item["text"] for item in merged), "零一二三四五六七八九十甲")


class ExplicitNoneSpeakerTests(unittest.TestCase):
    def test_normalize_groups_unknown_variants(self):
        self.assertIsNone(normalize_speaker(None))
        self.assertIsNone(normalize_speaker(""))
        self.assertIsNone(normalize_speaker("   "))
        self.assertIsNone(normalize_speaker("unknown"))
        self.assertIsNone(normalize_speaker("UNKNOWN"))
        self.assertEqual(normalize_speaker("SPEAKER_00"), "SPEAKER_00")

    def test_explicit_none_speaker_turns_do_not_create_keys_or_calls(self):
        merged_segments = [
            {"start": 0.0, "end": 4.0, "text": "没人认领的话", "speaker": None},
            {"start": 4.0, "end": 8.0, "text": "   ", "speaker": "SPEAKER_00"},
            {"start": 8.0, "end": 12.0, "text": "这周末配送份额", "speaker": "SPEAKER_01"},
            {"start": 12.0, "end": 14.0, "text": "也没人说话", "speaker": "unknown"},
        ]

        calls = []

        def fake_llm(samples):
            calls.append(samples)
            return {sample["speaker"]: "消费者" for sample in samples}

        roles = classify_speaker_roles(merged_segments, llm_classifier=fake_llm)

        self.assertNotIn(None, roles)
        self.assertNotIn("unknown", roles)
        self.assertNotIn("", roles)
        self.assertEqual(set(roles), {"SPEAKER_01"})
        self.assertEqual(len(calls), 1)
        sent_speakers = {sample["speaker"] for sample in calls[0]}
        self.assertEqual(sent_speakers, {"SPEAKER_01"})
        self.assertTrue(all(sample["text"].strip() for sample in calls[0]))

    def test_empty_input_skips_llm_entirely(self):
        def fail_llm(samples):
            raise AssertionError("没有可分类片段时不应调用 LLM")

        self.assertEqual(classify_speaker_roles([], llm_classifier=fail_llm), {})
        self.assertEqual(
            classify_speaker_roles(
                [{"start": 0.0, "end": 1.0, "text": "", "speaker": None}],
                llm_classifier=fail_llm,
            ),
            {},
        )

    def test_get_with_default_does_not_mask_explicit_none(self):
        # 复现并锁死原始 bug：seg.get("speaker", "unknown") 对显式 None 不生效。
        segment = {"speaker": None, "text": ""}
        self.assertIsNone(normalize_speaker(segment.get("speaker", "unknown")))


if __name__ == "__main__":
    unittest.main()

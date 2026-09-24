"""说话人合并逻辑的回归测试。

覆盖：
- 同角色多人必须区分命名（张博士1 / 张博士2），同一 label 全程同名；
- 完全匹配不上的转写段保留原文并标注未知说话人；
- 同一份输入重复跑（含打乱输入顺序）结果完全一致；
- speaker 与 speaker_role 来自同一条 diarization 段；
- 中点优先、否则取最近，且多候选时裁决稳定。
"""

import importlib.util
import itertools
import os
import unittest

_UTIL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "app", "utils", "speaker_matching.py"
)
_spec = importlib.util.spec_from_file_location("speaker_matching", _UTIL_PATH)
speaker_matching = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(speaker_matching)

attach_speaker_info = speaker_matching.attach_speaker_info
build_speaker_profile_map = speaker_matching.build_speaker_profile_map

ROLE_NAME_MAP = {
    "molecular_biologist": "张博士",
    "horticulturist": "王研究员",
}


def diarization(label, start, end, role):
    return {"speaker": label, "start_time": start, "end_time": end, "role": role}


def transcript(start, end, text):
    return {"start_time": start, "end_time": end, "text": text}


class SameRoleMultipleSpeakersTest(unittest.TestCase):
    """同角色多人回归：重现 molecular_biologist 两个 label 被合并成一个名字。"""

    def setUp(self):
        self.speaker_segments = [
            diarization("SPEAKER_00", 0.0, 10.0, "molecular_biologist"),
            diarization("SPEAKER_01", 10.0, 20.0, "horticulturist"),
            diarization("SPEAKER_02", 20.0, 30.0, "molecular_biologist"),
        ]
        self.transcript_segments = [
            transcript(1.0, 3.0, "先看一下诱变株的表型。"),
            transcript(11.0, 13.0, "培养基的盐度要调整。"),
            transcript(21.0, 23.0, "这一批我来做基因表达分析。"),
        ]

    def test_same_role_labels_get_distinct_numbered_names(self):
        results = attach_speaker_info(
            self.transcript_segments,
            self.speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        names = [seg["speaker"] for seg in results]
        self.assertEqual(names, ["张博士1", "王研究员", "张博士2"])

    def test_same_label_always_resolves_to_same_name_and_role(self):
        segments = self.transcript_segments + [
            transcript(4.0, 6.0, "再补一组对照。"),
            transcript(24.0, 26.0, "表达量我这边同步测。"),
        ]
        results = attach_speaker_info(
            segments, self.speaker_segments, role_name_map=ROLE_NAME_MAP
        )
        self.assertEqual(results[0]["speaker"], results[3]["speaker"])
        self.assertEqual(results[0]["speaker_role"], results[3]["speaker_role"])
        self.assertEqual(results[2]["speaker"], results[4]["speaker"])
        self.assertEqual(results[2]["speaker_role"], results[4]["speaker_role"])
        self.assertNotEqual(results[0]["speaker"], results[2]["speaker"])

    def test_speaker_and_role_come_from_same_diarization_segment(self):
        # 两个同角色段区间重叠，名字和角色必须成对来自被选中的同一段。
        overlapping = [
            diarization("SPEAKER_00", 0.0, 12.0, "molecular_biologist"),
            diarization("SPEAKER_02", 5.0, 10.0, "molecular_biologist"),
            diarization("SPEAKER_01", 12.0, 20.0, "horticulturist"),
        ]
        results = attach_speaker_info(
            self.transcript_segments, overlapping, role_name_map=ROLE_NAME_MAP
        )
        valid_pairs = {
            ("张博士1", "molecular_biologist"),
            ("张博士2", "molecular_biologist"),
            ("王研究员", "horticulturist"),
        }
        for seg in results:
            self.assertIn((seg["speaker"], seg["speaker_role"]), valid_pairs)


class NoMatchingSpeakerTest(unittest.TestCase):
    """完全匹配不上：保留原文，标注未知说话人，不硬塞给任何人。"""

    def test_empty_diarization_keeps_text_and_marks_unknown(self):
        transcript_segments = [
            transcript(0.0, 5.0, "这一段没有任何说话人段。"),
            transcript(8.0, 12.0, "同样匹配不上的内容。"),
        ]
        results = attach_speaker_info(transcript_segments, [])
        self.assertEqual(len(results), 2)
        for original, result in zip(transcript_segments, results):
            self.assertEqual(result["speaker"], speaker_matching.UNKNOWN_SPEAKER)
            self.assertEqual(result["speaker_role"], speaker_matching.UNKNOWN_ROLE)
            self.assertEqual(result["text"], original["text"])
            self.assertEqual(result["start_time"], original["start_time"])
            self.assertEqual(result["end_time"], original["end_time"])

    def test_returned_contract_fields_unchanged(self):
        results = attach_speaker_info(
            [transcript(0.0, 2.0, "原文")],
            [diarization("SPEAKER_00", 0.0, 5.0, "molecular_biologist")],
            role_name_map=ROLE_NAME_MAP,
        )
        self.assertEqual(
            set(results[0].keys()),
            {"speaker", "speaker_role", "start_time", "end_time", "text"},
        )


class DeterministicMatchingTest(unittest.TestCase):
    """重复跑结果一致：中点优先，否则最近，多候选时裁决稳定。"""

    def test_midpoint_inside_segment_wins(self):
        speaker_segments = [
            diarization("SPEAKER_00", 0.0, 10.0, "molecular_biologist"),
            diarization("SPEAKER_01", 10.0, 20.0, "horticulturist"),
        ]
        # 转写段 8-11 与两段都有重叠，但中点 9.5 只落在 SPEAKER_00。
        results = attach_speaker_info(
            [transcript(8.0, 11.0, "中点决定归属")],
            speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        self.assertEqual(results[0]["speaker"], "张博士")

    def test_overlapping_segments_containing_midpoint_are_stable(self):
        # 中点 7.0 同时落在两个区间内：start_time 更早者胜出。
        speaker_segments = [
            diarization("SPEAKER_02", 5.0, 12.0, "molecular_biologist"),
            diarization("SPEAKER_00", 0.0, 12.0, "molecular_biologist"),
        ]
        results = attach_speaker_info(
            [transcript(5.0, 9.0, "两段都能匹配")],
            speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        self.assertEqual(results[0]["speaker"], "张博士1")

    def test_nearest_midpoint_when_none_contains_it(self):
        # 中点 3.0 不在任何区间，到 SPEAKER_00 中点 1.0 最近。
        speaker_segments = [
            diarization("SPEAKER_00", 0.0, 2.0, "molecular_biologist"),
            diarization("SPEAKER_01", 4.0, 6.0, "horticulturist"),
        ]
        results = attach_speaker_info(
            [transcript(2.5, 3.5, "取最近的段")],
            speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        self.assertEqual(results[0]["speaker"], "张博士")

    def test_equidistant_fallback_breaks_tie_deterministically(self):
        # 中点 3.0 到两段中点距离相同：start_time 更早者胜出。
        speaker_segments = [
            diarization("SPEAKER_01", 4.0, 6.0, "horticulturist"),
            diarization("SPEAKER_00", 0.0, 2.0, "molecular_biologist"),
        ]
        results = attach_speaker_info(
            [transcript(2.5, 3.5, "等距取更早的段")],
            speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        self.assertEqual(results[0]["speaker"], "张博士")

    def test_repeated_runs_and_input_permutations_identical(self):
        speaker_segments = [
            diarization("SPEAKER_00", 0.0, 10.0, "molecular_biologist"),
            diarization("SPEAKER_01", 10.0, 20.0, "horticulturist"),
            diarization("SPEAKER_02", 20.0, 30.0, "molecular_biologist"),
        ]
        transcript_segments = [
            transcript(1.0, 3.0, "第一段。"),
            transcript(8.0, 12.0, "跨边界，中点在第二段。"),
            transcript(22.0, 24.0, "同角色的另一个人。"),
            transcript(35.0, 38.0, "区间外，靠最近邻。"),
        ]

        baseline = attach_speaker_info(
            transcript_segments, speaker_segments, role_name_map=ROLE_NAME_MAP
        )
        for _ in range(5):
            self.assertEqual(
                attach_speaker_info(
                    transcript_segments,
                    speaker_segments,
                    role_name_map=ROLE_NAME_MAP,
                ),
                baseline,
            )

        permutations = set(itertools.permutations(range(len(speaker_segments))))
        for order in permutations:
            shuffled = [speaker_segments[i] for i in order]
            results = attach_speaker_info(
                transcript_segments, shuffled, role_name_map=ROLE_NAME_MAP
            )
            self.assertEqual(
                [seg["speaker"] for seg in results],
                [seg["speaker"] for seg in baseline],
            )
            self.assertEqual(
                [seg["speaker_role"] for seg in results],
                [seg["speaker_role"] for seg in baseline],
            )

    def test_profile_map_order_independent(self):
        segments_a = [
            diarization("SPEAKER_00", 0.0, 10.0, "molecular_biologist"),
            diarization("SPEAKER_02", 20.0, 30.0, "molecular_biologist"),
        ]
        segments_b = list(reversed(segments_a))
        self.assertEqual(
            build_speaker_profile_map(segments_a, ROLE_NAME_MAP),
            build_speaker_profile_map(segments_b, ROLE_NAME_MAP),
        )


if __name__ == "__main__":
    unittest.main()

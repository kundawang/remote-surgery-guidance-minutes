"""说话人合并逻辑的回归测试。

覆盖三组回归场景：
- 同角色多人：同一角色的多个 diarization label 必须区分命名
  （张博士1 / 张博士2），同一 label 全程同名；
- 完全匹配不上：没有任何说话人段时保留原文，标注未知说话人；
- 重复跑结果一致：同一份输入重复跑、输入顺序打乱，输出完全一致。
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
    """同角色多人：两个 molecular_biologist 不能再共用一个"张博士"。"""

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
        self.assertEqual(
            [seg["speaker"] for seg in results],
            ["张博士1", "王研究员", "张博士2"],
        )

    def test_same_label_always_resolves_to_same_name_and_role(self):
        segments = self.transcript_segments + [
            transcript(4.0, 6.0, "再补一组对照。"),
            transcript(24.0, 26.0, "表达量我这边同步测。"),
        ]
        results = attach_speaker_info(
            segments, self.speaker_segments, role_name_map=ROLE_NAME_MAP
        )
        # 同一 label 的多段转写：名字和角色始终一致
        self.assertEqual(results[0]["speaker"], results[3]["speaker"])
        self.assertEqual(results[0]["speaker_role"], results[3]["speaker_role"])
        self.assertEqual(results[2]["speaker"], results[4]["speaker"])
        self.assertEqual(results[2]["speaker_role"], results[4]["speaker_role"])
        # 同角色的两个 label：名字必须能区分
        self.assertNotEqual(results[0]["speaker"], results[2]["speaker"])

    def test_speaker_and_role_come_from_same_diarization_segment(self):
        # 两个同角色段区间重叠时，名字和角色必须成对来自被选中的同一段
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
    """重复跑结果一致：同输入同输出，与 diarization 段传入顺序无关。"""

    def setUp(self):
        self.speaker_segments = [
            diarization("SPEAKER_00", 0.0, 10.0, "molecular_biologist"),
            diarization("SPEAKER_01", 10.0, 20.0, "horticulturist"),
            diarization("SPEAKER_02", 20.0, 30.0, "molecular_biologist"),
        ]
        self.transcript_segments = [
            transcript(1.0, 3.0, "甲"),
            transcript(9.0, 11.0, "乙"),   # 中点 10.0 同时落在两段边界上
            transcript(15.0, 17.0, "丙"),
            transcript(28.0, 32.0, "丁"),  # 中点落在所有区间之外，取最近段
        ]

    def test_repeated_runs_produce_identical_output(self):
        first = attach_speaker_info(
            self.transcript_segments, self.speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        for _ in range(5):
            again = attach_speaker_info(
                self.transcript_segments, self.speaker_segments,
                role_name_map=ROLE_NAME_MAP,
            )
            self.assertEqual(first, again)

    def test_input_order_of_diarization_segments_does_not_matter(self):
        baseline = attach_speaker_info(
            self.transcript_segments, self.speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        for perm in itertools.permutations(self.speaker_segments):
            result = attach_speaker_info(
                self.transcript_segments, list(perm),
                role_name_map=ROLE_NAME_MAP,
            )
            self.assertEqual(baseline, result)

    def test_boundary_midpoint_resolves_deterministically(self):
        # 中点 10.0 同时是 SPEAKER_00 的终点和 SPEAKER_01 的起点，
        # 无论段顺序如何都必须裁决到同一个说话人
        speakers = set()
        for perm in itertools.permutations(self.speaker_segments):
            result = attach_speaker_info(
                [transcript(9.0, 11.0, "边界")], list(perm),
                role_name_map=ROLE_NAME_MAP,
            )
            speakers.add(result[0]["speaker"])
        self.assertEqual(len(speakers), 1)

    def test_midpoint_inside_interval_beats_nearer_midpoint_segment(self):
        # 中点落在 SPEAKER_00 区间内，即使 SPEAKER_01 的中点更近也选前者
        speaker_segments = [
            diarization("SPEAKER_00", 0.0, 10.0, "molecular_biologist"),
            diarization("SPEAKER_01", 10.0, 11.0, "horticulturist"),
        ]
        results = attach_speaker_info(
            [transcript(8.0, 12.0, "中点在左段内")], speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        self.assertEqual(results[0]["speaker_role"], "molecular_biologist")

    def test_nearest_segment_used_when_midpoint_outside_all_intervals(self):
        results = attach_speaker_info(
            [transcript(28.0, 32.0, "超出所有区间")], self.speaker_segments,
            role_name_map=ROLE_NAME_MAP,
        )
        self.assertEqual(results[0]["speaker"], "张博士2")
        self.assertEqual(results[0]["speaker_role"], "molecular_biologist")


if __name__ == "__main__":
    unittest.main()

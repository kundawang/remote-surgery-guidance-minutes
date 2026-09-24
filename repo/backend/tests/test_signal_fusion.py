"""signal_fusion 回归测试：单路/双路置信度、时间对齐、文本合并确定性。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from signal_fusion import SignalFusion, TextSegment


def seg(text, start, end, confidence):
    return TextSegment(text=text, start=start, end=end, confidence=confidence)


class SingleStreamConfidenceTest(unittest.TestCase):
    """单路信号置信度不应被融合权重打折。"""

    def test_audio_only_confidence_not_discounted(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("切开皮肤", 0.0, 2.0, 1.0))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 1)
        self.assertAlmostEqual(result["segments"][0].confidence, 1.0)
        self.assertAlmostEqual(result["overall_confidence"], 1.0)

    def test_bci_only_confidence_not_discounted(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(seg("分离组织", 0.0, 2.0, 0.9))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 1)
        self.assertAlmostEqual(result["segments"][0].confidence, 0.9)
        self.assertAlmostEqual(result["overall_confidence"], 0.9)

    def test_single_stream_sources_marked(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("止血", 0.0, 1.0, 0.8))
        fusion.add_bci_segment(seg("缝合", 5.0, 6.0, 0.7))
        result = fusion.fuse()
        sources = [s.sources for s in result["segments"]]
        self.assertEqual(sources, [["audio"], ["bci"]])


class DualStreamConfidenceTest(unittest.TestCase):
    """双路配对时按归一化权重做加权平均。"""

    def test_paired_confidence_is_normalized_weighted_average(self):
        fusion = SignalFusion(bci_weight=0.4, audio_weight=0.6)
        fusion.add_bci_segment(seg("分离组织", 0.0, 2.0, 0.8))
        fusion.add_audio_segment(seg("分离组织", 0.2, 2.2, 1.0))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 1)
        expected = (0.8 * 0.4 + 1.0 * 0.6) / (0.4 + 0.6)
        self.assertAlmostEqual(result["segments"][0].confidence, expected)
        self.assertAlmostEqual(result["overall_confidence"], expected)

    def test_weights_not_summing_to_one_are_normalized(self):
        fusion = SignalFusion(bci_weight=2.0, audio_weight=1.0)
        fusion.add_bci_segment(seg("吻合血管", 0.0, 1.0, 0.9))
        fusion.add_audio_segment(seg("吻合血管", 0.1, 1.1, 0.6))
        result = fusion.fuse()
        expected = (0.9 * 2.0 + 0.6 * 1.0) / 3.0
        self.assertAlmostEqual(result["segments"][0].confidence, expected)

    def test_overall_not_polluted_by_single_stream_discount(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("消毒铺巾", 0.0, 1.0, 1.0))
        fusion.add_bci_segment(seg("止血", 2.0, 3.0, 0.5))
        fusion.add_audio_segment(seg("止血", 2.1, 3.1, 0.7))
        result = fusion.fuse()
        pair_conf = (0.5 * 0.4 + 0.7 * 0.6) / 1.0
        expected = (1.0 * 1.0 + pair_conf * 1.1) / 2.1
        self.assertAlmostEqual(result["overall_confidence"], expected)
        self.assertGreater(result["overall_confidence"], pair_conf)


class AlignmentTest(unittest.TestCase):
    """max_delay 之外不配对、配过的不复用、未配上的音频单独成段。"""

    def test_beyond_max_delay_not_paired(self):
        fusion = SignalFusion(max_delay=1.0)
        fusion.add_bci_segment(seg("切皮", 0.0, 1.0, 0.8))
        fusion.add_audio_segment(seg("切皮", 5.0, 6.0, 0.9))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(
            [s.sources for s in result["segments"]], [["bci"], ["audio"]]
        )

    def test_audio_segment_not_reused(self):
        fusion = SignalFusion(max_delay=1.0)
        fusion.add_bci_segment(seg("探查", 0.0, 1.0, 0.8))
        fusion.add_bci_segment(seg("探查", 0.4, 1.4, 0.6))
        fusion.add_audio_segment(seg("探查", 0.2, 1.2, 0.9))
        result = fusion.fuse()
        paired = [s for s in result["segments"] if len(s.sources) == 2]
        single = [s for s in result["segments"] if len(s.sources) == 1]
        self.assertEqual(len(paired), 1)
        self.assertEqual(len(single), 1)
        # 音频与延时更小的第一路 BCI 配对
        self.assertAlmostEqual(paired[0].confidence, (0.8 * 0.4 + 0.9 * 0.6))
        self.assertAlmostEqual(single[0].confidence, 0.6)

    def test_unpaired_audio_becomes_own_segment(self):
        fusion = SignalFusion(max_delay=0.5)
        fusion.add_bci_segment(seg("结扎", 0.0, 1.0, 0.7))
        fusion.add_audio_segment(seg("放置引流管", 10.0, 11.0, 0.95))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 2)
        audio_seg = result["segments"][1]
        self.assertEqual(audio_seg.sources, ["audio"])
        self.assertEqual(audio_seg.text, "放置引流管")
        self.assertAlmostEqual(audio_seg.confidence, 0.95)

    def test_segments_sorted_by_start(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("晚段", 8.0, 9.0, 0.9))
        fusion.add_bci_segment(seg("早段", 1.0, 2.0, 0.8))
        result = fusion.fuse()
        self.assertEqual([s.text for s in result["segments"]], ["早段", "晚段"])


class MergeTextTest(unittest.TestCase):
    """重叠高取更长文本，否则按固定顺序拼接；结果确定。"""

    def test_high_overlap_takes_longer_text(self):
        fusion = SignalFusion(overlap_threshold=0.5)
        fusion.add_bci_segment(seg("分离 组织 粘连", 0.0, 1.0, 0.8))
        fusion.add_audio_segment(seg("分离 组织", 0.1, 1.1, 0.9))
        result = fusion.fuse()
        self.assertEqual(result["segments"][0].text, "分离 组织 粘连")

    def test_low_overlap_concatenates_bci_first(self):
        fusion = SignalFusion(overlap_threshold=0.5)
        fusion.add_bci_segment(seg("建立气腹", 0.0, 1.0, 0.8))
        fusion.add_audio_segment(seg("置入戳卡", 0.1, 1.1, 0.9))
        result = fusion.fuse()
        self.assertEqual(result["segments"][0].text, "建立气腹 置入戳卡")

    def test_merge_is_deterministic_across_runs(self):
        texts = []
        for _ in range(5):
            fusion = SignalFusion()
            fusion.add_bci_segment(seg("肝 左叶 切除 术", 0.0, 1.0, 0.8))
            fusion.add_audio_segment(seg("肝 左叶 切除", 0.1, 1.1, 0.9))
            texts.append(fusion.fuse()["fused_text"])
        self.assertEqual(len(set(texts)), 1)


class PublicApiTest(unittest.TestCase):
    def test_fuse_return_fields(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("冲洗腹腔", 0.0, 1.0, 0.9))
        result = fusion.fuse()
        self.assertEqual(
            set(result.keys()), {"segments", "fused_text", "overall_confidence"}
        )
        self.assertEqual(result["fused_text"], "冲洗腹腔")

    def test_empty_fuse(self):
        result = SignalFusion().fuse()
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["fused_text"], "")
        self.assertEqual(result["overall_confidence"], 0.0)

    def test_get_fused_segments(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(seg("缝合切口", 0.0, 1.0, 0.8))
        segments = fusion.get_fused_segments()
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "缝合切口")

    def test_clear_buffers(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(seg("包扎", 0.0, 1.0, 0.8))
        fusion.add_audio_segment(seg("包扎", 0.1, 1.1, 0.9))
        fusion.fuse()
        fusion.clear_buffers()
        result = fusion.fuse()
        self.assertEqual(result["segments"], [])
        self.assertEqual(fusion.get_fused_segments(), [])


if __name__ == "__main__":
    unittest.main()

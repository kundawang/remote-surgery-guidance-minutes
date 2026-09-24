"""signal_fusion 回归测试。

覆盖：
- 模块可正常 import（类型注解语法修复）
- 单路信号置信度不被融合权重打折
- 双路配对置信度为归一化加权平均
- overall_confidence 不被单路折扣污染
- 文本合并规则确定（高重叠取更长，低重叠 BCI 前音频后拼接）
- 时间对齐规则（max_delay 外不配对、配过不复用、未配对音频单独成段）
- 公开方法名与返回字段稳定
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from signal_fusion import FusedSegment, SignalFusion, TextSegment  # noqa: E402


def seg(text, start, end, confidence):
    return TextSegment(text=text, start=start, end=end, confidence=confidence)


class SingleSourceConfidenceTest(unittest.TestCase):
    """单路信号直接输出自身置信度，不乘融合权重。"""

    def test_audio_only_confidence_not_discounted(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("纯音频段", 0.0, 2.0, 1.0))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 1)
        fused = result["segments"][0]
        self.assertEqual(fused.sources, ["audio"])
        self.assertAlmostEqual(fused.confidence, 1.0)
        self.assertAlmostEqual(result["overall_confidence"], 1.0)

    def test_bci_only_confidence_not_discounted(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(seg("纯脑机段", 0.0, 2.0, 0.9))
        result = fusion.fuse()
        fused = result["segments"][0]
        self.assertEqual(fused.sources, ["bci"])
        self.assertAlmostEqual(fused.confidence, 0.9)
        self.assertAlmostEqual(result["overall_confidence"], 0.9)

    def test_mixed_single_segments_keep_own_confidence(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("音频", 0.0, 1.0, 0.8))
        fusion.add_bci_segment(seg("脑机", 5.0, 6.0, 0.6))
        result = fusion.fuse()
        confidences = {s.sources[0]: s.confidence for s in result["segments"]}
        self.assertAlmostEqual(confidences["audio"], 0.8)
        self.assertAlmostEqual(confidences["bci"], 0.6)
        # 两段等长，整体置信度为 (0.8 + 0.6) / 2，不含权重折扣
        self.assertAlmostEqual(result["overall_confidence"], 0.7)


class DualSourceConfidenceTest(unittest.TestCase):
    """双路配对时按 (bci_weight, audio_weight) 归一化加权平均。"""

    def test_pair_confidence_is_normalized_weighted_average(self):
        fusion = SignalFusion(bci_weight=0.4, audio_weight=0.6)
        fusion.add_bci_segment(seg("切开 皮肤", 0.0, 2.0, 0.8))
        fusion.add_audio_segment(seg("切开 皮肤", 0.1, 2.1, 1.0))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 1)
        fused = result["segments"][0]
        self.assertEqual(fused.sources, ["bci", "audio"])
        expected = (0.8 * 0.4 + 1.0 * 0.6) / (0.4 + 0.6)
        self.assertAlmostEqual(fused.confidence, expected)
        self.assertAlmostEqual(result["overall_confidence"], expected)

    def test_weights_not_summing_to_one_are_normalized(self):
        fusion = SignalFusion(bci_weight=1.0, audio_weight=3.0)
        fusion.add_bci_segment(seg("止血", 0.0, 1.0, 0.5))
        fusion.add_audio_segment(seg("止血", 0.2, 1.2, 0.9))
        fused = fusion.fuse()["segments"][0]
        self.assertAlmostEqual(fused.confidence, (0.5 * 1.0 + 0.9 * 3.0) / 4.0)

    def test_overall_confidence_not_polluted_by_single_source(self):
        fusion = SignalFusion()
        # 配对段：时长 2，置信度 0.8*0.4 + 1.0*0.6 = 0.92
        fusion.add_bci_segment(seg("配对", 0.0, 2.0, 0.8))
        fusion.add_audio_segment(seg("配对", 0.0, 2.0, 1.0))
        # 单路段：时长 1，置信度保持 0.5（不打折成 0.5*0.6）
        fusion.add_audio_segment(seg("单路", 10.0, 11.0, 0.5))
        result = fusion.fuse()
        expected = (0.92 * 2.0 + 0.5 * 1.0) / 3.0
        self.assertAlmostEqual(result["overall_confidence"], expected)


class AlignmentTest(unittest.TestCase):
    """时间对齐：max_delay 之外不配对、配过不复用、未配对音频单独成段。"""

    def test_beyond_max_delay_not_paired(self):
        fusion = SignalFusion(max_delay=1.0)
        fusion.add_bci_segment(seg("脑机", 0.0, 1.0, 0.7))
        fusion.add_audio_segment(seg("音频", 5.0, 6.0, 0.9))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 2)
        sources = sorted(tuple(s.sources) for s in result["segments"])
        self.assertEqual(sources, [("audio",), ("bci",)])

    def test_audio_segment_used_at_most_once(self):
        fusion = SignalFusion(max_delay=1.0)
        fusion.add_bci_segment(seg("近", 0.0, 1.0, 0.7))
        fusion.add_bci_segment(seg("远", 0.5, 1.5, 0.6))
        fusion.add_audio_segment(seg("音频", 0.1, 1.1, 0.9))
        result = fusion.fuse()
        paired = [s for s in result["segments"] if len(s.sources) == 2]
        singles = [s for s in result["segments"] if len(s.sources) == 1]
        self.assertEqual(len(paired), 1)
        self.assertEqual(len(singles), 1)
        # 距离更近的 bci（start=0.0）配对成功，另一段单独成段
        self.assertEqual(paired[0].text.split()[0], "近")
        self.assertEqual(singles[0].text, "远")

    def test_unmatched_audio_becomes_own_segment(self):
        fusion = SignalFusion(max_delay=0.5)
        fusion.add_bci_segment(seg("脑机", 0.0, 1.0, 0.7))
        fusion.add_audio_segment(seg("音频一", 0.2, 1.2, 0.9))
        fusion.add_audio_segment(seg("音频二", 10.0, 11.0, 0.8))
        result = fusion.fuse()
        self.assertEqual(len(result["segments"]), 2)
        lone = [s for s in result["segments"] if s.sources == ["audio"]]
        self.assertEqual(len(lone), 1)
        self.assertEqual(lone[0].text, "音频二")
        self.assertAlmostEqual(lone[0].confidence, 0.8)

    def test_segments_sorted_by_start(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("后", 10.0, 11.0, 0.8))
        fusion.add_bci_segment(seg("先", 0.0, 1.0, 0.7))
        result = fusion.fuse()
        starts = [s.start for s in result["segments"]]
        self.assertEqual(starts, sorted(starts))


class MergeTextTest(unittest.TestCase):
    """文本合并：高重叠取更长，低重叠按 BCI 前音频后拼接，结果确定。"""

    def test_high_overlap_takes_longer_text(self):
        fusion = SignalFusion(overlap_threshold=0.5)
        self.assertEqual(
            fusion._merge_texts("切开 皮肤 止血", "切开 皮肤"), "切开 皮肤 止血"
        )
        self.assertEqual(
            fusion._merge_texts("切开 皮肤", "切开 皮肤 止血 缝合"),
            "切开 皮肤 止血 缝合",
        )

    def test_low_overlap_concatenates_bci_first(self):
        fusion = SignalFusion(overlap_threshold=0.5)
        self.assertEqual(
            fusion._merge_texts("脑机 文本", "音频 内容"), "脑机 文本 音频 内容"
        )

    def test_merge_is_deterministic(self):
        fusion = SignalFusion(overlap_threshold=0.5)
        # 完全相同的文本：多次合并结果唯一，不依赖 set 遍历顺序
        results = {fusion._merge_texts("aaa bbb", "aaa bbb") for _ in range(50)}
        self.assertEqual(results, {"aaa bbb"})
        # 词数相同、字符数不同：固定取字符更长的那路，与参数顺序无关
        results = {fusion._merge_texts("aaaa bbb", "aaaa bb") for _ in range(50)}
        self.assertEqual(results, {"aaaa bbb"})
        results = {fusion._merge_texts("aaaa bb", "aaaa bbb") for _ in range(50)}
        self.assertEqual(results, {"aaaa bbb"})

    def test_empty_side_returns_other(self):
        fusion = SignalFusion()
        self.assertEqual(fusion._merge_texts("", "音频"), "音频")
        self.assertEqual(fusion._merge_texts("脑机", ""), "脑机")


class PublicApiTest(unittest.TestCase):
    """公开方法名与返回字段保持稳定。"""

    def test_fuse_returns_expected_fields(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(seg("脑机", 0.0, 1.0, 0.7))
        fusion.add_audio_segment(seg("音频", 0.1, 1.1, 0.9))
        result = fusion.fuse()
        self.assertIn("segments", result)
        self.assertIn("fused_text", result)
        self.assertIn("overall_confidence", result)
        self.assertIsInstance(result["fused_text"], str)
        self.assertTrue(
            all(isinstance(s, FusedSegment) for s in result["segments"])
        )

    def test_get_fused_segments_triggers_fuse_lazily(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(seg("音频", 0.0, 1.0, 0.9))
        segments = fusion.get_fused_segments()
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "音频")

    def test_clear_buffers_resets_state(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(seg("脑机", 0.0, 1.0, 0.7))
        fusion.fuse()
        fusion.clear_buffers()
        result = fusion.fuse()
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["fused_text"], "")
        self.assertEqual(result["overall_confidence"], 0.0)
        self.assertEqual(fusion.get_fused_segments(), [])


if __name__ == "__main__":
    unittest.main()

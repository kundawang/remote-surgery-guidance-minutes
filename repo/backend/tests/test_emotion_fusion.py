"""EmotionFusionEngine 归一化与权重融合的单元测试。

覆盖：
- 生理单路信号必须能覆盖整个 [0,1]（历史 bug：上限只有 0.6）；
- 双模态融合不得对生理一路二次打折（历史 bug：整体上限只有 0.76）；
- 缺失信号（None / 空 dict / 部分分量缺失）按"没有这份证据"处理，
  只在剩余信号上做权重归一化，不填 0.5、不摊薄；
- 情绪分类锚点区间不变，高唤醒场景落在 angry/anxious 一侧。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.emotion_fusion import (  # noqa: E402
    EMOTION_ANCHORS,
    EmotionFusionEngine,
)


class TestPhysioOnlyRange(unittest.TestCase):
    """只有生理信号时，arousal 必须覆盖完整 [0,1]。"""

    def setUp(self):
        self.engine = EmotionFusionEngine()

    def test_max_eda_min_hrv_gives_arousal_near_one(self):
        # 皮电顶格(20) + HRV 触底(10)：最紧张，arousal 应接近 1.0
        result = self.engine.analyze(physio={"eda": 20.0, "hrv": 10.0})
        self.assertAlmostEqual(result["arousal"], 1.0, places=6)

    def test_min_eda_max_hrv_gives_arousal_near_zero(self):
        # 皮电触底(0) + HRV 顶格(100)：最放松，arousal 应接近 0.0
        result = self.engine.analyze(physio={"eda": 0.0, "hrv": 100.0})
        self.assertAlmostEqual(result["arousal"], 0.0, places=6)

    def test_physio_only_not_diluted_by_missing_voice(self):
        # 语音缺失时，生理结果不得再乘 0.6 被摊薄
        arousal = self.engine.compute_physio_arousal({"eda": 20.0, "hrv": 10.0})
        result = self.engine.analyze(physio={"eda": 20.0, "hrv": 10.0})
        self.assertAlmostEqual(result["arousal"], arousal, places=6)


class TestModalityFusion(unittest.TestCase):
    """双模态融合：真正的加权平均，生理一路不被二次打折。"""

    def setUp(self):
        self.engine = EmotionFusionEngine()
        self.max_physio = {"eda": 20.0, "hrv": 10.0}
        self.angry_voice = {"arousal": 0.8, "valence": 0.15}

    def test_fused_arousal_exceeds_old_double_discount_cap(self):
        # 历史 bug 上限 0.76；修复后 0.6*1.0 + 0.4*0.8 = 0.92
        result = self.engine.analyze(physio=self.max_physio, voice=self.angry_voice)
        self.assertGreater(result["arousal"], 0.76)
        self.assertAlmostEqual(result["arousal"], 0.92, places=6)

    def test_fused_valence_is_weighted_average(self):
        # 0.6*0.0 + 0.4*0.15 = 0.06
        result = self.engine.analyze(physio=self.max_physio, voice=self.angry_voice)
        self.assertAlmostEqual(result["valence"], 0.06, places=6)

    def test_max_physio_plus_angry_voice_maps_to_angry_or_anxious(self):
        # 生理顶格 + 语音愤怒：必须落在 angry/anxious 一侧，而非 calm/neutral
        result = self.engine.analyze(physio=self.max_physio, voice=self.angry_voice)
        self.assertIn(result["emotion"], ("angry", "anxious"))

    def test_relaxed_physio_plus_calm_voice_maps_to_calm(self):
        result = self.engine.analyze(
            physio={"eda": 2.0, "hrv": 95.0},
            voice={"arousal": 0.2, "valence": 0.7},
        )
        self.assertEqual(result["emotion"], "calm")

    def test_results_always_within_unit_interval(self):
        cases = [
            ({"eda": 20.0, "hrv": 10.0}, {"arousal": 1.0, "valence": 0.0}),
            ({"eda": 0.0, "hrv": 100.0}, {"arousal": 0.0, "valence": 1.0}),
            ({"eda": 10.0, "hrv": 55.0}, {"arousal": 0.5, "valence": 0.5}),
            ({"eda": 99.0, "hrv": 1.0}, {"arousal": 1.2, "valence": -0.3}),  # 超量程需截断
        ]
        for physio, voice in cases:
            with self.subTest(physio=physio, voice=voice):
                result = self.engine.analyze(physio=physio, voice=voice)
                for key in ("arousal", "valence", "confidence", "score"):
                    self.assertGreaterEqual(result[key], 0.0, key)
                    self.assertLessEqual(result[key], 1.0, key)


class TestMissingSignals(unittest.TestCase):
    """缺失信号 = 没有这份证据：重新归一化，不填 0.5，不摊薄。"""

    def setUp(self):
        self.engine = EmotionFusionEngine()

    def test_none_physio_uses_voice_only(self):
        result = self.engine.analyze(physio=None, voice={"arousal": 0.8, "valence": 0.2})
        self.assertAlmostEqual(result["arousal"], 0.8, places=6)
        self.assertAlmostEqual(result["valence"], 0.2, places=6)

    def test_empty_physio_dict_uses_voice_only(self):
        result = self.engine.analyze(physio={}, voice={"arousal": 0.8, "valence": 0.2})
        self.assertAlmostEqual(result["arousal"], 0.8, places=6)
        self.assertAlmostEqual(result["valence"], 0.2, places=6)

    def test_none_voice_uses_physio_only(self):
        result = self.engine.analyze(physio={"eda": 20.0, "hrv": 10.0}, voice=None)
        self.assertAlmostEqual(result["arousal"], 1.0, places=6)

    def test_empty_voice_dict_uses_physio_only(self):
        result = self.engine.analyze(physio={"eda": 20.0, "hrv": 10.0}, voice={})
        self.assertAlmostEqual(result["arousal"], 1.0, places=6)

    def test_missing_hrv_component_not_padded_with_half(self):
        # 只有皮电顶格：arousal 应为 1.0，而不是 (1.0+0.5)/2 或被权重摊薄
        arousal = self.engine.compute_physio_arousal({"eda": 20.0, "hrv": None})
        self.assertAlmostEqual(arousal, 1.0, places=6)
        arousal = self.engine.compute_physio_arousal({"eda": 20.0})
        self.assertAlmostEqual(arousal, 1.0, places=6)

    def test_missing_eda_component_uses_hrv_only(self):
        arousal = self.engine.compute_physio_arousal({"eda": None, "hrv": 10.0})
        self.assertAlmostEqual(arousal, 1.0, places=6)
        arousal = self.engine.compute_physio_arousal({"hrv": 100.0})
        self.assertAlmostEqual(arousal, 0.0, places=6)

    def test_all_signals_missing_returns_neutral_zero_confidence(self):
        for physio, voice in ((None, None), ({}, {}), ({"eda": None}, None)):
            with self.subTest(physio=physio, voice=voice):
                result = self.engine.analyze(physio=physio, voice=voice)
                self.assertEqual(result["emotion"], "neutral")
                self.assertEqual(result["confidence"], 0.0)
                self.assertEqual(result["score"], 0.0)


class TestEmotionMappingAndSchema(unittest.TestCase):
    """情绪锚点区间与返回字段保持不变。"""

    def setUp(self):
        self.engine = EmotionFusionEngine()

    def test_anchor_points_map_to_their_own_category(self):
        for name, (arousal, valence) in EMOTION_ANCHORS.items():
            with self.subTest(emotion=name):
                self.assertEqual(
                    self.engine.map_to_emotion_category(arousal, valence), name
                )

    def test_high_arousal_low_valence_is_not_calm_or_happy(self):
        emotion = self.engine.map_to_emotion_category(0.9, 0.1)
        self.assertIn(emotion, ("angry", "anxious"))

    def test_public_result_fields(self):
        result = self.engine.analyze(
            physio={"eda": 10.0, "hrv": 50.0},
            voice={"arousal": 0.5, "valence": 0.5},
        )
        self.assertEqual(
            set(result.keys()),
            {"arousal", "valence", "emotion", "confidence", "score"},
        )


if __name__ == "__main__":
    unittest.main()

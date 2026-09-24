"""EmotionFusionEngine 归一化与权重融合的单元测试。

覆盖：
- 只有生理信号时 arousal 能覆盖完整 [0,1]（历史 bug：生理内部未归一化，上限 0.6）；
- 双模态融合是权重和为 1 的真正加权平均，生理一路不被二次打折
  （历史 bug：融合后上限只有 0.76，紧张/愤怒被判成 calm/neutral）；
- 某一路信号缺失（None / 空 dict / 单个分量缺失）按"没有这份证据"处理：
  在剩余信号上重新归一化权重，不填 0.5、不摊薄、不整体压小；
- 情绪锚点与 score 区间不变，"生理顶格 + 语音愤怒"落在 angry/anxious 一侧；
- 返回字段与方法签名保持公开契约不变。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.emotion_fusion import (  # noqa: E402
    EMOTION_ANCHORS,
    MODALITY_WEIGHTS,
    EmotionFusionEngine,
)


class TestPhysioOnlyRange(unittest.TestCase):
    """只有生理信号时，arousal 必须能覆盖整个 [0,1]。"""

    def setUp(self):
        self.engine = EmotionFusionEngine()

    def test_max_eda_min_hrv_gives_arousal_one(self):
        result = self.engine.analyze(physio={"eda": 20.0, "hrv": 10.0})
        self.assertAlmostEqual(result["arousal"], 1.0, places=6)

    def test_min_eda_max_hrv_gives_arousal_zero(self):
        result = self.engine.analyze(physio={"eda": 0.0, "hrv": 100.0})
        self.assertAlmostEqual(result["arousal"], 0.0, places=6)

    def test_mid_range_signals_give_half_arousal(self):
        result = self.engine.analyze(physio={"eda": 10.0, "hrv": 55.0})
        self.assertAlmostEqual(result["arousal"], 0.5, places=6)

    def test_analyze_physio_only_equals_compute_physio(self):
        physio = {"eda": 17.0, "hrv": 24.0}
        result = self.engine.analyze(physio=physio)
        self.assertAlmostEqual(
            result["arousal"], self.engine.compute_physio_arousal(physio), places=6
        )
        self.assertAlmostEqual(
            result["valence"], self.engine.compute_physio_valence(physio), places=6
        )

    def test_out_of_range_signals_are_clamped(self):
        high = self.engine.analyze(physio={"eda": 99.0, "hrv": 1.0})
        self.assertEqual(high["arousal"], 1.0)
        low = self.engine.analyze(physio={"eda": -5.0, "hrv": 500.0})
        self.assertEqual(low["arousal"], 0.0)


class TestModalityFusion(unittest.TestCase):
    """双模态融合：权重和为 1，生理一路不被二次打折。"""

    def setUp(self):
        self.engine = EmotionFusionEngine()
        self.max_physio = {"eda": 20.0, "hrv": 10.0}
        self.angry_voice = {"arousal": 0.8, "valence": 0.15}

    def test_fused_arousal_is_true_weighted_average(self):
        # 修复前上限 0.76；修复后 0.6*1.0 + 0.4*0.8 = 0.92
        result = self.engine.analyze(physio=self.max_physio, voice=self.angry_voice)
        self.assertAlmostEqual(result["arousal"], 0.92, places=6)
        self.assertGreater(result["arousal"], 0.76)

    def test_fused_valence_is_true_weighted_average(self):
        # 生理 valence：eda 顶格(0)、hrv 触底(0) -> 0.0；语音 0.15
        # 0.6*0.0 + 0.4*0.15 = 0.06
        result = self.engine.analyze(physio=self.max_physio, voice=self.angry_voice)
        self.assertAlmostEqual(result["valence"], 0.06, places=6)

    def test_modality_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(MODALITY_WEIGHTS.values()), 1.0, places=9)

    def test_max_physio_plus_angry_voice_maps_to_angry_or_anxious(self):
        result = self.engine.analyze(physio=self.max_physio, voice=self.angry_voice)
        self.assertIn(result["emotion"], ("angry", "anxious"))
        self.assertNotIn(result["emotion"], ("calm", "happy", "neutral", "sad"))

    def test_high_arousal_voice_with_tense_physio_not_calm(self):
        result = self.engine.analyze(
            physio={"eda": 18.0, "hrv": 15.0},
            voice={"arousal": 0.9, "valence": 0.3},
        )
        self.assertGreaterEqual(result["arousal"], 0.75)
        self.assertIn(result["emotion"], ("angry", "anxious"))

    def test_relaxed_physio_plus_calm_voice_maps_to_calm(self):
        result = self.engine.analyze(
            physio={"eda": 2.0, "hrv": 95.0},
            voice={"arousal": 0.2, "valence": 0.7},
        )
        self.assertEqual(result["emotion"], "calm")

    def test_all_numeric_outputs_within_unit_interval(self):
        cases = [
            ({"eda": 20.0, "hrv": 10.0}, {"arousal": 1.0, "valence": 0.0}),
            ({"eda": 0.0, "hrv": 100.0}, {"arousal": 0.0, "valence": 1.0}),
            ({"eda": 99.0, "hrv": 1.0}, {"arousal": 1.2, "valence": -0.3}),
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
        self.assertAlmostEqual(result["valence"], 0.0, places=6)

    def test_empty_voice_dict_uses_physio_only(self):
        result = self.engine.analyze(physio={"eda": 20.0, "hrv": 10.0}, voice={})
        self.assertAlmostEqual(result["arousal"], 1.0, places=6)

    def test_missing_physio_does_not_shrink_voice(self):
        # 语音一路不得因生理缺失而被乘 0.4 压小
        result = self.engine.analyze(physio=None, voice={"arousal": 1.0, "valence": 1.0})
        self.assertEqual(result["arousal"], 1.0)
        self.assertEqual(result["valence"], 1.0)

    def test_missing_hrv_component_not_padded_with_half(self):
        self.assertAlmostEqual(
            self.engine.compute_physio_arousal({"eda": 20.0, "hrv": None}),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            self.engine.compute_physio_arousal({"eda": 20.0}), 1.0, places=6
        )
        self.assertAlmostEqual(
            self.engine.compute_physio_arousal({"eda": 0.0}), 0.0, places=6
        )

    def test_missing_eda_component_uses_hrv_only(self):
        self.assertAlmostEqual(
            self.engine.compute_physio_arousal({"eda": None, "hrv": 10.0}),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            self.engine.compute_physio_arousal({"hrv": 100.0}), 0.0, places=6
        )

    def test_partial_voice_dimension_uses_that_dimension_only(self):
        # 语音只给出 arousal：arousal 走双模加权，valence 只用生理
        result = self.engine.analyze(
            physio={"eda": 20.0, "hrv": 10.0}, voice={"arousal": 0.8}
        )
        self.assertAlmostEqual(result["arousal"], 0.92, places=6)
        self.assertAlmostEqual(result["valence"], 0.0, places=6)

    def test_all_signals_missing_return_neutral_zero_confidence(self):
        for physio, voice in (
            (None, None),
            ({}, {}),
            ({"eda": None, "hrv": None}, None),
            (None, {"arousal": None}),
        ):
            with self.subTest(physio=physio, voice=voice):
                result = self.engine.analyze(physio=physio, voice=voice)
                self.assertEqual(result["arousal"], 0.5)
                self.assertEqual(result["valence"], 0.5)
                self.assertEqual(result["emotion"], "neutral")
                self.assertEqual(result["confidence"], 0.0)
                self.assertEqual(result["score"], 0.0)


class TestEmotionMappingAndSchema(unittest.TestCase):
    """情绪锚点区间、score 与公开返回字段保持不变。"""

    def setUp(self):
        self.engine = EmotionFusionEngine()

    def test_anchor_points_map_to_their_own_category(self):
        for name, (arousal, valence) in EMOTION_ANCHORS.items():
            with self.subTest(emotion=name):
                self.assertEqual(
                    self.engine.map_to_emotion_category(arousal, valence), name
                )

    def test_high_arousal_low_valence_is_angry_or_anxious(self):
        self.assertIn(
            self.engine.map_to_emotion_category(0.9, 0.1), ("angry", "anxious")
        )

    def test_low_arousal_is_calm_not_angry(self):
        self.assertEqual(self.engine.map_to_emotion_category(0.2, 0.65), "calm")

    def test_public_result_fields_unchanged(self):
        result = self.engine.analyze(
            physio={"eda": 10.0, "hrv": 50.0},
            voice={"arousal": 0.5, "valence": 0.5},
        )
        self.assertEqual(
            set(result.keys()),
            {"arousal", "valence", "emotion", "confidence", "score"},
        )

    def test_default_arguments_supported(self):
        result = self.engine.analyze()
        self.assertEqual(result["emotion"], "neutral")
        self.assertEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()

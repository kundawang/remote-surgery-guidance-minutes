"""团体心理辅导多模态情绪融合：腕带生理信号（皮电 EDA + 心率变异性 HRV）与语音情绪。

融合原则（修复历史 bug 后的语义，改动点见各方法注释）：

1. arousal / valence 均为 [0, 1] 区间内的"真正的加权平均"：
   每个中间结果都按"权重和 = 1"归一化，任何一路信号都不会被重复打折。
   （历史 bug：生理内部两项各乘 0.3 直接相加，上限只有 0.6；
   融合时又乘 0.6，生理一路被二次打折，整体上限只有 0.76。）

2. 缺失信号的语义：某一路信号缺失（None 或空 dict）表示"没有这份证据"，
   而不是"这份证据取中性值"。因此：
   - 不给缺失项填充 0.5 之类的默认值；
   - 不保留缺失项的权重把结果摊薄；
   - 只在剩余有效信号上重新归一化权重后做加权平均。
   两路信号全部缺失时，返回中性结果且 confidence/score 为 0（无证据）。
"""

from __future__ import annotations

from math import sqrt
from typing import Optional

# ---- 生理信号量程（腕带规格） ----
EDA_MIN, EDA_MAX = 0.0, 20.0   # 皮电，单位 µS
HRV_MIN, HRV_MAX = 10.0, 100.0  # 心率变异性 RMSSD，单位 ms

# ---- 生理内部各分量权重（历史值 0.3 保留，但按权重和归一化后使用） ----
PHYSIO_COMPONENT_WEIGHTS = {"eda": 0.3, "hrv": 0.3}

# ---- 模态间融合权重：生理 0.6 / 语音 0.4 ----
MODALITY_WEIGHTS = {"physio": 0.6, "voice": 0.4}

# ---- 情绪锚点：(arousal, valence) 平面坐标，区间定义保持不变 ----
EMOTION_ANCHORS = {
    "happy": (0.70, 0.85),
    "calm": (0.20, 0.65),
    "neutral": (0.50, 0.50),
    "sad": (0.25, 0.20),
    "angry": (0.75, 0.20),
    "anxious": (0.85, 0.35),
}

# 锚点平面最大距离，用于把"离锚点的距离"换算成 [0,1] 的 score
_MAX_ANCHOR_DISTANCE = sqrt(2.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _weighted_average(components: dict[str, tuple[float, float]]) -> Optional[float]:
    """对 {name: (value, weight)} 做权重归一化的加权平均。

    只统计实际存在的分量；全部缺失时返回 None（表示"没有证据"，
    由调用方决定如何在更大范围内重新归一化），绝不返回 0.5 之类的填充值。
    """
    total_weight = sum(weight for _, weight in components.values())
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in components.values()) / total_weight


class EmotionFusionEngine:
    """腕带生理信号 + 语音情绪的 arousal/valence 融合引擎。"""

    # ---------- 生理信号 ----------

    def compute_physio_arousal(self, physio: Optional[dict]) -> Optional[float]:
        """由皮电 + HRV 计算生理 arousal，落在 [0, 1]。

        皮电越高 arousal 越高；HRV 越低 arousal 越高。
        两个分量按 PHYSIO_COMPONENT_WEIGHTS 归一化加权（权重和恒为 1），
        因此皮电顶格 + HRV 触底时结果为 1.0，反之为 0.0。
        某个分量缺失（键不存在或值为 None）时只用剩余分量；
        整路缺失（None / 空 dict / 无有效分量）时返回 None。
        """
        components: dict[str, tuple[float, float]] = {}
        if physio:
            eda = physio.get("eda")
            if eda is not None:
                eda_norm = _clamp01((float(eda) - EDA_MIN) / (EDA_MAX - EDA_MIN))
                components["eda"] = (eda_norm, PHYSIO_COMPONENT_WEIGHTS["eda"])
            hrv = physio.get("hrv")
            if hrv is not None:
                hrv_norm = _clamp01((float(hrv) - HRV_MIN) / (HRV_MAX - HRV_MIN))
                components["hrv"] = (1.0 - hrv_norm, PHYSIO_COMPONENT_WEIGHTS["hrv"])
        return _weighted_average(components)

    def compute_physio_valence(self, physio: Optional[dict]) -> Optional[float]:
        """由皮电 + HRV 估计生理 valence，落在 [0, 1]。

        生理信号对 valence 的区分度弱，仅作温和先验：
        HRV 越高 valence 越高，皮电越高 valence 略低。
        缺失语义与 compute_physio_arousal 一致。
        """
        components: dict[str, tuple[float, float]] = {}
        if physio:
            eda = physio.get("eda")
            if eda is not None:
                eda_norm = _clamp01((float(eda) - EDA_MIN) / (EDA_MAX - EDA_MIN))
                components["eda"] = (1.0 - eda_norm, PHYSIO_COMPONENT_WEIGHTS["eda"])
            hrv = physio.get("hrv")
            if hrv is not None:
                hrv_norm = _clamp01((float(hrv) - HRV_MIN) / (HRV_MAX - HRV_MIN))
                components["hrv"] = (hrv_norm, PHYSIO_COMPONENT_WEIGHTS["hrv"])
        return _weighted_average(components)

    # ---------- 情绪分类 ----------

    def map_to_emotion_category(self, arousal: float, valence: float) -> str:
        """按 (arousal, valence) 到各情绪锚点的欧氏距离取最近类别。

        锚点坐标与区间保持既有定义，未做改动。
        """
        return min(
            EMOTION_ANCHORS,
            key=lambda name: (arousal - EMOTION_ANCHORS[name][0]) ** 2
            + (valence - EMOTION_ANCHORS[name][1]) ** 2,
        )

    # ---------- 融合入口 ----------

    def analyze(self, physio: Optional[dict] = None, voice: Optional[dict] = None) -> dict:
        """融合生理与语音信号，返回 arousal/valence/emotion/confidence/score。

        physio: {"eda": float, "hrv": float}，允许为 None / {} / 部分键缺失。
        voice:  {"arousal": float, "valence": float}（语音情绪模型输出，[0,1]），
                允许为 None / {}。

        缺失的模态不参与加权：剩余模态的权重重新归一化到和为 1，
        结果不会被缺失方"摊薄"或整体压小。
        """
        physio_arousal = self.compute_physio_arousal(physio)
        physio_valence = self.compute_physio_valence(physio)

        voice_arousal: Optional[float] = None
        voice_valence: Optional[float] = None
        if voice:
            if voice.get("arousal") is not None:
                voice_arousal = _clamp01(float(voice["arousal"]))
            if voice.get("valence") is not None:
                voice_valence = _clamp01(float(voice["valence"]))

        # 模态级加权平均：只统计实际有证据的模态，权重重新归一化。
        arousal_parts: dict[str, tuple[float, float]] = {}
        valence_parts: dict[str, tuple[float, float]] = {}
        if physio_arousal is not None:
            arousal_parts["physio"] = (physio_arousal, MODALITY_WEIGHTS["physio"])
        if voice_arousal is not None:
            arousal_parts["voice"] = (voice_arousal, MODALITY_WEIGHTS["voice"])
        if physio_valence is not None:
            valence_parts["physio"] = (physio_valence, MODALITY_WEIGHTS["physio"])
        if voice_valence is not None:
            valence_parts["voice"] = (voice_valence, MODALITY_WEIGHTS["voice"])

        arousal = _weighted_average(arousal_parts)
        valence = _weighted_average(valence_parts)

        if arousal is None or valence is None:
            # 两路信号全部缺失：没有任何证据，返回中性占位，
            # confidence/score 置 0 以明确表达"这不是一次有效测量"。
            return {
                "arousal": 0.5,
                "valence": 0.5,
                "emotion": "neutral",
                "confidence": 0.0,
                "score": 0.0,
            }

        arousal = _clamp01(arousal)
        valence = _clamp01(valence)
        emotion = self.map_to_emotion_category(arousal, valence)

        # confidence：两路都在时取模态间一致性（差异越小越自信）；
        # 只有一路证据时给固定的中等置信度。
        if physio_arousal is not None and voice_arousal is not None:
            disagreement = sqrt(
                (physio_arousal - voice_arousal) ** 2
                + (physio_valence - voice_valence) ** 2
            ) / _MAX_ANCHOR_DISTANCE
            confidence = _clamp01(1.0 - disagreement)
        else:
            confidence = 0.65

        # score：融合点到所判情绪锚点的贴近程度，[0,1]，越贴近越高。
        anchor = EMOTION_ANCHORS[emotion]
        distance = sqrt((arousal - anchor[0]) ** 2 + (valence - anchor[1]) ** 2)
        score = _clamp01(1.0 - distance / _MAX_ANCHOR_DISTANCE)

        return {
            "arousal": arousal,
            "valence": valence,
            "emotion": emotion,
            "confidence": confidence,
            "score": score,
        }

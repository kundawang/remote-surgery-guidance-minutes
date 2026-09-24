"""团体心理辅导多模态情绪融合：腕带生理信号（皮电 EDA + 心率变异性 HRV）与语音情绪。

融合原则（本文件修复了历史权重 bug）：

1. arousal / valence 一律落在 [0, 1]，且每一层都是"真正的加权平均"：
   参与平均的权重按实际有效分量归一化到和为 1。任何一路信号都不会被
   重复打折（历史 bug：生理内部各乘 0.3 直接相加、未归一化，生理 arousal
   上限只有 0.6；模态融合时又把该结果乘 0.6，生理一路被二次打折，
   融合后 arousal 上限只有 0.6*0.6 + 0.4*1 = 0.76，高唤醒情绪锚点
   angry(0.75)/anxious(0.85) 基本够不着）。

2. 缺失信号的语义：某一路信号缺失（None 或空 dict，或某个分量为 None）
   表示"没有这份证据"，而不是"这份证据取中性值 0.5"。因此：
   - 绝不给缺失项填 0.5 之类的默认值；
   - 绝不保留缺失项的权重把结果摊薄或整体压小；
   - 只在剩余有效信号上重新归一化权重后做加权平均。
   当所有维度都没有任何证据时，返回中性占位 arousal=valence=0.5、
   emotion="neutral"，但 confidence=score=0，以标明这不是一次有效测量。
"""

from __future__ import annotations

import math
from typing import Optional

# ---- 生理信号量程（腕带规格） ----
EDA_MIN, EDA_MAX = 0.0, 20.0     # 皮电，单位 µS
HRV_MIN, HRV_MAX = 10.0, 100.0   # 心率变异性 RMSSD，单位 ms

# ---- 生理内部各分量权重：取值沿用历史配置，但使用时按有效分量归一化 ----
PHYSIO_COMPONENT_WEIGHTS = {"eda": 0.3, "hrv": 0.3}

# ---- 模态间融合权重：生理 0.6 / 语音 0.4，同样按有效模态归一化使用 ----
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

# (arousal, valence) 平面的最大可能距离，用于距离 -> [0,1] 的换算
_MAX_PLANE_DISTANCE = math.sqrt(2.0)

# 只有单模态证据时的固定置信度（两路一致性无从计算）
_SINGLE_MODALITY_CONFIDENCE = 0.65


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _weighted_average(parts: dict[str, tuple[float, float]]) -> Optional[float]:
    """对 {名称: (取值, 权重)} 做权重归一化的加权平均。

    只对实际存在的分量求和；总权重为 0（即没有任何证据）时返回 None，
    由调用方决定降级策略。绝不返回 0.5 之类的中性填充值。
    """
    total_weight = sum(weight for _, weight in parts.values())
    if total_weight <= 0.0:
        return None
    return sum(value * weight for value, weight in parts.values()) / total_weight


class EmotionFusionEngine:
    """腕带生理信号 + 语音情绪的 arousal/valence 融合引擎。"""

    # ---------- 生理信号 ----------

    def compute_physio_arousal(self, physio: Optional[dict]) -> Optional[float]:
        """由皮电 + HRV 计算生理 arousal，结果落在 [0, 1]。

        皮电越高越唤醒；HRV 越低越唤醒。两个分量的权重归一化到和为 1，
        因此皮电顶格(20) + HRV 触底(10) 时结果为 1.0，
        皮电触底(0) + HRV 顶格(100) 时结果为 0.0。
        单个分量缺失（键不存在或值为 None）时只使用剩余分量并重新归一化；
        整路缺失（None / 空 dict / 无有效分量）时返回 None。
        """
        parts: dict[str, tuple[float, float]] = {}
        if physio:
            eda = physio.get("eda")
            if eda is not None:
                eda_norm = _clamp01((float(eda) - EDA_MIN) / (EDA_MAX - EDA_MIN))
                parts["eda"] = (eda_norm, PHYSIO_COMPONENT_WEIGHTS["eda"])
            hrv = physio.get("hrv")
            if hrv is not None:
                hrv_norm = _clamp01((float(hrv) - HRV_MIN) / (HRV_MAX - HRV_MIN))
                parts["hrv"] = (1.0 - hrv_norm, PHYSIO_COMPONENT_WEIGHTS["hrv"])
        return _weighted_average(parts)

    def compute_physio_valence(self, physio: Optional[dict]) -> Optional[float]:
        """由皮电 + HRV 估计生理 valence，结果落在 [0, 1]。

        生理信号对 valence 的区分度弱，仅作温和先验：
        HRV 越高、皮电越低，valence 越高。缺失语义同
        compute_physio_arousal：缺什么就只在剩余分量上重新归一化。
        """
        parts: dict[str, tuple[float, float]] = {}
        if physio:
            eda = physio.get("eda")
            if eda is not None:
                eda_norm = _clamp01((float(eda) - EDA_MIN) / (EDA_MAX - EDA_MIN))
                parts["eda"] = (1.0 - eda_norm, PHYSIO_COMPONENT_WEIGHTS["eda"])
            hrv = physio.get("hrv")
            if hrv is not None:
                hrv_norm = _clamp01((float(hrv) - HRV_MIN) / (HRV_MAX - HRV_MIN))
                parts["hrv"] = (hrv_norm, PHYSIO_COMPONENT_WEIGHTS["hrv"])
        return _weighted_average(parts)

    # ---------- 情绪分类 ----------

    def map_to_emotion_category(self, arousal: float, valence: float) -> str:
        """按 (arousal, valence) 到各情绪锚点的欧氏距离取最近类别。

        锚点坐标与 score 的区间定义均保持既有定义，未做改动。
        """
        def distance_squared(name: str) -> float:
            anchor_arousal, anchor_valence = EMOTION_ANCHORS[name]
            return (arousal - anchor_arousal) ** 2 + (valence - anchor_valence) ** 2

        return min(EMOTION_ANCHORS, key=distance_squared)

    # ---------- 融合入口 ----------

    def analyze(
        self,
        physio: Optional[dict] = None,
        voice: Optional[dict] = None,
    ) -> dict:
        """融合生理与语音信号，返回 arousal/valence/emotion/confidence/score。

        physio: {"eda": float, "hrv": float}，允许 None / {} / 部分分量缺失。
        voice:  {"arousal": float, "valence": float}（语音情绪模型输出），
                允许 None / {} / 部分分量缺失；超出 [0,1] 的值会被截断。

        每个维度独立处理缺失：缺失的模态不参与该维度的加权，剩余模态的
        权重重新归一化到和为 1，结果既不会被缺失方填 0.5 拉向中性，也
        不会被"保留权重"摊薄或整体压小。
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

        arousal_parts: dict[str, tuple[float, float]] = {}
        valence_parts: dict[str, tuple[float, float]] = {}
        if physio_arousal is not None:
            arousal_parts["physio"] = (physio_arousal, MODALITY_WEIGHTS["physio"])
        if physio_valence is not None:
            valence_parts["physio"] = (physio_valence, MODALITY_WEIGHTS["physio"])
        if voice_arousal is not None:
            arousal_parts["voice"] = (voice_arousal, MODALITY_WEIGHTS["voice"])
        if voice_valence is not None:
            valence_parts["voice"] = (voice_valence, MODALITY_WEIGHTS["voice"])

        arousal = _weighted_average(arousal_parts)
        valence = _weighted_average(valence_parts)

        if arousal is None or valence is None:
            # 某一维度两路证据全部缺失：没有可报告的测量结果。
            # 用中性坐标占位并把 confidence/score 置 0，明确"无证据"。
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

        # confidence：两路都在的维度上比较模态间一致性，差异越小越自信；
        # 若某维度只有一路证据，则该维度无法比较，使用固定中等置信度。
        shared_dimensions = [
            (physio_arousal, voice_arousal),
            (physio_valence, voice_valence),
        ]
        shared_dimensions = [
            pair for pair in shared_dimensions if pair[0] is not None and pair[1] is not None
        ]
        if shared_dimensions:
            disagreement = math.sqrt(
                sum((a - b) ** 2 for a, b in shared_dimensions)
            ) / (math.sqrt(float(len(shared_dimensions))) * _MAX_PLANE_DISTANCE)
            confidence = _clamp01(1.0 - disagreement)
        else:
            confidence = _SINGLE_MODALITY_CONFIDENCE

        # score：融合点到所判情绪锚点的贴近程度，[0,1]，越贴近越高。
        anchor_arousal, anchor_valence = EMOTION_ANCHORS[emotion]
        distance = math.sqrt(
            (arousal - anchor_arousal) ** 2 + (valence - anchor_valence) ** 2
        )
        score = _clamp01(1.0 - distance / _MAX_PLANE_DISTANCE)

        return {
            "arousal": arousal,
            "valence": valence,
            "emotion": emotion,
            "confidence": confidence,
            "score": score,
        }

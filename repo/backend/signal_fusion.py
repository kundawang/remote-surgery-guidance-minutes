"""静默会议 BCI / 音频双路信号融合。

将 BCI 解码文本与语音转写文本按时间戳对齐、合并，
输出统一的融合分段、拼接文本与整体置信度。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TextSegment:
    """单路（BCI 或音频）识别出的文本分段。"""

    text: str
    start: float
    end: float
    confidence: float


@dataclass
class FusedSegment:
    """融合后的文本分段。"""

    text: str
    start: float
    end: float
    confidence: float
    sources: List[str] = field(default_factory=list)


class SignalFusion:
    """BCI 与音频双路信号融合器。

    - 时间对齐：|start 差值| 不超过 max_delay 的两路段配成一对，
      每段最多参与一次配对，未配上的段单独成段。
    - 置信度：双路配对时按 (bci_weight, audio_weight) 归一化加权平均；
      单路时直接采用该路自身的置信度，不做权重打折。
    - 文本合并：词重叠比例达到 overlap_threshold 时取更长的一路，
      否则按 BCI 在前、音频在后的固定顺序拼接。
    """

    def __init__(
        self,
        bci_weight: float = 0.4,
        audio_weight: float = 0.6,
        max_delay: float = 1.0,
        overlap_threshold: float = 0.5,
    ) -> None:
        self.bci_weight = bci_weight
        self.audio_weight = audio_weight
        self.max_delay = max_delay
        self.overlap_threshold = overlap_threshold
        self._bci_buffer: List[TextSegment] = []
        self._audio_buffer: List[TextSegment] = []
        self._fused_segments: List[FusedSegment] = []

    def add_bci_segment(self, segment: TextSegment) -> None:
        self._bci_buffer.append(segment)

    def add_audio_segment(self, segment: TextSegment) -> None:
        self._audio_buffer.append(segment)

    def _align_timestamps(
        self,
    ) -> List[Tuple[Optional[TextSegment], Optional[TextSegment]]]:
        """按时间戳贪心配对，返回 (bci, audio) 二元组列表（可为单路）。"""
        pairs: List[Tuple[Optional[TextSegment], Optional[TextSegment]]] = []
        used_audio_indices = set()
        for bci in self._bci_buffer:
            best_index = None
            best_delay = None
            for index, audio in enumerate(self._audio_buffer):
                if index in used_audio_indices:
                    continue
                delay = abs(bci.start - audio.start)
                if delay > self.max_delay:
                    continue
                if best_delay is None or delay < best_delay:
                    best_delay = delay
                    best_index = index
            if best_index is not None:
                used_audio_indices.add(best_index)
                pairs.append((bci, self._audio_buffer[best_index]))
            else:
                pairs.append((bci, None))
        for index, audio in enumerate(self._audio_buffer):
            if index not in used_audio_indices:
                pairs.append((None, audio))
        pairs.sort(key=lambda pair: (pair[0] or pair[1]).start)
        return pairs

    def _merge_texts(self, bci_text: str, audio_text: str) -> str:
        bci_words = bci_text.split()
        audio_words = audio_text.split()
        if not bci_words:
            return audio_text
        if not audio_words:
            return bci_text
        common = set(bci_words) & set(audio_words)
        overlap_ratio = len(common) / min(len(bci_words), len(audio_words))
        if overlap_ratio >= self.overlap_threshold:
            # 取更长的那路；词数相同比字符数，再相同固定取音频路，保证确定性
            if len(bci_words) != len(audio_words):
                return bci_text if len(bci_words) > len(audio_words) else audio_text
            if len(bci_text) != len(audio_text):
                return bci_text if len(bci_text) > len(audio_text) else audio_text
            return audio_text
        return "{} {}".format(bci_text, audio_text)

    def _fuse_pair(
        self,
        bci: Optional[TextSegment],
        audio: Optional[TextSegment],
    ) -> FusedSegment:
        if bci is not None and audio is not None:
            total_weight = self.bci_weight + self.audio_weight
            confidence = (
                bci.confidence * self.bci_weight
                + audio.confidence * self.audio_weight
            ) / total_weight
            return FusedSegment(
                text=self._merge_texts(bci.text, audio.text),
                start=min(bci.start, audio.start),
                end=max(bci.end, audio.end),
                confidence=confidence,
                sources=["bci", "audio"],
            )
        segment = bci if bci is not None else audio
        return FusedSegment(
            text=segment.text,
            start=segment.start,
            end=segment.end,
            confidence=segment.confidence,
            sources=["bci"] if bci is not None else ["audio"],
        )

    def _overall_confidence(self, segments: List[FusedSegment]) -> float:
        if not segments:
            return 0.0
        durations = [max(segment.end - segment.start, 0.0) for segment in segments]
        total_duration = sum(durations)
        if total_duration <= 0:
            return sum(s.confidence for s in segments) / len(segments)
        return (
            sum(s.confidence * d for s, d in zip(segments, durations))
            / total_duration
        )

    def fuse(self) -> Dict:
        pairs = self._align_timestamps()
        segments = [self._fuse_pair(bci, audio) for bci, audio in pairs]
        self._fused_segments = segments
        fused_text = " ".join(s.text for s in segments if s.text)
        return {
            "segments": segments,
            "fused_text": fused_text,
            "overall_confidence": self._overall_confidence(segments),
        }

    def get_fused_segments(self) -> List[FusedSegment]:
        if not self._fused_segments and (self._bci_buffer or self._audio_buffer):
            self.fuse()
        return list(self._fused_segments)

    def clear_buffers(self) -> None:
        self._bci_buffer.clear()
        self._audio_buffer.clear()
        self._fused_segments = []

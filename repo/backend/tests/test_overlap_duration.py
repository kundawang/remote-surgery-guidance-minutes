"""回归测试：重叠区间只算一次，总时长不超过音频实际长度。"""
from app.models.schemas import TranscriptSegment
from app.services.speaker_diarization import SpeakerDiarization
from app.services.summary_generator import SummaryGenerator
from app.utils.intervals import merge_intervals, total_duration


class TestOverlapDuration:
    def test_merge_overlapping_intervals(self):
        assert merge_intervals([(0, 10), (5, 15)]) == [(0.0, 15.0)]
        assert merge_intervals([(5, 15), (0, 10)]) == [(0.0, 15.0)]
        assert merge_intervals([]) == []

    def test_total_duration_counts_overlap_once(self):
        assert total_duration([(0, 10), (5, 15)]) == 15
        assert total_duration([(0, 10), (20, 30)]) == 20
        assert total_duration([]) == 0

    def test_total_duration_capped_by_audio_length(self):
        assert total_duration([(0, 10), (5, 15)], audio_duration=12) == 12
        assert total_duration([(0, 5)], audio_duration=12) == 5

    def test_speaker_statistics_merge_overlap(self):
        diarization = SpeakerDiarization()
        segments = [
            {"speaker": "SPEAKER_00", "start_time": 0, "end_time": 10, "duration": 10},
            {"speaker": "SPEAKER_00", "start_time": 5, "end_time": 15, "duration": 10},
        ]
        stats = diarization.get_speaker_statistics(segments)
        assert stats["SPEAKER_00"]["total_duration"] == 15
        assert stats["SPEAKER_00"]["segment_count"] == 2

    def test_session_total_not_exceed_audio_length(self):
        diarization = SpeakerDiarization()
        segments = [
            {"speaker": "SPEAKER_00", "start_time": 0, "end_time": 10},
            {"speaker": "SPEAKER_01", "start_time": 5, "end_time": 15},
        ]
        assert diarization.get_total_duration(segments) == 15
        assert diarization.get_total_duration(segments, audio_duration=12) == 12

    def test_overall_assessment_uses_merged_duration(self):
        generator = SummaryGenerator()
        transcripts = [
            TranscriptSegment(
                speaker="发音人1", speaker_role="主刀医生",
                start_time=0, end_time=600, text="第一段", confidence=0.9,
            ),
            TranscriptSegment(
                speaker="发音人1", speaker_role="主刀医生",
                start_time=300, end_time=900, text="第二段", confidence=0.9,
            ),
        ]
        assessment = generator._generate_overall_assessment(
            transcripts, {"surgery_type": "测试手术"}
        )
        assert "手术总时长约15分钟" in assessment
        assert "主刀医生发言约15分钟" in assessment

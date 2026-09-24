from datetime import datetime
from types import SimpleNamespace

from app.services import email_archiver as email_archiver_module
from app.services.email_archiver import EmailArchiver
from app.services.summary_generator import SummaryGenerator
from app.models.schemas import SurgerySummaryResponse, TranscriptSegment
from app.utils.time_utils import format_timestamp


def make_session(start_time):
    return SimpleNamespace(
        session_id="SURG-20260924-TEST01",
        patient_name="张三",
        patient_id="P001",
        surgery_type="腹腔镜胆囊切除术",
        primary_surgeon="李医生",
        remote_expert="王专家",
        operating_room="OR-1",
        start_time=start_time,
        end_time=None,
    )


def make_summary():
    return SurgerySummaryResponse(
        key_points=["要点一"],
        surgical_steps=[{"time": 3725.0, "step": "分离", "description": "分离组织"}],
        anatomical_landmarks=["胆囊管"],
        technical_improvements=[],
        complications=[],
        overall_assessment="手术顺利。",
    )


def make_transcripts():
    return [
        TranscriptSegment(
            speaker="SPEAKER_00",
            speaker_role="主刀医生",
            start_time=12.5,
            end_time=15.0,
            text="开始分离。",
            confidence=0.95,
        ),
        TranscriptSegment(
            speaker="SPEAKER_01",
            speaker_role="远程专家",
            start_time=3725.0,
            end_time=3730.0,
            text="注意胆管位置。",
            confidence=0.9,
        ),
    ]


class TestFormatTimestamp:
    def test_float_seconds(self):
        assert format_timestamp(12.5) == "00:12"
        assert format_timestamp(3725.0) == "1:02:05"

    def test_int_seconds(self):
        assert format_timestamp(12) == "00:12"
        assert format_timestamp(3725) == "1:02:05"

    def test_boundaries(self):
        assert format_timestamp(0) == "00:00"
        assert format_timestamp(59) == "00:59"
        assert format_timestamp(60) == "01:00"
        assert format_timestamp(3599) == "59:59"
        assert format_timestamp(3600) == "1:00:00"

    def test_formatted_string_passthrough(self):
        assert format_timestamp("12:05") == "12:05"
        assert format_timestamp("1:02:05") == "1:02:05"

    def test_numeric_string(self):
        assert format_timestamp("3725.0") == "1:02:05"
        assert format_timestamp("12") == "00:12"

    def test_none_and_empty(self):
        assert format_timestamp(None) == "00:00"
        assert format_timestamp("") == "00:00"

    def test_archiver_and_generator_share_format(self):
        assert EmailArchiver().format_time(3725.0) == "1:02:05"
        assert SummaryGenerator()._format_time(3725.0) == "1:02:05"


class TestMeetingTimeSource:
    def test_header_uses_session_start_time(self):
        archiver = EmailArchiver()
        session = make_session(datetime(2026, 9, 24, 9, 30, 0))
        html = archiver._render_email(session, make_summary(), make_transcripts())
        assert "会议时间: 2026-09-24 09:30:00" in html

    def test_plain_text_uses_session_start_time(self):
        archiver = EmailArchiver()
        session = make_session(datetime(2026, 9, 24, 9, 30, 0))
        text = archiver._render_plain_text(session, make_summary(), make_transcripts())
        assert "会议时间: 2026-09-24 09:30:00" in text

    def test_meeting_time_not_generation_time(self):
        archiver = EmailArchiver()
        session = make_session(datetime(2020, 1, 2, 3, 4, 5))
        html = archiver._render_email(session, make_summary(), make_transcripts())
        assert "会议时间: 2020-01-02 03:04:05" in html
        assert "会议时间: " + datetime.now().strftime("%Y-%m-%d") not in html

    def test_string_start_time_used_verbatim(self):
        archiver = EmailArchiver()
        session = make_session("2026-09-24 09:30")
        html = archiver._render_email(session, make_summary(), make_transcripts())
        assert "会议时间: 2026-09-24 09:30" in html

    def test_missing_meeting_time_shows_placeholder(self):
        archiver = EmailArchiver()
        for missing in (None, ""):
            session = make_session(missing)
            html = archiver._render_email(session, make_summary(), make_transcripts())
            text = archiver._render_plain_text(session, make_summary(), make_transcripts())
            assert "会议时间: 未记录" in html
            assert "会议时间: 未记录" in text


class TestGeneratedAtFreshness:
    def test_same_instance_generates_fresh_timestamps(self, monkeypatch):
        now_values = [datetime(2026, 9, 24, 10, 0, 0), datetime(2026, 9, 24, 10, 0, 5)]
        calls = {"n": 0}

        class FakeDatetime:
            @classmethod
            def now(cls):
                value = now_values[min(calls["n"], len(now_values) - 1)]
                calls["n"] += 1
                return value

        monkeypatch.setattr(email_archiver_module, "datetime", FakeDatetime)

        archiver = EmailArchiver()
        session = make_session(datetime(2026, 9, 24, 9, 30, 0))
        first = archiver._render_email(session, make_summary(), make_transcripts())
        second = archiver._render_email(session, make_summary(), make_transcripts())

        assert "生成时间: 2026-09-24 10:00:00" in first
        assert "生成时间: 2026-09-24 10:00:05" in second


class TestTranscriptTimestamps:
    def test_report_formats_transcript_timestamps(self):
        archiver = EmailArchiver()
        session = make_session(datetime(2026, 9, 24, 9, 30, 0))
        html = archiver._render_email(session, make_summary(), make_transcripts())
        text = archiver._render_plain_text(session, make_summary(), make_transcripts())
        for report in (html, text):
            assert "1:02:05" in report
            assert "00:12" in report
            assert "3725.0" not in report
            assert "12.5" not in report

    def test_summary_generator_prompt_formats_timestamps(self):
        generator = SummaryGenerator()
        formatted = generator._format_transcripts(make_transcripts())
        assert "[00:12]" in formatted
        assert "[1:02:05]" in formatted
        assert "3725.0" not in formatted

"""回归测试：零转写不崩溃、状态不停在中间态，失败写 error，重跑恢复 completed。"""
import pytest

from app.core.database import Transcript
from app.models.schemas import TranscriptSegment
from app.services.transcription_service import TranscriptionService


def _make_segment(speaker="SPEAKER_00"):
    return TranscriptSegment(
        speaker=speaker,
        speaker_role="其他",
        start_time=0.0,
        end_time=5.0,
        text="有效语音",
        confidence=0.9,
    )


class TestEmptyTranscription:
    def test_empty_result_sets_terminal_status(self, db, surgery_session, monkeypatch):
        service = TranscriptionService()
        monkeypatch.setattr(service, "transcribe_segment", lambda path: [])

        service.transcribe_and_save(surgery_session.id, "noise.wav", db)

        db.refresh(surgery_session)
        assert surgery_session.status == "no_valid_speech"
        assert surgery_session.error_message
        assert db.query(Transcript).filter_by(session_id=surgery_session.id).count() == 0

    def test_failure_sets_error_status_with_reason(self, db, surgery_session, monkeypatch):
        service = TranscriptionService()

        def boom(path):
            raise RuntimeError("audio decode failed")

        monkeypatch.setattr(service, "transcribe_segment", boom)

        with pytest.raises(RuntimeError):
            service.transcribe_and_save(surgery_session.id, "bad.wav", db)

        db.refresh(surgery_session)
        assert surgery_session.status == "error"
        assert "audio decode failed" in surgery_session.error_message

    def test_rerun_recovers_to_completed(self, db, surgery_session, monkeypatch):
        service = TranscriptionService()
        monkeypatch.setattr(service, "transcribe_segment", lambda path: [])
        service.transcribe_and_save(surgery_session.id, "noise.wav", db)
        db.refresh(surgery_session)
        assert surgery_session.status == "no_valid_speech"

        monkeypatch.setattr(service, "transcribe_segment", lambda path: [_make_segment()])
        service.transcribe_and_save(surgery_session.id, "good.wav", db)

        db.refresh(surgery_session)
        assert surgery_session.status == "completed"
        assert surgery_session.error_message is None
        assert db.query(Transcript).filter_by(session_id=surgery_session.id).count() == 1

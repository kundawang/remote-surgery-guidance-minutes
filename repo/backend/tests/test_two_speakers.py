"""回归测试：同一会话两个发音人必须得到不同的编号和姓名，且重跑幂等。"""
from app.core.database import Speaker, Transcript
from app.models.schemas import TranscriptSegment
from app.services.speaker_diarization import SpeakerDiarization
from app.services.transcription_service import TranscriptionService


def _make_segment(speaker, start, end, text="测试语音"):
    return TranscriptSegment(
        speaker=speaker,
        speaker_role="其他",
        start_time=start,
        end_time=end,
        text=text,
        confidence=0.9,
    )


class TestTwoSpeakers:
    def test_each_label_gets_unique_index_and_name(self, db, surgery_session):
        diarization = SpeakerDiarization()
        speakers = diarization.resolve_session_speakers(
            db, surgery_session.id, ["SPEAKER_00", "SPEAKER_01"]
        )
        db.commit()

        assert len(speakers) == 2
        names = {speaker.name for speaker in speakers.values()}
        assert len(names) == 2
        indices = sorted(speaker.speaker_index for speaker in speakers.values())
        assert indices == [1, 2]

    def test_reprocess_is_idempotent(self, db, surgery_session):
        diarization = SpeakerDiarization()
        first = diarization.resolve_session_speakers(
            db, surgery_session.id, ["SPEAKER_00", "SPEAKER_01"]
        )
        db.commit()
        first_ids = {label: speaker.id for label, speaker in first.items()}

        second = diarization.resolve_session_speakers(
            db, surgery_session.id, ["SPEAKER_00", "SPEAKER_01"]
        )
        db.commit()

        assert db.query(Speaker).filter_by(session_id=surgery_session.id).count() == 2
        assert {label: speaker.id for label, speaker in second.items()} == first_ids

    def test_new_label_continues_numbering(self, db, surgery_session):
        diarization = SpeakerDiarization()
        diarization.resolve_session_speakers(db, surgery_session.id, ["SPEAKER_00"])
        db.commit()

        speakers = diarization.resolve_session_speakers(
            db, surgery_session.id, ["SPEAKER_00", "SPEAKER_01"]
        )
        db.commit()

        assert speakers["SPEAKER_00"].speaker_index == 1
        assert speakers["SPEAKER_01"].speaker_index == 2
        assert speakers["SPEAKER_00"].name != speakers["SPEAKER_01"].name

    def test_transcribe_and_save_assigns_distinct_names(self, db, surgery_session, monkeypatch):
        service = TranscriptionService()
        segments = [
            _make_segment("SPEAKER_00", 0, 5),
            _make_segment("SPEAKER_01", 5, 10),
        ]
        monkeypatch.setattr(service, "transcribe_segment", lambda path: segments)

        service.transcribe_and_save(surgery_session.id, "dummy.wav", db)
        service.transcribe_and_save(surgery_session.id, "dummy.wav", db)

        speakers = db.query(Speaker).filter_by(session_id=surgery_session.id).all()
        assert len(speakers) == 2
        assert len({speaker.name for speaker in speakers}) == 2

        transcripts = db.query(Transcript).filter_by(session_id=surgery_session.id).all()
        assert {t.speaker for t in transcripts} == {speaker.name for speaker in speakers}

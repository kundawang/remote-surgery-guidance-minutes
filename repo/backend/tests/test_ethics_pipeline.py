import json
import os
import struct
import wave
from email import message_from_bytes
from email.header import decode_header

import pytest

from app.pipeline import tasks
from app.pipeline.config import DEFAULT_CONFIG
from app.pipeline.tasks import (
    ArchiveMinutesTask,
    GenerateSummaryTask,
    ProcessAudioTask,
    TranscribeAudioTask,
)
from app.pipeline.workflow import main, run_workflow

MEETING_ID = "ETHICS-2026-001"
MEETING_TITLE = "基因编辑伦理复审研讨会"
MEETING_DATE = "2026-09-24"
COMMITTEE_EMAIL = "ethics@hospital.example.org"
REVIEWERS = ["alice@example.org", "bob@example.org"]


def _make_wav(path, seconds=0.5, framerate=8000):
    frames = int(seconds * framerate)
    samples = []
    for i in range(frames):
        samples.append(50 if i % 2 == 0 else 5000)  # 底噪 + 有效信号交替
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(framerate)
        writer.writeframes(struct.pack("<%dh" % len(samples), *samples))


@pytest.fixture()
def meeting(tmp_path):
    input_audio = tmp_path / "raw" / "meeting_raw.wav"
    _make_wav(str(input_audio))
    return {
        "meeting_id": MEETING_ID,
        "meeting_title": MEETING_TITLE,
        "meeting_date": MEETING_DATE,
        "input_audio_path": str(input_audio),
        "output_dir": str(tmp_path / "out"),
        "ethics_committee_email": COMMITTEE_EMAIL,
        "reviewer_emails": list(REVIEWERS),
    }


def _paths(kwargs):
    base = os.path.join(kwargs["output_dir"], kwargs["meeting_id"])
    mid = kwargs["meeting_id"]
    return {
        "audio": os.path.join(base, "audio", f"{mid}_cleaned_audio.wav"),
        "status": os.path.join(base, "audio", f"{mid}_audio_status.json"),
        "transcript": os.path.join(base, "transcript", f"{mid}_transcript.json"),
        "summary": os.path.join(base, "summary", f"{mid}_summary.md"),
        "email": os.path.join(base, "email", f"{mid}_review_email.eml"),
        "archive": os.path.join(base, "archive", f"{mid}_minutes_archive.json"),
    }


def test_artifacts_do_not_overwrite_each_other(meeting):
    assert run_workflow(**meeting)
    paths = _paths(meeting)

    # 清洗后的音频必须是可解析的 WAV 音频，而不是被状态 JSON 覆盖
    with wave.open(paths["audio"], "rb") as reader:
        assert reader.getframerate() == 8000
        assert reader.getnframes() > 0
    with open(paths["audio"], "rb") as fh:
        assert fh.read(4) == b"RIFF"

    # 状态文件是独立的 JSON，与音频文件分离
    with open(paths["status"], "r", encoding="utf-8") as fh:
        status = json.load(fh)
    assert status["status"] == "ok"
    assert status["cleaned_audio_path"] == paths["audio"]

    # 各阶段产物互不覆盖、内容各异
    contents = set()
    for key in ("audio", "status", "transcript", "summary", "email", "archive"):
        assert os.path.exists(paths[key]), f"缺少产物 {paths[key]}"
        with open(paths[key], "rb") as fh:
            contents.add(fh.read())
    assert len(contents) == 6


def test_parameters_propagate_to_all_outputs(meeting):
    assert run_workflow(**meeting)
    paths = _paths(meeting)

    with open(paths["email"], "rb") as fh:
        message = message_from_bytes(fh.read())
    subject, charset = decode_header(message["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(charset or "utf-8")
    assert MEETING_TITLE in subject
    assert message["From"] == COMMITTEE_EMAIL
    assert message["To"] == ", ".join(REVIEWERS)
    body = message.get_payload(decode=True).decode("utf-8")
    assert MEETING_TITLE in body
    assert MEETING_DATE in body

    with open(paths["summary"], "r", encoding="utf-8") as fh:
        summary = fh.read()
    assert MEETING_TITLE in summary
    assert COMMITTEE_EMAIL in summary

    with open(paths["archive"], "r", encoding="utf-8") as fh:
        archive = json.load(fh)
    assert archive["meeting_title"] == MEETING_TITLE
    assert archive["ethics_committee_email"] == COMMITTEE_EMAIL
    assert archive["reviewer_emails"] == REVIEWERS

    # 传入的参数不能被 MeetingConfig 默认值覆盖
    assert DEFAULT_CONFIG.meeting_title not in summary
    assert DEFAULT_CONFIG.ethics_committee_email not in body
    assert DEFAULT_CONFIG.ethics_committee_email not in summary


def test_rerun_does_not_resend_email(meeting, monkeypatch):
    sent = []

    class RecordingSender(tasks.EmailSender):
        def send(self, message, out_path):
            sent.append(message["Subject"])
            super().send(message, out_path)

    monkeypatch.setattr(tasks, "_default_sender", RecordingSender())
    assert run_workflow(**meeting)
    assert run_workflow(**meeting)  # 已完成任务重跑必须幂等
    assert len(sent) == 1


def test_missing_upstream_artifact_raises(meeting):
    task = GenerateSummaryTask(**meeting)
    with pytest.raises(RuntimeError, match="TranscribeAudioTask"):
        task.run()

    archive_task = ArchiveMinutesTask(**meeting)
    with pytest.raises(RuntimeError, match="SendReviewEmailTask"):
        archive_task.run()


def test_transcription_rejects_non_audio(meeting, tmp_path):
    # 模拟旧 bug：清洗音频路径被写成了 JSON 文本时必须报错而不是喂给 Whisper
    fake_audio = tmp_path / "fake_cleaned_audio.wav"
    fake_audio.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ProcessAudioTask"):
        tasks.transcribe_audio(str(fake_audio))


def test_missing_input_audio_raises(meeting):
    meeting["input_audio_path"] = os.path.join(meeting["output_dir"], "nope.wav")
    task = ProcessAudioTask(**meeting)
    with pytest.raises(RuntimeError, match="ProcessAudioTask"):
        task.run()


def test_cli_entry_matches_run_workflow(meeting, tmp_path):
    cli_output = str(tmp_path / "cli_out")
    ok = main([
        "--meeting-id", meeting["meeting_id"],
        "--meeting-title", meeting["meeting_title"],
        "--meeting-date", meeting["meeting_date"],
        "--input-audio-path", meeting["input_audio_path"],
        "--output-dir", cli_output,
        "--ethics-committee-email", meeting["ethics_committee_email"],
        "--reviewer-emails", ",".join(meeting["reviewer_emails"]),
    ])
    assert ok
    cli_meeting = dict(meeting, output_dir=cli_output)
    paths = _paths(cli_meeting)
    with open(paths["archive"], "r", encoding="utf-8") as fh:
        archive = json.load(fh)
    assert archive["meeting_title"] == MEETING_TITLE
    assert archive["ethics_committee_email"] == COMMITTEE_EMAIL
    assert archive["reviewer_emails"] == REVIEWERS
    with open(paths["audio"], "rb") as fh:
        assert fh.read(4) == b"RIFF"

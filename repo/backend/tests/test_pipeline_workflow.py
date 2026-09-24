"""基因伦理研讨会 luigi 流水线回归测试。

覆盖两类历史缺陷：
1. 产物互相覆盖：状态 JSON 曾写进 cleaned_audio.wav，导致下游把 JSON 当音频喂给 Whisper；
2. 参数被 MeetingConfig 默认值覆盖：复审邮件里的会议标题 / 委员会邮箱不是入口传入的值。
以及幂等性（重跑不重复发邮件）和上游产物缺失时的显式报错。
"""

import json
from pathlib import Path

import pytest

from app.pipeline import services, workflow

RAW_AUDIO = b"RAW-AUDIO-BYTES"
CLEANED_AUDIO = b"CLEANED-WAV-BYTES"


def make_params(tmp_path):
    input_audio = tmp_path / "raw_input.wav"
    input_audio.write_bytes(RAW_AUDIO)
    return {
        "meeting_id": "ethics-2026-0924",
        "meeting_title": "CRISPR 生殖系编辑伦理研讨会",
        "meeting_date": "2026-09-24",
        "input_audio_path": str(input_audio),
        "output_dir": str(tmp_path / "out"),
        "ethics_committee_email": "ethics-board@example.org",
        "reviewer_emails": ["alice@example.org", "bob@example.org"],
    }


@pytest.fixture
def fake_services(monkeypatch):
    """替换外部服务，记录发送出去的邮件；转写步骤校验读到的是音频而非 JSON。"""
    sent_emails = []

    def fake_clean_audio(input_path, output_path):
        assert Path(input_path).read_bytes() == RAW_AUDIO
        Path(output_path).write_bytes(CLEANED_AUDIO)
        return {"sample_rate": 16000, "duration_seconds": 1.0}

    def fake_transcribe(audio_path):
        content = Path(audio_path).read_bytes()
        assert content == CLEANED_AUDIO, (
            f"转写步骤读到的不是清洗后的音频产物：{content!r}"
        )
        return {
            "text": "讨论基因编辑的伦理边界。",
            "language": "zh",
            "segments": [{"start": 0.0, "end": 1.0, "text": "讨论基因编辑的伦理边界。"}],
        }

    def fake_generate_summary(transcript, meeting):
        return {
            "meeting_title": meeting["meeting_title"],
            "meeting_date": meeting["meeting_date"],
            "summary_text": "摘要：" + transcript["text"],
            "key_points": ["伦理边界需委员会复审"],
        }

    def fake_send_email(subject, html_body, to, cc=None):
        sent_emails.append(
            {"subject": subject, "body": html_body, "to": list(to), "cc": list(cc or [])}
        )

    monkeypatch.setattr(services, "clean_audio", fake_clean_audio)
    monkeypatch.setattr(services, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(services, "generate_summary", fake_generate_summary)
    monkeypatch.setattr(services, "send_email", fake_send_email)
    return sent_emails


def test_artifacts_do_not_overwrite_each_other(tmp_path, fake_services):
    params = make_params(tmp_path)
    assert workflow.run_workflow(**params)

    paths = workflow.artifact_paths(params["output_dir"], params["meeting_id"])

    # 所有产物路径互不相同且都存在
    assert len(set(paths.values())) == len(paths)
    for path in paths.values():
        assert Path(path).exists(), f"缺少产物：{path}"

    # 清洗后的音频是音频产物本身，不是 JSON 文本
    audio_bytes = Path(paths["cleaned_audio"]).read_bytes()
    assert audio_bytes == CLEANED_AUDIO
    with pytest.raises(json.JSONDecodeError):
        json.loads(audio_bytes)

    # 状态写在独立的 json 文件里，且指向音频产物
    status = json.loads(Path(paths["clean_status"]).read_text(encoding="utf-8"))
    assert status["cleaned_audio_path"] == paths["cleaned_audio"]
    assert status["input_audio_path"] == params["input_audio_path"]

    # 下游产物各自是可解析的 JSON，且显式记录上游产物路径
    transcript = json.loads(Path(paths["transcript"]).read_text(encoding="utf-8"))
    assert transcript["source_audio_path"] == paths["cleaned_audio"]
    assert transcript["transcript"]["text"] == "讨论基因编辑的伦理边界。"

    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["source_transcript_path"] == paths["transcript"]

    record = json.loads(Path(paths["review_email"]).read_text(encoding="utf-8"))
    assert record["source_summary_path"] == paths["summary"]


def test_parameters_propagate_to_review_email(tmp_path, fake_services):
    params = make_params(tmp_path)
    assert workflow.run_workflow(**params)

    assert len(fake_services) == 1
    mail = fake_services[0]
    assert params["meeting_title"] in mail["subject"]
    assert params["meeting_date"] in mail["subject"]
    assert mail["to"] == [params["ethics_committee_email"]]
    assert mail["cc"] == params["reviewer_emails"]
    assert params["meeting_title"] in mail["body"]
    assert params["ethics_committee_email"] in mail["body"]

    # 归档产物里记录的也是入口传入的参数
    paths = workflow.artifact_paths(params["output_dir"], params["meeting_id"])
    record = json.loads(Path(paths["review_email"]).read_text(encoding="utf-8"))
    assert record["meeting"]["meeting_title"] == params["meeting_title"]
    assert record["meeting"]["ethics_committee_email"] == params["ethics_committee_email"]
    assert record["meeting"]["reviewer_emails"] == params["reviewer_emails"]


def test_requires_chain_propagates_parameters(tmp_path):
    """任务图上每一层拿到的都是入口参数，不存在默认值覆盖。"""
    params = make_params(tmp_path)
    task = workflow.SendReviewEmailTask(**params)
    seen = []
    node = task
    while isinstance(node, workflow.MeetingPipelineTask):
        seen.append(type(node).__name__)
        for key, value in params.items():
            actual = getattr(node, key)
            if key == "reviewer_emails":
                actual = list(actual)
            assert actual == value, f"{type(node).__name__}.{key} 被改写为 {actual!r}"
        node = node.requires()
    assert seen == [
        "SendReviewEmailTask",
        "GenerateSummaryTask",
        "TranscribeAudioTask",
        "CleanAudioTask",
    ]


def test_rerun_is_idempotent_and_does_not_resend_email(tmp_path, fake_services):
    params = make_params(tmp_path)
    assert workflow.run_workflow(**params)
    assert workflow.run_workflow(**params)
    assert len(fake_services) == 1, "重跑不应重复发送复审邮件"


def test_cli_entry_matches_run_workflow(tmp_path, fake_services):
    params = make_params(tmp_path)
    argv = [
        "--meeting-id", params["meeting_id"],
        "--meeting-title", params["meeting_title"],
        "--meeting-date", params["meeting_date"],
        "--input-audio-path", params["input_audio_path"],
        "--output-dir", params["output_dir"],
        "--ethics-committee-email", params["ethics_committee_email"],
        "--reviewer-emails", ",".join(params["reviewer_emails"]),
    ]
    assert workflow.main(argv) == 0
    assert len(fake_services) == 1
    mail = fake_services[0]
    assert params["meeting_title"] in mail["subject"]
    assert mail["to"] == [params["ethics_committee_email"]]
    assert mail["cc"] == params["reviewer_emails"]


def test_missing_upstream_output_raises_explicit_error(tmp_path, fake_services):
    params = make_params(tmp_path)

    transcribe = workflow.TranscribeAudioTask(**params)
    with pytest.raises(RuntimeError, match="CleanAudioTask"):
        transcribe.run()

    summarize = workflow.GenerateSummaryTask(**params)
    with pytest.raises(RuntimeError, match="TranscribeAudioTask"):
        summarize.run()

    send_email = workflow.SendReviewEmailTask(**params)
    with pytest.raises(RuntimeError, match="GenerateSummaryTask"):
        send_email.run()


def test_missing_input_audio_raises(tmp_path, fake_services):
    params = make_params(tmp_path)
    params["input_audio_path"] = str(tmp_path / "does_not_exist.wav")
    task = workflow.CleanAudioTask(**params)
    with pytest.raises(FileNotFoundError, match="input_audio_path"):
        task.run()

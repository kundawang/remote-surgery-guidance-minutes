from __future__ import annotations

import json
import os
import struct
import wave
from email.message import EmailMessage
from typing import Dict, List

import luigi

from .config import DEFAULT_CONFIG, MeetingConfig

NOISE_GATE_THRESHOLD = 200  # 16-bit PCM 振幅门限，低于该值视为底噪清零


def _require_upstream(task: luigi.Task) -> str:
    """返回上游任务的产物路径；缺失时报错并说明缺哪一步的输出。"""
    upstream = task.requires()
    target = upstream.output()
    if not target.exists():
        raise RuntimeError(
            f"缺少上游产物 {target.path}（应由 {type(upstream).__name__} 生成），"
            f"请先运行该步骤"
        )
    return target.path


def clean_audio(input_path: str, output_path: str) -> Dict[str, object]:
    """读取 WAV，施加底噪门限后写出清洗后的 WAV，返回音频元信息。"""
    if not os.path.exists(input_path):
        raise RuntimeError(
            f"ProcessAudioTask 的输入音频不存在: {input_path}"
        )
    try:
        reader = wave.open(input_path, "rb")
    except wave.Error as exc:
        raise RuntimeError(
            f"ProcessAudioTask 的输入不是有效的 WAV 音频: {input_path} ({exc})"
        ) from exc
    with reader:
        channels = reader.getnchannels()
        sampwidth = reader.getsampwidth()
        framerate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
        nframes = reader.getnframes()
    if sampwidth != 2:
        raise RuntimeError(
            f"ProcessAudioTask 仅支持 16-bit PCM WAV，{input_path} 的采样宽度为 {sampwidth}"
        )
    count = len(frames) // 2
    samples = list(struct.unpack("<%dh" % count, frames))
    cleaned = [0 if -NOISE_GATE_THRESHOLD < s < NOISE_GATE_THRESHOLD else s for s in samples]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with wave.open(output_path, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sampwidth)
        writer.setframerate(framerate)
        writer.writeframes(struct.pack("<%dh" % count, *cleaned))
    return {
        "sample_rate": framerate,
        "channels": channels,
        "frames": nframes,
        "duration_seconds": round(nframes / float(framerate), 3) if framerate else 0.0,
    }


def transcribe_audio(audio_path: str) -> List[Dict[str, object]]:
    """转写清洗后的音频。优先使用 openai-whisper，不可用时退回确定性占位实现。"""
    with open(audio_path, "rb") as fh:
        header = fh.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise RuntimeError(
            f"TranscribeAudioTask 期望 ProcessAudioTask 生成的 WAV 音频，"
            f"但 {audio_path} 不是有效的音频文件（内容可能不是音频）"
        )
    try:
        import whisper  # type: ignore
    except ImportError:
        whisper = None
    if whisper is not None:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result.get("segments", [])
        ]
    with wave.open(audio_path, "rb") as reader:
        duration = reader.getnframes() / float(reader.getframerate() or 1)
    return [
        {
            "start": 0.0,
            "end": round(duration, 3),
            "text": "[占位转写] openai-whisper 未安装，未执行真实语音识别",
        }
    ]


class EmailSender:
    """邮件发送接口；默认实现把邮件落盘为 .eml（同时作为发送归档）。"""

    def send(self, message: EmailMessage, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(bytes(message))


_default_sender: EmailSender = EmailSender()


def get_email_sender() -> EmailSender:
    return _default_sender


class MeetingTask(luigi.Task):
    """所有流水线任务的基类：统一声明会议参数并提供透传辅助。"""

    meeting_id = luigi.Parameter(default=DEFAULT_CONFIG.meeting_id)
    meeting_title = luigi.Parameter(default=DEFAULT_CONFIG.meeting_title)
    meeting_date = luigi.Parameter(default=DEFAULT_CONFIG.meeting_date)
    input_audio_path = luigi.Parameter(default=DEFAULT_CONFIG.input_audio_path)
    output_dir = luigi.Parameter(default=DEFAULT_CONFIG.output_dir)
    ethics_committee_email = luigi.Parameter(default=DEFAULT_CONFIG.ethics_committee_email)
    reviewer_emails = luigi.ListParameter(default=list(DEFAULT_CONFIG.reviewer_emails))

    def task_params(self) -> Dict[str, object]:
        return {
            "meeting_id": self.meeting_id,
            "meeting_title": self.meeting_title,
            "meeting_date": self.meeting_date,
            "input_audio_path": self.input_audio_path,
            "output_dir": self.output_dir,
            "ethics_committee_email": self.ethics_committee_email,
            "reviewer_emails": list(self.reviewer_emails),
        }

    def meeting_config(self) -> MeetingConfig:
        return MeetingConfig(**self.task_params())

    def meeting_dir(self) -> str:
        return os.path.join(self.output_dir, self.meeting_id)

    def artifact_path(self, stage: str, filename: str) -> str:
        return os.path.join(self.meeting_dir(), stage, filename)


class ProcessAudioTask(MeetingTask):
    """清洗原始音频：产物为 WAV 音频文件，处理状态写入独立的 JSON。"""

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            self.artifact_path("audio", f"{self.meeting_id}_cleaned_audio.wav")
        )

    def status_output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            self.artifact_path("audio", f"{self.meeting_id}_audio_status.json")
        )

    def run(self) -> None:
        cleaned_path = self.output().path
        meta = clean_audio(self.input_audio_path, cleaned_path)
        status = {
            "meeting_id": self.meeting_id,
            "stage": "process_audio",
            "status": "ok",
            "source_audio_path": self.input_audio_path,
            "cleaned_audio_path": cleaned_path,
            "noise_gate_threshold": NOISE_GATE_THRESHOLD,
            **meta,
        }
        status_path = self.status_output().path
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, ensure_ascii=False, indent=2)


class TranscribeAudioTask(MeetingTask):
    """转写清洗后的音频为文字稿。"""

    def requires(self) -> luigi.Task:
        return ProcessAudioTask(**self.task_params())

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            self.artifact_path("transcript", f"{self.meeting_id}_transcript.json")
        )

    def run(self) -> None:
        audio_path = _require_upstream(self)
        segments = transcribe_audio(audio_path)
        payload = {
            "meeting_id": self.meeting_id,
            "meeting_title": self.meeting_title,
            "meeting_date": self.meeting_date,
            "audio_path": audio_path,
            "segments": segments,
        }
        with self.output().open("w") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2))


class GenerateSummaryTask(MeetingTask):
    """根据文字稿生成会议纪要摘要。"""

    def requires(self) -> luigi.Task:
        return TranscribeAudioTask(**self.task_params())

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            self.artifact_path("summary", f"{self.meeting_id}_summary.md")
        )

    def run(self) -> None:
        transcript_path = _require_upstream(self)
        with open(transcript_path, "r", encoding="utf-8") as fh:
            transcript = json.load(fh)
        lines = [
            f"# {self.meeting_title} 会议纪要",
            "",
            f"- 会议编号: {self.meeting_id}",
            f"- 会议日期: {self.meeting_date}",
            f"- 伦理委员会邮箱: {self.ethics_committee_email}",
            "",
            "## 转写要点",
        ]
        for seg in transcript.get("segments", []):
            lines.append(f"- [{seg['start']}s - {seg['end']}s] {seg['text']}")
        with self.output().open("w") as fh:
            fh.write("\n".join(lines) + "\n")


class SendReviewEmailTask(MeetingTask):
    """把摘要发送给伦理委员会与复审专家，邮件落盘为 .eml 归档。"""

    def requires(self) -> luigi.Task:
        return GenerateSummaryTask(**self.task_params())

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            self.artifact_path("email", f"{self.meeting_id}_review_email.eml")
        )

    def run(self) -> None:
        summary_path = _require_upstream(self)
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = fh.read()
        message = EmailMessage()
        message["Subject"] = f"[伦理复审] {self.meeting_title} 会议纪要待审"
        message["From"] = self.ethics_committee_email
        message["To"] = ", ".join(self.reviewer_emails)
        message.set_content(
            f"会议: {self.meeting_title}\n"
            f"日期: {self.meeting_date}\n"
            f"编号: {self.meeting_id}\n\n"
            f"请各位专家复审以下纪要：\n\n{summary}",
            charset="utf-8",
        )
        get_email_sender().send(message, self.output().path)


class ArchiveMinutesTask(MeetingTask):
    """归档整场会议的纪要产物（流水线根任务）。"""

    def requires(self) -> luigi.Task:
        return SendReviewEmailTask(**self.task_params())

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            self.artifact_path("archive", f"{self.meeting_id}_minutes_archive.json")
        )

    def run(self) -> None:
        email_path = _require_upstream(self)
        summary_task = GenerateSummaryTask(**self.task_params())
        transcript_task = TranscribeAudioTask(**self.task_params())
        audio_task = ProcessAudioTask(**self.task_params())
        for task in (summary_task, transcript_task, audio_task):
            if not task.output().exists():
                raise RuntimeError(
                    f"缺少上游产物 {task.output().path}（应由 {type(task).__name__} 生成），"
                    f"请先运行该步骤"
                )
        archive = {
            "meeting_id": self.meeting_id,
            "meeting_title": self.meeting_title,
            "meeting_date": self.meeting_date,
            "ethics_committee_email": self.ethics_committee_email,
            "reviewer_emails": list(self.reviewer_emails),
            "artifacts": {
                "cleaned_audio": audio_task.output().path,
                "audio_status": audio_task.status_output().path,
                "transcript": transcript_task.output().path,
                "summary": summary_task.output().path,
                "review_email": email_path,
            },
        }
        with self.output().open("w") as fh:
            fh.write(json.dumps(archive, ensure_ascii=False, indent=2))

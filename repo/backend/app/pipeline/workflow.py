"""基因伦理研讨会纪要 luigi 流水线。

任务链：
    CleanAudioTask -> TranscribeAudioTask -> GenerateSummaryTask -> SendReviewEmailTask

设计约定：
- 每个产物都有独立路径（见 artifact_paths），任何一步都不会覆盖上游产物；
  清洗后的音频是 wav 文件，处理状态写在同目录的独立 json 里。
- 步骤之间通过上游任务的 output() 显式传递路径，不靠字符串 replace 猜扩展名；
  上游产物缺失时立即报错并指明缺的是哪一步的哪个输出。
- 会议参数（meeting_id / meeting_title / meeting_date / input_audio_path /
  output_dir / ethics_committee_email / reviewer_emails）由入口一路透传到底，
  不存在任何会被默认值覆盖的中间配置对象。
- 任务是否执行完全由 output() 目标是否存在决定（luigi 语义），重跑幂等，
  已完成的任务不会重复执行、不会重复发邮件。

入口：
- Python：run_workflow(meeting_id=..., meeting_title=..., ...)
- 命令行：python -m app.pipeline.workflow --meeting-id ... --meeting-title ...
两种入口构造完全相同的任务图，行为一致。
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import luigi

from . import services


def artifact_paths(output_dir: str, meeting_id: str) -> Dict[str, str]:
    """集中定义全部产物路径。

    步骤间传递路径一律从上游任务的 output() 取，不在各处拼字符串。
    """
    base = os.path.join(output_dir, meeting_id)
    return {
        "cleaned_audio": f"{base}_cleaned_audio.wav",
        "clean_status": f"{base}_cleaned_audio_status.json",
        "transcript": f"{base}_transcript.json",
        "summary": f"{base}_summary.json",
        "review_email": f"{base}_review_email.json",
    }


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _require_upstream(task_name: str, artifact_desc: str, path: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(
            f"缺少上游 {task_name} 的输出（{artifact_desc}）：{path}。"
            f"请先成功运行 {task_name}。"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MeetingPipelineTask(luigi.Task):
    """流水线任务基类：统一声明会议参数，保证入口参数一路透传到底。"""

    meeting_id = luigi.Parameter()
    meeting_title = luigi.Parameter()
    meeting_date = luigi.Parameter()
    input_audio_path = luigi.Parameter()
    output_dir = luigi.Parameter()
    ethics_committee_email = luigi.Parameter()
    reviewer_emails = luigi.ListParameter(default=[])

    def meeting_params(self) -> Dict[str, Any]:
        """构造下游任务时原样透传的全部参数。"""
        return {
            "meeting_id": self.meeting_id,
            "meeting_title": self.meeting_title,
            "meeting_date": self.meeting_date,
            "input_audio_path": self.input_audio_path,
            "output_dir": self.output_dir,
            "ethics_committee_email": self.ethics_committee_email,
            "reviewer_emails": list(self.reviewer_emails),
        }

    def meeting_info(self) -> Dict[str, Any]:
        """写入产物 / 邮件的会议信息，全部来自透传参数。"""
        return {
            "meeting_id": self.meeting_id,
            "meeting_title": self.meeting_title,
            "meeting_date": self.meeting_date,
            "ethics_committee_email": self.ethics_committee_email,
            "reviewer_emails": list(self.reviewer_emails),
        }


class CleanAudioTask(MeetingPipelineTask):
    """清洗原始音频。

    产物分两类、互不覆盖：
    - audio：清洗后的 wav 音频文件；
    - status：处理状态 json（同目录独立文件）。
    """

    def output(self) -> Dict[str, luigi.LocalTarget]:
        paths = artifact_paths(self.output_dir, self.meeting_id)
        return {
            "audio": luigi.LocalTarget(paths["cleaned_audio"]),
            "status": luigi.LocalTarget(paths["clean_status"]),
        }

    def run(self) -> None:
        if not os.path.exists(self.input_audio_path):
            raise FileNotFoundError(
                f"输入音频不存在：{self.input_audio_path}（参数 input_audio_path）"
            )
        os.makedirs(self.output_dir, exist_ok=True)
        targets = self.output()

        tmp_audio = f"{targets['audio'].path}.partial"
        status = services.clean_audio(self.input_audio_path, tmp_audio)
        os.replace(tmp_audio, targets["audio"].path)

        _write_json_atomic(
            targets["status"].path,
            {
                "task": "CleanAudioTask",
                "meeting_id": self.meeting_id,
                "input_audio_path": self.input_audio_path,
                "cleaned_audio_path": targets["audio"].path,
                "finished_at": _utc_now(),
                **status,
            },
        )


class TranscribeAudioTask(MeetingPipelineTask):
    """读取 CleanAudioTask 的音频产物做 Whisper 转写。"""

    def requires(self) -> MeetingPipelineTask:
        return CleanAudioTask(**self.meeting_params())

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            artifact_paths(self.output_dir, self.meeting_id)["transcript"]
        )

    def run(self) -> None:
        audio_path = self.input()["audio"].path
        _require_upstream("CleanAudioTask", "清洗后音频", audio_path)
        transcript = services.transcribe_audio(audio_path)
        _write_json_atomic(
            self.output().path,
            {
                "task": "TranscribeAudioTask",
                "meeting_id": self.meeting_id,
                "source_audio_path": audio_path,
                "transcript": transcript,
                "finished_at": _utc_now(),
            },
        )


class GenerateSummaryTask(MeetingPipelineTask):
    """基于转写结果生成会议纪要摘要。"""

    def requires(self) -> MeetingPipelineTask:
        return TranscribeAudioTask(**self.meeting_params())

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            artifact_paths(self.output_dir, self.meeting_id)["summary"]
        )

    def run(self) -> None:
        transcript_path = self.input().path
        _require_upstream("TranscribeAudioTask", "转写结果", transcript_path)
        with open(transcript_path, encoding="utf-8") as f:
            transcript_payload = json.load(f)
        summary = services.generate_summary(
            transcript_payload["transcript"], self.meeting_info()
        )
        _write_json_atomic(
            self.output().path,
            {
                "task": "GenerateSummaryTask",
                "meeting": self.meeting_info(),
                "source_transcript_path": transcript_path,
                "summary": summary,
                "finished_at": _utc_now(),
            },
        )


class SendReviewEmailTask(MeetingPipelineTask):
    """把纪要摘要发送给伦理委员会和复审人，并把发送记录归档为 json。

    output 目标（归档 json）一旦存在，任务即视为完成，
    重跑整个流水线不会重复发邮件。
    """

    def requires(self) -> MeetingPipelineTask:
        return GenerateSummaryTask(**self.meeting_params())

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            artifact_paths(self.output_dir, self.meeting_id)["review_email"]
        )

    def run(self) -> None:
        summary_path = self.input().path
        _require_upstream("GenerateSummaryTask", "纪要摘要", summary_path)
        with open(summary_path, encoding="utf-8") as f:
            summary_payload = json.load(f)

        info = self.meeting_info()
        subject = f"[伦理复审] {self.meeting_title}（{self.meeting_date}）会议纪要"
        html_body = services.render_review_email(info, summary_payload["summary"])
        services.send_email(
            subject=subject,
            html_body=html_body,
            to=[self.ethics_committee_email],
            cc=list(self.reviewer_emails),
        )
        _write_json_atomic(
            self.output().path,
            {
                "task": "SendReviewEmailTask",
                "meeting": info,
                "subject": subject,
                "to": [self.ethics_committee_email],
                "cc": list(self.reviewer_emails),
                "source_summary_path": summary_path,
                "sent_at": _utc_now(),
            },
        )


def run_workflow(
    meeting_id: str,
    meeting_title: str,
    meeting_date: str,
    input_audio_path: str,
    output_dir: str,
    ethics_committee_email: str,
    reviewer_emails,
) -> bool:
    """流水线入口：构造根任务并运行整个任务图，返回是否成功。"""
    root = SendReviewEmailTask(
        meeting_id=meeting_id,
        meeting_title=meeting_title,
        meeting_date=meeting_date,
        input_audio_path=input_audio_path,
        output_dir=output_dir,
        ethics_committee_email=ethics_committee_email,
        reviewer_emails=list(reviewer_emails),
    )
    return luigi.build([root], local_scheduler=True)


def main(argv=None) -> int:
    """命令行入口：与 run_workflow 构造完全相同的任务图。"""
    parser = argparse.ArgumentParser(description="基因伦理研讨会纪要流水线")
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--meeting-title", required=True)
    parser.add_argument("--meeting-date", required=True)
    parser.add_argument("--input-audio-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ethics-committee-email", required=True)
    parser.add_argument(
        "--reviewer-emails",
        default="",
        help="复审人邮箱，多个用逗号分隔",
    )
    args = parser.parse_args(argv)
    reviewers = [e.strip() for e in args.reviewer_emails.split(",") if e.strip()]
    ok = run_workflow(
        meeting_id=args.meeting_id,
        meeting_title=args.meeting_title,
        meeting_date=args.meeting_date,
        input_audio_path=args.input_audio_path,
        output_dir=args.output_dir,
        ethics_committee_email=args.ethics_committee_email,
        reviewer_emails=reviewers,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

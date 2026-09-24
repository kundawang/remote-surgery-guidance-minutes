from __future__ import annotations

import argparse
from typing import List, Optional

import luigi

from .config import DEFAULT_CONFIG, MeetingConfig
from .tasks import ArchiveMinutesTask


def run_workflow(
    meeting_id: Optional[str] = None,
    meeting_title: Optional[str] = None,
    meeting_date: Optional[str] = None,
    input_audio_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    ethics_committee_email: Optional[str] = None,
    reviewer_emails: Optional[List[str]] = None,
) -> bool:
    """以编程方式运行整条流水线；未显式传入的参数才回落到 MeetingConfig 默认值。"""
    config = MeetingConfig(
        meeting_id=meeting_id if meeting_id is not None else DEFAULT_CONFIG.meeting_id,
        meeting_title=meeting_title if meeting_title is not None else DEFAULT_CONFIG.meeting_title,
        meeting_date=meeting_date if meeting_date is not None else DEFAULT_CONFIG.meeting_date,
        input_audio_path=(
            input_audio_path if input_audio_path is not None else DEFAULT_CONFIG.input_audio_path
        ),
        output_dir=output_dir if output_dir is not None else DEFAULT_CONFIG.output_dir,
        ethics_committee_email=(
            ethics_committee_email
            if ethics_committee_email is not None
            else DEFAULT_CONFIG.ethics_committee_email
        ),
        reviewer_emails=(
            list(reviewer_emails)
            if reviewer_emails is not None
            else list(DEFAULT_CONFIG.reviewer_emails)
        ),
    )
    root = ArchiveMinutesTask(**config.as_task_params())
    return luigi.build([root], local_scheduler=True, detailed_summary=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ethics-minutes-pipeline",
        description="基因伦理研讨会纪要 luigi 流水线",
    )
    parser.add_argument("--meeting-id", default=None)
    parser.add_argument("--meeting-title", default=None)
    parser.add_argument("--meeting-date", default=None)
    parser.add_argument("--input-audio-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ethics-committee-email", default=None)
    parser.add_argument(
        "--reviewer-emails",
        default=None,
        help="复审专家邮箱，逗号分隔",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> bool:
    """命令行入口：与 run_workflow 走同一条路径，行为完全一致。"""
    args = build_parser().parse_args(argv)
    reviewer_emails = (
        [item.strip() for item in args.reviewer_emails.split(",") if item.strip()]
        if args.reviewer_emails is not None
        else None
    )
    return run_workflow(
        meeting_id=args.meeting_id,
        meeting_title=args.meeting_title,
        meeting_date=args.meeting_date,
        input_audio_path=args.input_audio_path,
        output_dir=args.output_dir,
        ethics_committee_email=args.ethics_committee_email,
        reviewer_emails=reviewer_emails,
    )


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

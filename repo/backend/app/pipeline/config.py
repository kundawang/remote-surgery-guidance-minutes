from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class MeetingConfig:
    """会议参数集合。

    默认值仅作为 luigi 参数的缺省值使用；任何入口（CLI / run_workflow）
    显式传入的参数都必须一路透传到每个任务，不允许被这里的默认值覆盖。
    """

    meeting_id: str = "MEETING-0000"
    meeting_title: str = "基因伦理研讨会"
    meeting_date: str = "1970-01-01"
    input_audio_path: str = ""
    output_dir: str = "./meeting_output"
    ethics_committee_email: str = "ethics-committee@example.org"
    reviewer_emails: List[str] = field(default_factory=lambda: ["reviewer@example.org"])

    def as_task_params(self) -> Dict[str, object]:
        return {
            "meeting_id": self.meeting_id,
            "meeting_title": self.meeting_title,
            "meeting_date": self.meeting_date,
            "input_audio_path": self.input_audio_path,
            "output_dir": self.output_dir,
            "ethics_committee_email": self.ethics_committee_email,
            "reviewer_emails": list(self.reviewer_emails),
        }


DEFAULT_CONFIG = MeetingConfig()

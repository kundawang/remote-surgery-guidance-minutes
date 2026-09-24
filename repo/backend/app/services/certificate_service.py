"""生态缸造景证书生成与保存。

证书编号统一规则（主流程与简化接口同一套）：

    AQ-YYYYMMDD-XXXXXXXX

- ``AQ``：业务前缀（Aquascape）。
- ``YYYYMMDD``：签发日期段，便于人工检索。
- ``XXXXXXXX``：8 位十六进制区分段，随机生成并在进程内登记，
  保证同一天连续签发多份也不会重复。

同一份证书正文里的编号、保存/返回的文件名都来自同一个编号，
不再出现主流程带会议 id、简化接口只带日期两套格式。
"""

from __future__ import annotations

import re
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

CERTIFICATE_NUMBER_PREFIX = "AQ"
CERTIFICATE_NUMBER_PATTERN = re.compile(
    rf"{CERTIFICATE_NUMBER_PREFIX}-\d{{8}}-[0-9A-F]{{8}}"
)

DEFAULT_CERTIFICATE_DIR = Path("certificates")

_issued_numbers: set[str] = set()
_issued_numbers_lock = threading.Lock()
_save_lock = threading.Lock()


def generate_certificate_number(
    meeting_id: Optional[str] = None,
    issued_at: Optional[datetime] = None,
    _registry: Optional[set] = None,
) -> str:
    """生成唯一证书编号。

    主流程（传 ``meeting_id``）与简化接口（不传）使用同一套规则，
    编号本身不再嵌入会议 id，区分段统一为随机十六进制。
    """
    issued_at = issued_at or datetime.now()
    date_segment = issued_at.strftime("%Y%m%d")
    registry = _registry if _registry is not None else _issued_numbers
    with _issued_numbers_lock:
        while True:
            discriminator = secrets.token_hex(4).upper()
            number = f"{CERTIFICATE_NUMBER_PREFIX}-{date_segment}-{discriminator}"
            if number not in registry:
                registry.add(number)
                return number


def extract_certificate_number(markdown: str) -> Optional[str]:
    """从证书正文中提取证书编号，未找到时返回 None。"""
    match = CERTIFICATE_NUMBER_PATTERN.search(markdown)
    return match.group(0) if match else None


def generate_markdown_certificate(
    recipient_name: str,
    aquascape_title: str,
    meeting_id: Optional[str] = None,
    issued_at: Optional[datetime] = None,
) -> str:
    """生成证书 markdown，正文中"证书编号"字段只出现一处。"""
    issued_at = issued_at or datetime.now()
    number = generate_certificate_number(meeting_id=meeting_id, issued_at=issued_at)
    issue_date = issued_at.strftime("%Y-%m-%d")
    return (
        "# 生态缸造景证书\n"
        "\n"
        "| 字段 | 内容 |\n"
        "| --- | --- |\n"
        f"| 证书编号 | {number} |\n"
        f"| 持证人 | {recipient_name} |\n"
        f"| 造景作品 | {aquascape_title} |\n"
        f"| 签发日期 | {issue_date} |\n"
    )


def save_certificate(
    markdown: str,
    directory: Path | str = DEFAULT_CERTIFICATE_DIR,
    filename: Optional[str] = None,
) -> Path:
    """保存证书 markdown，返回实际写入的路径。

    - 显式传入 ``filename``：按给定文件名写入，允许覆盖（保留原有行为）。
    - 未传 ``filename``：用正文里的证书编号作为文件名；若目标已存在
      （例如同一秒内同内容保存两次），自动追加 ``-1``、``-2`` 等后缀，
      绝不覆盖已有文件。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        path = directory / filename
        path.write_text(markdown, encoding="utf-8")
        return path

    number = extract_certificate_number(markdown)
    if number is None:
        raise ValueError("证书正文中未找到证书编号，无法生成默认文件名")

    with _save_lock:
        path = directory / f"{number}.md"
        suffix = 1
        while path.exists():
            path = directory / f"{number}-{suffix}.md"
            suffix += 1
        path.write_text(markdown, encoding="utf-8")
    return path

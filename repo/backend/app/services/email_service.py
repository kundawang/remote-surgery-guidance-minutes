"""Outbound email service.

Provides ``send_email`` for delivering notification/archive emails with
optional markdown content and file attachments.

Attachment handling notes:

* ``Content-Disposition`` filenames follow RFC 2231: non-ASCII filenames
  (e.g. Chinese) are parameter-encoded so mail clients do not mangle them.
* Textual attachment content is UTF-8 encoded and the part carries an
  explicit ``charset=utf-8`` so non-ASCII text survives transport.
"""

import logging
import mimetypes
import os
import smtplib
from dataclasses import dataclass
from email import encoders
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# application/* subtypes whose payload is textual and must be UTF-8 encoded.
_TEXTUAL_APPLICATION_SUBTYPES = {
    "json",
    "xml",
    "javascript",
    "csv",
    "yaml",
    "x-yaml",
    "x-markdown",
    "markdown",
}


@dataclass
class SmtpConfig:
    host: str
    port: int
    username: Optional[str]
    password: Optional[str]
    from_email: str

    @property
    def configured(self) -> bool:
        return bool(self.host and self.username and self.password)


def _load_smtp_config() -> SmtpConfig:
    """Load SMTP settings from app config, falling back to env vars."""
    try:
        from ..core.config import settings

        return SmtpConfig(
            host=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USERNAME,
            password=settings.SMTP_PASSWORD,
            from_email=settings.SMTP_FROM_EMAIL,
        )
    except Exception:  # pragma: no cover - config backend unavailable
        return SmtpConfig(
            host=os.environ.get("SMTP_HOST", ""),
            port=int(os.environ.get("SMTP_PORT", "587")),
            username=os.environ.get("SMTP_USERNAME"),
            password=os.environ.get("SMTP_PASSWORD"),
            from_email=os.environ.get("SMTP_FROM_EMAIL", "no-reply@localhost"),
        )


def _is_textual(maintype: str, subtype: str) -> bool:
    if maintype == "text":
        return True
    return maintype == "application" and subtype in _TEXTUAL_APPLICATION_SUBTYPES


def _guess_content_type(filename: str, content: Union[str, bytes]) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        maintype, subtype = guessed.split("/", 1)
        if _is_textual(maintype, subtype) or isinstance(content, bytes):
            return guessed
    if isinstance(content, str):
        return "text/plain"
    return "application/octet-stream"


def _build_attachment(attachment: Dict[str, Any]) -> MIMEBase:
    filename = attachment["filename"]
    content = attachment.get("content", b"")
    content_type = attachment.get("content_type") or _guess_content_type(
        filename, content
    )
    maintype, subtype = content_type.split("/", 1)

    part = MIMEBase(maintype, subtype)
    if isinstance(content, str):
        content = content.encode("utf-8")
    if _is_textual(maintype, subtype):
        part.set_param("charset", "utf-8", header="Content-Type")
    part.set_payload(content)
    encoders.encode_base64(part)
    # add_header() emits RFC 2231 encoding (filename*=utf-8''...) for
    # non-ASCII filenames and a quoted string for ASCII ones.
    part.add_header("Content-Disposition", "attachment", filename=filename)
    return part


def build_email_message(
    to: Union[str, List[str]],
    subject: str,
    body: str,
    markdown_content: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    from_email: str = "no-reply@localhost",
) -> MIMEMultipart:
    recipients = [to] if isinstance(to, str) else list(to)

    msg = MIMEMultipart("mixed")
    msg["From"] = from_email
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = Header(subject, "utf-8")

    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(body, "plain", "utf-8"))
    if markdown_content:
        alternative.attach(MIMEText(markdown_content, "markdown", "utf-8"))
    msg.attach(alternative)

    for attachment in attachments or []:
        msg.attach(_build_attachment(attachment))

    return msg


def send_email(
    to: Union[str, List[str]],
    subject: str,
    body: str,
    markdown_content: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Send an email. Returns True on success, False otherwise.

    Failure modes are distinguished in the logs: a WARNING is emitted when
    SMTP is not configured, and an ERROR (with traceback) when the send
    itself fails.
    """
    recipients = [to] if isinstance(to, str) else list(to)
    config = _load_smtp_config()

    if not config.configured:
        logger.warning(
            "email to %s (subject=%r) not sent: SMTP is not configured "
            "(missing host/username/password)",
            recipients,
            subject,
        )
        return False

    msg = build_email_message(
        to=recipients,
        subject=subject,
        body=body,
        markdown_content=markdown_content,
        attachments=attachments,
        from_email=config.from_email,
    )

    try:
        with smtplib.SMTP(config.host, config.port, timeout=30) as server:
            server.starttls()
            server.login(config.username, config.password)
            server.send_message(
                msg, from_addr=config.from_email, to_addrs=recipients
            )
    except Exception:
        logger.exception(
            "failed to send email to %s (subject=%r) via %s:%s",
            recipients,
            subject,
            config.host,
            config.port,
        )
        return False

    logger.info("sent email to %s (subject=%r)", recipients, subject)
    return True

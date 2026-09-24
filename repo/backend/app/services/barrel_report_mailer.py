import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from ..core.config import settings
from ..models.schemas import BarrelTastingReportResponse


class BarrelReportMailer:
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_FROM_EMAIL
        self.default_recipient = settings.EMAIL_ARCHIVE_TO

    def send_report(
        self,
        report: BarrelTastingReportResponse,
        recipient_email: Optional[str] = None,
    ) -> dict:
        to_email = recipient_email or self.default_recipient
        subject = report.title

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_email
            msg["To"] = to_email
            msg["Subject"] = subject
            msg["X-Report-Vintage"] = str(report.vintage)
            msg["X-Report-Source"] = report.source

            msg.attach(MIMEText(report.markdown, "plain", "utf-8"))
            msg.attach(MIMEText(self._render_html(report), "html", "utf-8"))

            message_id = self._send_email(msg, to_email)

            return {
                "success": True,
                "message_id": message_id,
                "recipient": to_email,
                "subject": subject,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recipient": to_email,
                "subject": subject,
            }

    def _render_html(self, report: BarrelTastingReportResponse) -> str:
        rows = "".join(
            f"<tr><td>{c.variety}</td>"
            f"<td>{c.ratio_percent if c.ratio_percent is not None else '待确认'}</td>"
            f"<td>{c.volume_liters if c.volume_liters is not None else '-'}</td></tr>"
            for c in report.blend_components
        )
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>{report.title}</title></head>
<body>
    <h1>{report.title}</h1>
    <p>年份: {report.vintage} | 数据来源: {report.source} | 生成时间: {report.generated_at}</p>
    <table border="1" cellspacing="0" cellpadding="6">
        <thead><tr><th>品种</th><th>比例(%)</th><th>体积(L)</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <pre>{report.markdown}</pre>
</body>
</html>"""

    def _send_email(self, msg: MIMEMultipart, to_email: str) -> str:
        if not self.smtp_username or not self.smtp_password:
            print(f"[MOCK EMAIL] Would send to {to_email}: {msg['Subject']}")
            return f"mock-{datetime.now().timestamp()}"

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)

        return msg.get("Message-ID", str(datetime.now().timestamp()))

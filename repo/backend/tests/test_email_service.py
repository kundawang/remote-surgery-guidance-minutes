import smtplib
import unittest
from unittest import mock

from app.services import email_service
from app.services.email_service import SmtpConfig, build_email_message, send_email


def _only_attachment(msg):
    attachments = [
        part
        for part in msg.walk()
        if part.get_content_disposition() == "attachment"
    ]
    assert len(attachments) == 1
    return attachments[0]


class AttachmentFilenameTests(unittest.TestCase):
    def test_ascii_filename_has_no_space_after_equals(self):
        msg = build_email_message(
            to="curator@example.com",
            subject="s",
            body="b",
            attachments=[{"filename": "plan.md", "content": "hello"}],
        )
        part = _only_attachment(msg)
        disposition = part["Content-Disposition"]
        self.assertNotIn("filename= ", disposition)
        self.assertEqual(disposition, 'attachment; filename="plan.md"')
        self.assertEqual(part.get_filename(), "plan.md")

    def test_non_ascii_filename_is_rfc2231_encoded(self):
        filename = "策展方案-2026.md"
        msg = build_email_message(
            to="curator@example.com",
            subject="s",
            body="b",
            attachments=[{"filename": filename, "content": "hello"}],
        )
        part = _only_attachment(msg)
        disposition = part["Content-Disposition"]
        self.assertNotIn("filename= ", disposition)
        self.assertIn("filename*=utf-8''", disposition)
        # Raw non-ASCII bytes must not appear in the header.
        self.assertNotIn(filename, disposition)
        disposition.encode("ascii")  # must be ASCII-safe on the wire
        # And it must round-trip back to the original name.
        self.assertEqual(part.get_filename(), filename)


class AttachmentContentTests(unittest.TestCase):
    def test_text_attachment_is_utf8_with_charset(self):
        content = "展览前言：中文内容不能乱码。"
        msg = build_email_message(
            to="curator@example.com",
            subject="s",
            body="b",
            attachments=[
                {
                    "filename": "intro.md",
                    "content": content,
                    "content_type": "text/markdown",
                }
            ],
        )
        part = _only_attachment(msg)
        self.assertEqual(part.get_content_type(), "text/markdown")
        self.assertEqual(part.get_content_charset(), "utf-8")
        self.assertEqual(part["Content-Type"], 'text/markdown; charset="utf-8"')
        self.assertEqual(part.get_payload(decode=True), content.encode("utf-8"))

    def test_text_attachment_defaults_to_text_plain(self):
        msg = build_email_message(
            to="curator@example.com",
            subject="s",
            body="b",
            attachments=[{"filename": "notes.unknownext", "content": "中文"}],
        )
        part = _only_attachment(msg)
        self.assertEqual(part.get_content_type(), "text/plain")
        self.assertEqual(part.get_content_charset(), "utf-8")
        self.assertEqual(part.get_payload(decode=True), "中文".encode("utf-8"))

    def test_binary_attachment_round_trips_bytes(self):
        payload = b"\x89PNG\r\n\x1a\n\x00\xff"
        msg = build_email_message(
            to="curator@example.com",
            subject="s",
            body="b",
            attachments=[
                {
                    "filename": "poster.png",
                    "content": payload,
                    "content_type": "image/png",
                }
            ],
        )
        part = _only_attachment(msg)
        self.assertEqual(part.get_content_type(), "image/png")
        self.assertIsNone(part.get_content_charset())
        self.assertEqual(part.get_payload(decode=True), payload)


class SendEmailOutcomeTests(unittest.TestCase):
    def _configured(self):
        return SmtpConfig(
            host="smtp.example.com",
            port=587,
            username="user",
            password="pass",
            from_email="noreply@example.com",
        )

    def test_not_configured_returns_false_and_logs_warning(self):
        unconfigured = SmtpConfig(
            host="", port=587, username=None, password=None, from_email="x@y.z"
        )
        with mock.patch.object(
            email_service, "_load_smtp_config", return_value=unconfigured
        ), mock.patch.object(email_service.smtplib, "SMTP") as smtp_cls:
            with self.assertLogs(email_service.logger, level="WARNING") as logs:
                result = send_email("a@b.c", "subject", "body")
        self.assertFalse(result)
        self.assertIn("not configured", "".join(logs.output))
        smtp_cls.assert_not_called()

    def test_smtp_failure_returns_false_and_logs_error(self):
        with mock.patch.object(
            email_service, "_load_smtp_config", return_value=self._configured()
        ), mock.patch.object(
            email_service.smtplib, "SMTP", side_effect=smtplib.SMTPException("boom")
        ):
            with self.assertLogs(email_service.logger, level="ERROR") as logs:
                result = send_email("a@b.c", "subject", "body")
        self.assertFalse(result)
        self.assertIn("failed to send email", "".join(logs.output))

    def test_success_returns_true(self):
        smtp_instance = mock.MagicMock()
        smtp_cls = mock.MagicMock()
        smtp_cls.return_value.__enter__.return_value = smtp_instance
        with mock.patch.object(
            email_service, "_load_smtp_config", return_value=self._configured()
        ), mock.patch.object(email_service.smtplib, "SMTP", smtp_cls):
            with self.assertLogs(email_service.logger, level="INFO") as logs:
                result = send_email(
                    ["a@b.c", "d@e.f"],
                    "展览开幕通知",
                    "正文",
                    markdown_content="# 标题\n\n中文内容",
                    attachments=[{"filename": "策展方案.md", "content": "中文附件"}],
                )
        self.assertTrue(result)
        smtp_instance.login.assert_called_once_with("user", "pass")
        smtp_instance.send_message.assert_called_once()
        self.assertIn("sent email", "".join(logs.output))


if __name__ == "__main__":
    unittest.main()

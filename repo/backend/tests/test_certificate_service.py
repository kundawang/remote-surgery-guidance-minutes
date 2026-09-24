"""证书编号唯一性与保存行为的回归测试。"""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.certificate_service import (
    CERTIFICATE_NUMBER_PATTERN,
    extract_certificate_number,
    generate_certificate_number,
    generate_markdown_certificate,
    save_certificate,
)


class CertificateNumberTests(unittest.TestCase):
    def test_same_day_numbers_are_unique(self):
        """同一天连续签发多份，编号不能重复。"""
        issued_at = datetime(2026, 9, 24, 10, 0, 0)
        registry: set = set()
        numbers = {
            generate_certificate_number(issued_at=issued_at, _registry=registry)
            for _ in range(500)
        }
        self.assertEqual(len(numbers), 500)
        for number in numbers:
            self.assertRegex(number, CERTIFICATE_NUMBER_PATTERN)
            self.assertTrue(number.startswith("AQ-20260924-"))

    def test_unified_rule_for_main_flow_and_simple_interface(self):
        """主流程（带会议 id）与简化接口（不带）产出同一套编号规则。"""
        issued_at = datetime(2026, 9, 24, 10, 0, 0)
        with_meeting = generate_certificate_number(
            meeting_id="meeting-123", issued_at=issued_at, _registry=set()
        )
        without_meeting = generate_certificate_number(
            issued_at=issued_at, _registry=set()
        )
        self.assertRegex(with_meeting, CERTIFICATE_NUMBER_PATTERN)
        self.assertRegex(without_meeting, CERTIFICATE_NUMBER_PATTERN)
        self.assertNotEqual(with_meeting, without_meeting)

    def test_markdown_has_single_certificate_number_field(self):
        """markdown 中"证书编号"字段只出现一处，且编号可被提取。"""
        markdown = generate_markdown_certificate("张三", "山水之间")
        self.assertEqual(markdown.count("证书编号"), 1)
        number = extract_certificate_number(markdown)
        self.assertIsNotNone(number)
        self.assertIn(f"| 证书编号 | {number} |", markdown)


class SaveCertificateTests(unittest.TestCase):
    def test_same_content_saved_twice_does_not_overwrite(self):
        """同一份内容在同一秒内保存两次，不互相覆盖。"""
        markdown = generate_markdown_certificate("张三", "山水之间")
        with tempfile.TemporaryDirectory() as tmp:
            first = save_certificate(markdown, directory=tmp)
            second = save_certificate(markdown, directory=tmp)
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(first.read_text(encoding="utf-8"), markdown)
            self.assertEqual(second.read_text(encoding="utf-8"), markdown)

    def test_default_filename_matches_number_in_body(self):
        """默认文件名与正文中的证书编号一致。"""
        markdown = generate_markdown_certificate("李四", "溪谷")
        number = extract_certificate_number(markdown)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_certificate(markdown, directory=tmp)
            self.assertEqual(path.name, f"{number}.md")

    def test_explicit_filename_may_overwrite(self):
        """显式给定 filename 时保留覆盖行为。"""
        first_md = generate_markdown_certificate("张三", "作品一")
        second_md = generate_markdown_certificate("李四", "作品二")
        with tempfile.TemporaryDirectory() as tmp:
            save_certificate(first_md, directory=tmp, filename="cert.md")
            path = save_certificate(second_md, directory=tmp, filename="cert.md")
            self.assertEqual(path.name, "cert.md")
            self.assertEqual(path.read_text(encoding="utf-8"), second_md)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()

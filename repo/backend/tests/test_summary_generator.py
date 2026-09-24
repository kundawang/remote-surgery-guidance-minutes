from app.services.summary_generator import SummaryGenerator
from app.models.schemas import TranscriptSegment


def make_segment(speaker, role, text, start=0.0, end=5.0, confidence=0.9):
    return TranscriptSegment(
        speaker=speaker,
        speaker_role=role,
        start_time=start,
        end_time=end,
        text=text,
        confidence=confidence,
    )


def make_generator():
    return SummaryGenerator()


class TestParticipantCountFromData:
    def test_count_comes_from_transcript_speakers(self):
        gen = make_generator()
        transcripts = [
            make_segment("张医生", "主刀医生", "开始分离组织"),
            make_segment("李专家", "远程专家", "注意止血"),
            make_segment("王护士", "护士", "器械已备好"),
        ]
        assessment = gen._generate_overall_assessment(transcripts, {})
        assert "共计3人参与讨论" in assessment
        assert "8人" not in assessment

    def test_count_comes_from_session_participants(self):
        gen = make_generator()
        transcripts = [make_segment("张医生", "主刀医生", "开始分离组织")]
        session_info = {"participants": ["张医生", "李专家", "王护士", "赵麻醉", "刘巡回"]}
        assessment = gen._generate_overall_assessment(transcripts, session_info)
        assert "共计5人参与讨论" in assessment

    def test_no_data_means_unrecorded(self):
        gen = make_generator()
        assessment = gen._generate_overall_assessment([], {})
        assert "未记录" in assessment
        assert "共计8人" not in assessment

    def test_conclusion_matches_data_quality(self):
        gen = make_generator()
        risky = [
            make_segment("李专家", "远程专家", "小心出血风险", confidence=0.5),
            make_segment("张医生", "主刀医生", "注意避免损伤", confidence=0.6),
        ]
        assessment = gen._generate_overall_assessment(risky, {})
        assert "手术过程顺利" not in assessment
        assert "风险点" in assessment
        assert "人工复核" in assessment

    def test_fallback_marks_source(self):
        gen = make_generator()
        result = gen.generate_summary("s1", [make_segment("张医生", "主刀医生", "开始手术")], {})
        assert result.source == "fallback"


class TestSectionSplitOnlyOnHeadings:
    MARKDOWN = """### 总体评估
手术整体顺利。

### 技术改进
根据总体评估可知，缝合张力需要降低。
- 建议使用更细的缝线

### 并发症
- 出血风险
"""

    def test_body_keyword_does_not_switch_section(self):
        gen = make_generator()
        result = gen._parse_model_output(self.MARKDOWN)
        assert any("根据总体评估可知" in item for item in result["technical_improvements"])
        assert "根据总体评估可知" not in result["overall_assessment"]

    def test_following_section_not_swallowed(self):
        gen = make_generator()
        result = gen._parse_model_output(self.MARKDOWN)
        assert result["complications"] == ["出血风险"]
        assert "出血风险" not in result["overall_assessment"]
        assert "出血风险" not in " ".join(result["technical_improvements"])

    def test_plain_heading_line_in_body_ignored(self):
        gen = make_generator()
        text = "### 总体评估\n恢复良好。\n并发症：无。\n"
        result = gen._parse_model_output(text)
        assert result["complications"] == []
        assert "并发症：无。" in result["overall_assessment"]


class TestMissingSections:
    def test_missing_section_gives_empty_field(self):
        gen = make_generator()
        text = "### 总体评估\n手术顺利。\n\n### 技术改进\n- 建议一\n"
        result = gen._parse_model_output(text)
        assert result["overall_assessment"] == "手术顺利。"
        assert result["technical_improvements"] == ["建议一"]
        assert result["complications"] == []
        assert result["key_points"] == []
        assert result["surgical_steps"] == []
        assert result["anatomical_landmarks"] == []

    def test_missing_section_not_stuffed_into_summary(self):
        gen = make_generator()
        text = "### 总体评估\n手术顺利。\n\n### 并发症\n- 出血\n"
        result = gen._parse_model_output(text)
        assert "出血" not in result["overall_assessment"]
        assert result["complications"] == ["出血"]

    def test_unparseable_text_goes_to_summary(self):
        gen = make_generator()
        text = "这是一段没有任何标题的模型输出，包含手术要点四个字。"
        result = gen._parse_model_output(text)
        assert result["overall_assessment"] == text
        assert result["key_points"] == []
        assert result["complications"] == []

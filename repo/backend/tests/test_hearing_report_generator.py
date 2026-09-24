import json
from unittest.mock import MagicMock

from app.models.schemas import TranscriptSegment
from app.services.hearing_report_generator import (
    EMPTY_MATRIX_HINT,
    FALLBACK_NOTICE,
    HearingReportGenerator,
)

LOCATIONS = ["滨江大道", "解放路"]
SOLUTIONS = ["优化滨江大道早高峰信号配时", "在解放路增设潮汐车道"]
UNRELATED_LOCATIONS = ["人民广场", "南京东路", "陆家嘴", "淮海中路"]
HARDCODED_MEASURES = ["公交专用道", "尾号限行", "错峰上下班"]

HEARING_INFO = {"会议主题": "中心城区交通拥堵治理听证会", "主持人": "市交通委"}


def make_transcripts():
    return [
        TranscriptSegment(
            speaker="张三", speaker_role="市民代表",
            start_time=65.0, end_time=80.0,
            text="滨江大道早高峰拥堵严重，建议优化信号配时。", confidence=0.95,
        ),
        TranscriptSegment(
            speaker="李四", speaker_role="市民代表",
            start_time=95.0, end_time=110.0,
            text="解放路接送学车辆占道，建议增设潮汐车道。", confidence=0.93,
        ),
        TranscriptSegment(
            speaker="王五", speaker_role="交管部门",
            start_time=130.0, end_time=145.0,
            text="我们已记录两条建议，会后会同相关部门评估。", confidence=0.9,
        ),
    ]


def make_fallback_generator():
    generator = HearingReportGenerator()
    generator.client = None
    return generator


def generate_fallback(locations=LOCATIONS, solutions=SOLUTIONS, transcripts=None):
    generator = make_fallback_generator()
    return generator.generate_report(
        hearing_info=HEARING_INFO,
        locations=locations,
        solutions=solutions,
        transcripts=make_transcripts() if transcripts is None else transcripts,
    )


def test_fallback_contains_only_input_locations():
    report = generate_fallback()

    for location in LOCATIONS:
        assert location in report.report_markdown
    for unrelated in UNRELATED_LOCATIONS:
        assert unrelated not in report.report_markdown
        assert unrelated not in report.cause_analysis
        assert unrelated not in report.minutes


def test_fallback_is_explicitly_marked_and_distinguishable():
    report = generate_fallback()

    assert report.is_fallback is True
    assert report.fallback_notice == FALLBACK_NOTICE
    assert FALLBACK_NOTICE in report.report_markdown
    assert FALLBACK_NOTICE in report.minutes
    assert FALLBACK_NOTICE in report.cause_analysis


def test_fallback_action_matrix_rows_come_only_from_solutions():
    report = generate_fallback()

    assert len(report.action_matrix) == len(SOLUTIONS)
    assert [row["建议措施"] for row in report.action_matrix] == SOLUTIONS
    for hardcoded in HARDCODED_MEASURES:
        assert hardcoded not in report.report_markdown


def test_fallback_action_matrix_empty_when_no_solutions():
    report = generate_fallback(solutions=[])

    assert report.action_matrix == []
    assert EMPTY_MATRIX_HINT in report.report_markdown
    matrix_section = report.report_markdown.split("## 三、行动矩阵")[1]
    matrix_section = matrix_section.split("## 四、完整发言记录")[0]
    data_rows = [
        line for line in matrix_section.splitlines()
        if line.startswith("|") and "序号" not in line and "---" not in line
    ]
    assert data_rows == []
    for hardcoded in HARDCODED_MEASURES:
        assert hardcoded not in report.report_markdown


def test_report_structure_and_timestamp_format():
    report = generate_fallback()
    markdown = report.report_markdown

    assert markdown.index("## 一、会议纪要") < markdown.index("## 二、成因分析")
    assert markdown.index("## 二、成因分析") < markdown.index("## 三、行动矩阵")
    assert markdown.index("## 三、行动矩阵") < markdown.index("## 四、完整发言记录")
    assert "[01:05] 张三(市民代表): 滨江大道早高峰拥堵严重，建议优化信号配时。" in markdown
    assert "[02:10] 王五(交管部门)" in markdown


def test_attendees_grouped_by_role():
    report = generate_fallback()

    assert "**参会人员**（按角色分组）:" in report.minutes
    assert "- 市民代表: 张三、李四" in report.minutes
    assert "- 交管部门: 王五" in report.minutes


def make_openai_generator(payload, side_effect=None):
    generator = HearingReportGenerator()
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.create.side_effect = side_effect
    else:
        message = MagicMock()
        message.content = json.dumps(payload, ensure_ascii=False)
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=message)]
        )
    generator.client = client
    return generator


def test_openai_path_structure_preserved():
    payload = {
        "minutes": "会议围绕滨江大道、解放路拥堵问题展开。",
        "cause_analysis": "滨江大道通勤车流集中。",
        "action_matrix": [
            {"建议措施": SOLUTIONS[0], "责任部门": "交管局", "时限": "三季度", "状态": "待评估"}
        ],
    }
    generator = make_openai_generator(payload)
    report = generator.generate_report(
        hearing_info=HEARING_INFO,
        locations=LOCATIONS,
        solutions=SOLUTIONS,
        transcripts=make_transcripts(),
    )

    assert report.is_fallback is False
    assert report.fallback_notice is None
    assert FALLBACK_NOTICE not in report.report_markdown
    assert report.minutes.startswith("会议围绕滨江大道、解放路拥堵问题展开。")
    assert "**参会人员**（按角色分组）:" in report.minutes
    assert report.cause_analysis == "滨江大道通勤车流集中。"
    assert report.action_matrix[0]["建议措施"] == SOLUTIONS[0]
    assert report.action_matrix[0]["序号"] == 1
    markdown = report.report_markdown
    assert markdown.index("## 一、会议纪要") < markdown.index("## 二、成因分析")
    assert markdown.index("## 二、成因分析") < markdown.index("## 三、行动矩阵")
    assert markdown.index("## 三、行动矩阵") < markdown.index("## 四、完整发言记录")
    assert "[01:05] 张三(市民代表)" in markdown


def test_openai_failure_falls_back_without_fabrication():
    generator = make_openai_generator(payload=None, side_effect=RuntimeError("api down"))
    report = generator.generate_report(
        hearing_info=HEARING_INFO,
        locations=LOCATIONS,
        solutions=SOLUTIONS,
        transcripts=make_transcripts(),
    )

    assert report.is_fallback is True
    assert FALLBACK_NOTICE in report.report_markdown
    for unrelated in UNRELATED_LOCATIONS:
        assert unrelated not in report.report_markdown
    assert [row["建议措施"] for row in report.action_matrix] == SOLUTIONS

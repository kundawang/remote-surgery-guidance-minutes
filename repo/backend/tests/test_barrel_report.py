import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import (  # noqa: E402
    BarrelParcelInput,
    BarrelTastingRequest,
)
from app.services.barrel_report_generator import (  # noqa: E402
    BLEND_PENDING_NOTE,
    BLEND_STATUS_CALCULATED,
    BLEND_STATUS_PENDING,
    SOURCE_FALLBACK,
    SOURCE_MODEL,
    BarrelReportGenerator,
)

VINTAGE = 2019


def make_parcels():
    return [
        BarrelParcelInput(
            name="东坡地块",
            variety="赤霞珠",
            volume_liters=600.0,
            lab_metrics={"残糖": 2.1, "酒精度": 13.5},
        ),
        BarrelParcelInput(
            name="河谷地块",
            variety="梅洛",
            volume_liters=250.0,
            lab_metrics={"残糖": 2.4},
        ),
        BarrelParcelInput(
            name="山坡地块",
            variety="品丽珠",
            volume_liters=100.0,
        ),
        BarrelParcelInput(
            name="老藤地块",
            variety="小维多",
            volume_liters=50.0,
        ),
    ]


def make_request(**overrides):
    data = {
        "vintage": VINTAGE,
        "wine_name": "珍藏干红",
        "winemaker": "张三",
        "parcels": make_parcels(),
        "tasting_notes": ["东坡地块单宁紧实，黑色水果香气突出。"],
        "keywords": ["单宁", "黑莓", "橡木桶"],
    }
    data.update(overrides)
    return BarrelTastingRequest(**data)


def make_generator():
    generator = BarrelReportGenerator()
    generator.client = None
    return generator


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps(self.payload, ensure_ascii=False)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_model_generator():
    generator = make_generator()
    completions = FakeCompletions(
        {
            "tasting_summary": "模型生成的品评总结。",
            "blend_rationale": "模型生成的调配解读。",
            "overall": "模型生成的总结。",
        }
    )
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return generator, completions


def markdown_years(markdown):
    return set(re.findall(r"(?:19|20)\d{2}", markdown))


class TestVintageFollowsRequest:
    """年份跟随入参：标题与正文只允许出现请求传入的 vintage。"""

    def test_fallback_title_and_body_use_request_vintage(self):
        report = make_generator().generate_report(make_request())

        assert report.vintage == VINTAGE
        assert report.title == f"{VINTAGE}年份珍藏干红桶边品评报告"
        assert f"年份: {VINTAGE}" in report.markdown
        assert markdown_years(report.markdown) == {str(VINTAGE)}

    @pytest.mark.parametrize("vintage", [2019, 2021, 2024])
    def test_vintage_not_hardcoded(self, vintage):
        report = make_generator().generate_report(make_request(vintage=vintage))

        assert report.title.startswith(f"{vintage}年份")
        assert markdown_years(report.markdown) == {str(vintage)}

    def test_model_path_uses_request_vintage(self):
        generator, completions = make_model_generator()
        report = generator.generate_report(make_request())

        assert report.source == SOURCE_MODEL
        assert report.title.startswith(f"{VINTAGE}年份")
        assert markdown_years(report.markdown) == {str(VINTAGE)}

        prompt = completions.calls[0]["messages"][1]["content"]
        assert f"{VINTAGE}年份" in prompt
        assert "2023" not in prompt


class TestFallbackHasNoFakeBlend:
    """兜底不带假比例：比例来自输入数据或明确占位，且内容可区分来源。"""

    def test_fallback_marked_and_ratios_computed_from_input(self):
        report = make_generator().generate_report(make_request())

        assert report.source == SOURCE_FALLBACK
        assert "本地兜底生成" in report.markdown

        components = report.blend_components
        assert all(c.status == BLEND_STATUS_CALCULATED for c in components)
        assert round(sum(c.ratio_percent for c in components), 1) == 100.0

        ratios = {c.variety: c.ratio_percent for c in components}
        assert ratios == {"赤霞珠": 60.0, "梅洛": 25.0, "品丽珠": 10.0, "小维多": 5.0}

    def test_blend_ratios_sum_to_100_with_rounding_drift(self):
        parcels = [
            BarrelParcelInput(name="甲", variety="品种A", volume_liters=333.0),
            BarrelParcelInput(name="乙", variety="品种B", volume_liters=333.0),
            BarrelParcelInput(name="丙", variety="品种C", volume_liters=334.0),
        ]
        report = make_generator().generate_report(make_request(parcels=parcels))

        total = sum(c.ratio_percent for c in report.blend_components)
        assert round(total, 1) == 100.0

    def test_fallback_without_parcels_uses_pending_placeholder(self):
        report = make_generator().generate_report(make_request(parcels=[]))

        assert report.source == SOURCE_FALLBACK
        assert len(report.blend_components) == 1
        placeholder = report.blend_components[0]
        assert placeholder.status == BLEND_STATUS_PENDING
        assert placeholder.ratio_percent is None
        assert placeholder.note == BLEND_PENDING_NOTE

        assert BLEND_PENDING_NOTE in report.markdown
        blend_section = report.markdown.split("调配建议")[1].split("总结")[0]
        assert "%" not in blend_section

    def test_fallback_contains_only_input_names_and_numbers(self):
        report = make_generator().generate_report(make_request())
        markdown = report.markdown

        for parcel in make_parcels():
            assert parcel.name in markdown
            assert parcel.variety in markdown

        assert "西拉" not in markdown
        assert "未知地块" not in markdown
        assert markdown_years(markdown) == {str(VINTAGE)}


class TestPromptKeywordOrderStable:
    """提示词顺序稳定：同一输入（无论关键词顺序）生成完全一致的提示词。"""

    def test_keyword_order_does_not_change_prompt(self):
        generator = make_generator()
        request_a = make_request(keywords=["单宁", "黑莓", "橡木桶"])
        request_b = make_request(keywords=["橡木桶", "单宁", "黑莓"])

        prompt_a = generator._build_prompt(request_a, generator._compute_blend(request_a.parcels))
        prompt_b = generator._build_prompt(request_b, generator._compute_blend(request_b.parcels))

        assert prompt_a == prompt_b

    def test_repeated_generation_produces_identical_prompt(self):
        generator = make_generator()
        request = make_request()

        first = generator._build_prompt(request, generator._compute_blend(request.parcels))
        for _ in range(5):
            again = generator._build_prompt(request, generator._compute_blend(request.parcels))
            assert again == first

    def test_keywords_sorted_and_deduplicated_in_prompt(self):
        generator = make_generator()
        request = make_request(keywords=["橡木桶", "单宁", "黑莓", "单宁", " 黑莓 "])

        _, user_prompt = generator._build_prompt(
            request, generator._compute_blend(request.parcels)
        )

        keyword_line = next(
            line for line in user_prompt.splitlines() if "黑莓" in line
        )
        assert keyword_line.index("单宁") < keyword_line.index("橡木桶") < keyword_line.index("黑莓")
        assert keyword_line.count("单宁") == 1
        assert keyword_line.count("黑莓") == 1

    def test_parcel_order_does_not_change_prompt(self):
        generator = make_generator()
        parcels = make_parcels()
        request_a = make_request(parcels=parcels)
        request_b = make_request(parcels=list(reversed(parcels)))

        prompt_a = generator._build_prompt(request_a, generator._compute_blend(request_a.parcels))
        prompt_b = generator._build_prompt(request_b, generator._compute_blend(request_b.parcels))

        assert prompt_a == prompt_b

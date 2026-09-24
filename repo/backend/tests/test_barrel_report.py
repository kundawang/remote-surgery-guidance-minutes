import re

import pytest

from app.models.schemas import BarrelParcelInput, BarrelTastingRequest
from app.services.barrel_report_generator import (
    BLEND_PENDING_NOTE,
    BLEND_STATUS_CALCULATED,
    BLEND_STATUS_PENDING,
    SOURCE_FALLBACK,
    BarrelReportGenerator,
)


def make_generator() -> BarrelReportGenerator:
    generator = BarrelReportGenerator()
    generator.client = None
    return generator


def make_request(**overrides) -> BarrelTastingRequest:
    data = {
        "vintage": 2019,
        "wine_name": "珍藏干红",
        "winemaker": "张三",
        "parcels": [
            BarrelParcelInput(
                name="东坡地块",
                variety="梅洛",
                volume_liters=700.0,
                lab_metrics={"酒精度": 14.2, "pH": 3.6},
            ),
            BarrelParcelInput(
                name="西坡地块",
                variety="品丽珠",
                volume_liters=300.0,
                lab_metrics={"酒精度": 13.5},
            ),
        ],
        "tasting_notes": ["果香浓郁，单宁细腻"],
        "keywords": ["果香", "单宁", "橡木"],
    }
    data.update(overrides)
    return BarrelTastingRequest(**data)


class TestVintageFollowsRequest:
    @pytest.mark.parametrize("vintage", [2019, 2020, 2021])
    def test_title_and_body_use_request_vintage(self, vintage):
        report = make_generator().generate_report(make_request(vintage=vintage))

        assert report.vintage == vintage
        assert report.title.startswith(f"{vintage}年份")
        assert f"年份: {vintage}" in report.markdown

    def test_no_hardcoded_year_leaks(self):
        vintage = 2019
        report = make_generator().generate_report(make_request(vintage=vintage))

        years_in_title = re.findall(r"20\d{2}", report.title)
        years_in_markdown = re.findall(r"20\d{2}", report.markdown)
        assert years_in_title == [str(vintage)]
        assert set(years_in_markdown) == {str(vintage)}

    def test_prompt_uses_request_vintage(self):
        generator = make_generator()
        request = make_request(vintage=2019)
        blend = generator._compute_blend(request.parcels)

        _, user_prompt = generator._build_prompt(request, blend)

        assert "2019年份" in user_prompt
        assert "年份: 2019" in user_prompt
        assert "2023" not in user_prompt

    def test_markdown_section_structure(self):
        report = make_generator().generate_report(make_request())

        assert "## 一、基本信息" in report.markdown
        assert "## 二、品评记录" in report.markdown
        assert "## 三、调配建议" in report.markdown
        assert "## 四、总结" in report.markdown


class TestFallbackHasNoFakeBlend:
    def test_ratios_computed_from_parcels_and_sum_to_100(self):
        report = make_generator().generate_report(make_request())

        ratios = {c.variety: c.ratio_percent for c in report.blend_components}
        assert ratios == {"梅洛": 70.0, "品丽珠": 30.0}
        assert sum(ratios.values()) == 100.0
        assert all(c.status == BLEND_STATUS_CALCULATED for c in report.blend_components)

    def test_ratios_aggregated_by_variety(self):
        request = make_request(
            parcels=[
                BarrelParcelInput(name="A区", variety="梅洛", volume_liters=200.0),
                BarrelParcelInput(name="B区", variety="梅洛", volume_liters=400.0),
                BarrelParcelInput(name="C区", variety="品丽珠", volume_liters=400.0),
            ]
        )
        report = make_generator().generate_report(request)

        ratios = {c.variety: c.ratio_percent for c in report.blend_components}
        assert ratios == {"梅洛": 60.0, "品丽珠": 40.0}

    def test_rounding_drift_still_sums_to_100(self):
        request = make_request(
            parcels=[
                BarrelParcelInput(name="A区", variety="甲", volume_liters=1.0),
                BarrelParcelInput(name="B区", variety="乙", volume_liters=1.0),
                BarrelParcelInput(name="C区", variety="丙", volume_liters=1.0),
            ]
        )
        report = make_generator().generate_report(request)

        total = sum(c.ratio_percent for c in report.blend_components)
        assert total == 100.0

    def test_no_parcels_marks_pending_placeholder(self):
        report = make_generator().generate_report(make_request(parcels=[]))

        assert len(report.blend_components) == 1
        component = report.blend_components[0]
        assert component.status == BLEND_STATUS_PENDING
        assert component.ratio_percent is None
        assert component.note == BLEND_PENDING_NOTE
        assert BLEND_PENDING_NOTE in report.markdown
        assert "| 品种 |" not in report.markdown

    def test_fallback_is_marked_and_distinguishable(self):
        report = make_generator().generate_report(make_request())

        assert report.source == SOURCE_FALLBACK
        assert "本地兜底生成" in report.markdown

    def test_fallback_contains_only_input_data(self):
        request = make_request()
        report = make_generator().generate_report(request)

        assert "东坡地块" in report.markdown
        assert "西坡地块" in report.markdown
        assert "梅洛" in report.markdown
        assert "品丽珠" in report.markdown
        assert "赤霞珠" not in report.markdown
        assert "小维多" not in report.markdown
        assert "南坡地块" not in report.markdown

    def test_fallback_has_no_fabricated_ratios(self):
        report = make_generator().generate_report(make_request())

        for fake_ratio in ("60.0%", "25.0%", "10.0%", "5.0%"):
            assert f"| {fake_ratio} " not in report.markdown
        assert "| 70.0% " in report.markdown
        assert "| 30.0% " in report.markdown

    def test_model_error_falls_back_with_fallback_source(self):
        class BrokenClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("boom")

        generator = make_generator()
        generator.client = BrokenClient()
        report = generator.generate_report(make_request())

        assert report.source == SOURCE_FALLBACK
        assert "本地兜底生成" in report.markdown


class TestPromptOrderingStable:
    def test_same_input_produces_identical_prompt(self):
        generator = make_generator()
        request = make_request()
        blend = generator._compute_blend(request.parcels)

        first = generator._build_prompt(request, blend)
        second = generator._build_prompt(request, blend)

        assert first == second

    def test_keyword_order_does_not_change_prompt(self):
        generator = make_generator()
        request_a = make_request(keywords=["果香", "单宁", "橡木"])
        request_b = make_request(keywords=["橡木", "果香", "单宁"])

        prompt_a = generator._build_prompt(
            request_a, generator._compute_blend(request_a.parcels)
        )
        prompt_b = generator._build_prompt(
            request_b, generator._compute_blend(request_b.parcels)
        )

        assert prompt_a == prompt_b

    def test_keywords_sorted_in_prompt(self):
        generator = make_generator()
        request = make_request(keywords=["橡木", "果香", "单宁", "果香"])

        _, user_prompt = generator._build_prompt(
            request, generator._compute_blend(request.parcels)
        )

        assert "单宁, 果香, 橡木" in user_prompt

    def test_parcel_order_does_not_change_prompt(self):
        generator = make_generator()
        shuffled = list(reversed(make_request().parcels))
        request_a = make_request()
        request_b = make_request(parcels=shuffled)

        prompt_a = generator._build_prompt(
            request_a, generator._compute_blend(request_a.parcels)
        )
        prompt_b = generator._build_prompt(
            request_b, generator._compute_blend(request_b.parcels)
        )

        assert prompt_a == prompt_b

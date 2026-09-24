import json
from datetime import datetime
from typing import List, Tuple

from openai import OpenAI

from ..core.config import settings
from ..models.schemas import (
    BarrelParcelInput,
    BarrelTastingRequest,
    BarrelTastingReportResponse,
    BlendComponent,
)

BLEND_STATUS_CALCULATED = "calculated"
BLEND_STATUS_PENDING = "pending_confirmation"
BLEND_PENDING_NOTE = "待酿酒师确认"

SOURCE_MODEL = "model"
SOURCE_FALLBACK = "fallback"

REPORT_SECTIONS = ["基本信息", "品评记录", "调配建议", "总结"]


class BarrelReportGenerator:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.client = None

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def generate_report(self, request: BarrelTastingRequest) -> BarrelTastingReportResponse:
        blend_components = self._compute_blend(request.parcels)
        title = self._build_title(request)

        if self.client:
            markdown, source = self._generate_with_model(request, title, blend_components)
        else:
            markdown, source = self._generate_fallback(request, title, blend_components)

        return BarrelTastingReportResponse(
            vintage=request.vintage,
            title=title,
            markdown=markdown,
            blend_components=blend_components,
            source=source,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _build_title(self, request: BarrelTastingRequest) -> str:
        wine_name = request.wine_name or ""
        return f"{request.vintage}年份{wine_name}桶边品评报告"

    def _compute_blend(self, parcels: List[BarrelParcelInput]) -> List[BlendComponent]:
        usable = [p for p in parcels if p.volume_liters and p.volume_liters > 0]
        total_volume = sum(p.volume_liters for p in usable)

        if not usable or total_volume <= 0:
            return [
                BlendComponent(
                    variety="",
                    ratio_percent=None,
                    volume_liters=None,
                    status=BLEND_STATUS_PENDING,
                    note=BLEND_PENDING_NOTE,
                )
            ]

        volumes_by_variety = {}
        for parcel in usable:
            volumes_by_variety[parcel.variety] = (
                volumes_by_variety.get(parcel.variety, 0.0) + parcel.volume_liters
            )

        varieties = sorted(volumes_by_variety.keys())
        ratios = {
            variety: round(volumes_by_variety[variety] / total_volume * 100, 1)
            for variety in varieties
        }
        drift = round(100.0 - sum(ratios.values()), 1)
        if drift != 0:
            largest = max(varieties, key=lambda v: volumes_by_variety[v])
            ratios[largest] = round(ratios[largest] + drift, 1)

        return [
            BlendComponent(
                variety=variety,
                ratio_percent=ratios[variety],
                volume_liters=round(volumes_by_variety[variety], 1),
                status=BLEND_STATUS_CALCULATED,
                note=None,
            )
            for variety in varieties
        ]

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        return sorted({kw.strip() for kw in keywords if kw and kw.strip()})

    def _sorted_parcels(self, parcels: List[BarrelParcelInput]) -> List[BarrelParcelInput]:
        return sorted(parcels, key=lambda p: (p.name, p.variety))

    def _format_parcel_line(self, parcel: BarrelParcelInput) -> str:
        metrics = ", ".join(
            f"{key}={parcel.lab_metrics[key]}"
            for key in sorted(parcel.lab_metrics.keys())
        )
        metrics_str = f"，理化指标: {metrics}" if metrics else ""
        return (
            f"地块 {parcel.name}，品种 {parcel.variety}，"
            f"体积 {parcel.volume_liters}L{metrics_str}"
        )

    def _build_prompt(
        self,
        request: BarrelTastingRequest,
        blend_components: List[BlendComponent],
    ) -> Tuple[str, str]:
        keywords = self._normalize_keywords(request.keywords)

        system_prompt = """你是一名资深葡萄酒酿酒师助理，擅长根据桶边品评记录撰写年份桶边品评报告。
必须严格使用用户提供的年份、地块、品种与理化数据，不得虚构任何数据。
请用中文回答，保持专业性和准确性。"""

        parcel_lines = [
            f"- {self._format_parcel_line(p)}" for p in self._sorted_parcels(request.parcels)
        ]
        parcels_text = "\n".join(parcel_lines) if parcel_lines else "（未提供地块数据）"

        blend_lines = []
        for component in blend_components:
            if component.status == BLEND_STATUS_PENDING:
                blend_lines.append(f"- {BLEND_PENDING_NOTE}")
            else:
                blend_lines.append(
                    f"- {component.variety}: {component.ratio_percent}%"
                    f"（{component.volume_liters}L）"
                )
        blend_text = "\n".join(blend_lines)

        notes_text = "\n".join(f"- {note}" for note in request.tasting_notes) or "（无）"
        keywords_text = ", ".join(keywords) if keywords else "（无）"

        user_prompt = f"""请根据以下信息撰写{request.vintage}年份桶边品评报告的核心内容：

=== 基本信息 ===
年份: {request.vintage}
酒款: {request.wine_name or '未命名'}
酿酒师: {request.winemaker or '未知'}

=== 地块与理化数据 ===
{parcels_text}

=== 品评关键词 ===
{keywords_text}

=== 品评记录 ===
{notes_text}

=== 调配方案（已根据地块体积计算，请仅作解读，不得改动比例） ===
{blend_text}

=== 输出要求 ===
请输出JSON格式，包含以下字段：
1. tasting_summary: 品评记录总结（150-250字）
2. blend_rationale: 对上述调配方案的解读（100-200字；若方案为"{BLEND_PENDING_NOTE}"则输出空字符串）
3. overall: 总结与建议（100-200字）
"""

        return system_prompt, user_prompt

    def _generate_with_model(
        self,
        request: BarrelTastingRequest,
        title: str,
        blend_components: List[BlendComponent],
    ) -> Tuple[str, str]:
        system_prompt, user_prompt = self._build_prompt(request, blend_components)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            markdown = self._render_markdown(
                request=request,
                title=title,
                blend_components=blend_components,
                tasting_summary=result.get("tasting_summary", ""),
                blend_rationale=result.get("blend_rationale", ""),
                overall=result.get("overall", ""),
                source=SOURCE_MODEL,
            )
            return markdown, SOURCE_MODEL
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback(request, title, blend_components)

    def _generate_fallback(
        self,
        request: BarrelTastingRequest,
        title: str,
        blend_components: List[BlendComponent],
    ) -> Tuple[str, str]:
        keywords = self._normalize_keywords(request.keywords)

        tasting_parts = [
            self._format_parcel_line(p) for p in self._sorted_parcels(request.parcels)
        ]
        if request.tasting_notes:
            tasting_parts.extend(request.tasting_notes)
        if keywords:
            tasting_parts.append(f"品评关键词：{'、'.join(keywords)}。")
        tasting_summary = "\n".join(tasting_parts) if tasting_parts else "（未提供品评记录）"

        overall = "本报告由本地兜底逻辑生成，仅整理输入数据，未包含模型分析结论。"

        markdown = self._render_markdown(
            request=request,
            title=title,
            blend_components=blend_components,
            tasting_summary=tasting_summary,
            blend_rationale="",
            overall=overall,
            source=SOURCE_FALLBACK,
        )
        return markdown, SOURCE_FALLBACK

    def _render_markdown(
        self,
        request: BarrelTastingRequest,
        title: str,
        blend_components: List[BlendComponent],
        tasting_summary: str,
        blend_rationale: str,
        overall: str,
        source: str,
    ) -> str:
        source_label = "AI模型生成" if source == SOURCE_MODEL else "本地兜底生成（未经模型）"

        lines = [
            f"# {title}",
            "",
            f"## 一、{REPORT_SECTIONS[0]}",
            "",
            f"- 年份: {request.vintage}",
            f"- 酒款: {request.wine_name or '未命名'}",
            f"- 酿酒师: {request.winemaker or '未知'}",
            f"- 数据来源: {source_label}",
            "",
            f"## 二、{REPORT_SECTIONS[1]}",
            "",
            tasting_summary,
            "",
            f"## 三、{REPORT_SECTIONS[2]}",
            "",
        ]

        if all(c.status == BLEND_STATUS_PENDING for c in blend_components):
            lines.append(f"调配方案：{BLEND_PENDING_NOTE}。")
        else:
            lines.append("| 品种 | 比例 | 体积(L) |")
            lines.append("| --- | --- | --- |")
            for component in blend_components:
                lines.append(
                    f"| {component.variety} | {component.ratio_percent}% "
                    f"| {component.volume_liters} |"
                )
            lines.append("")
            lines.append("注：比例按传入地块体积加权聚合计算，合计100%。")

        if blend_rationale:
            lines.extend(["", blend_rationale])

        lines.extend([
            "",
            f"## 四、{REPORT_SECTIONS[3]}",
            "",
            overall,
            "",
        ])

        return "\n".join(lines)

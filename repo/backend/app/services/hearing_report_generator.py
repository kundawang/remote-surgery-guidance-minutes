import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..core.config import settings
from ..models.schemas import TranscriptSegment, HearingReportResponse

FALLBACK_NOTICE = "以下为模板提示，需人工补充"
EMPTY_MATRIX_HINT = "本次会议未收集到代表建议，行动矩阵为空，待后续补充。"
MATRIX_COLUMNS = ["序号", "建议措施", "责任部门", "时限", "状态"]


class HearingReportGenerator:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.client = None

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def generate_report(
        self,
        hearing_info: Dict[str, Any],
        locations: List[str],
        solutions: List[str],
        transcripts: List[TranscriptSegment]
    ) -> HearingReportResponse:
        if self.client:
            return self._generate_with_openai(hearing_info, locations, solutions, transcripts)
        return self._generate_fallback(hearing_info, locations, solutions, transcripts)

    def _generate_with_openai(
        self,
        hearing_info: Dict[str, Any],
        locations: List[str],
        solutions: List[str],
        transcripts: List[TranscriptSegment]
    ) -> HearingReportResponse:
        full_transcript = self._format_transcripts(transcripts)
        locations_text = "、".join(locations) if locations else "（未提供）"
        solutions_text = "\n".join(f"- {s}" for s in solutions) if solutions else "（无）"

        system_prompt = """你是一名专业的城市交通治理分析师，擅长整理交通拥堵听证会材料。
请根据会议信息、涉及路段、代表建议和发言记录，生成结构化的听证会报告内容。
请用中文回答，保持客观、准确，不得引入材料之外的路段、部门或数据。"""

        user_prompt = f"""请分析以下交通拥堵听证会材料，并生成结构化报告内容：

=== 会议信息 ===
{json.dumps(hearing_info, ensure_ascii=False, indent=2)}

=== 涉及路段 ===
{locations_text}

=== 各方代表建议 ===
{solutions_text}

=== 发言记录 ===
{full_transcript}

=== 输出要求 ===
请输出JSON格式，包含以下字段：
1. minutes: 会议纪要正文（Markdown，不含"参会人员"，参会人员由系统按角色分组生成）
2. cause_analysis: 成因分析正文（Markdown，仅围绕上述涉及路段展开）
3. action_matrix: 行动矩阵行列表，每行包含"建议措施"、"责任部门"、"时限"、"状态"，
   且每一行必须对应一条代表建议，不得编造建议之外的措施
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            minutes = self._append_attendees(
                result.get("minutes", ""), transcripts
            )
            cause_analysis = result.get("cause_analysis", "")
            action_matrix = self._normalize_action_matrix(
                result.get("action_matrix", [])
            )

            return HearingReportResponse(
                minutes=minutes,
                cause_analysis=cause_analysis,
                action_matrix=action_matrix,
                full_transcript=full_transcript,
                report_markdown=self._assemble_markdown(
                    minutes, cause_analysis, action_matrix, full_transcript
                ),
                is_fallback=False,
                fallback_notice=None
            )

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback(hearing_info, locations, solutions, transcripts)

    def _generate_fallback(
        self,
        hearing_info: Dict[str, Any],
        locations: List[str],
        solutions: List[str],
        transcripts: List[TranscriptSegment]
    ) -> HearingReportResponse:
        minutes = self._build_fallback_minutes(hearing_info, locations, transcripts)
        cause_analysis = self._build_fallback_cause_analysis(locations)
        action_matrix = self._build_fallback_action_matrix(solutions)
        full_transcript = self._format_transcripts(transcripts)

        return HearingReportResponse(
            minutes=minutes,
            cause_analysis=cause_analysis,
            action_matrix=action_matrix,
            full_transcript=full_transcript,
            report_markdown=self._assemble_markdown(
                minutes, cause_analysis, action_matrix, full_transcript
            ),
            is_fallback=True,
            fallback_notice=FALLBACK_NOTICE
        )

    def _build_fallback_minutes(
        self,
        hearing_info: Dict[str, Any],
        locations: List[str],
        transcripts: List[TranscriptSegment]
    ) -> str:
        lines = [f"> {FALLBACK_NOTICE}：模型不可用，本纪要由离线模板生成，会议结论待人工整理。", ""]
        for key, value in hearing_info.items():
            if value:
                lines.append(f"**{key}**: {value}")
        if hearing_info:
            lines.append("")
        if locations:
            lines.append(f"**涉及路段**: {'、'.join(locations)}")
        else:
            lines.append("**涉及路段**: （本次会议未提供路段信息）")
        return self._append_attendees("\n".join(lines), transcripts)

    def _build_fallback_cause_analysis(self, locations: List[str]) -> str:
        lines = [f"> {FALLBACK_NOTICE}", ""]
        if not locations:
            lines.append("本次会议未提供涉及路段，成因分析待人工补充。")
        else:
            for loc in locations:
                lines.append(f"- 关于「{loc}」的拥堵成因：待人工分析补充。")
        return "\n".join(lines)

    def _build_fallback_action_matrix(self, solutions: List[str]) -> List[Dict[str, Any]]:
        rows = []
        for index, solution in enumerate(solutions, start=1):
            rows.append({
                "序号": index,
                "建议措施": solution,
                "责任部门": "待人工确定",
                "时限": "待人工确定",
                "状态": "待人工评估"
            })
        return rows

    def _normalize_action_matrix(self, rows: Any) -> List[Dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        normalized = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            normalized.append({
                "序号": row.get("序号", index),
                "建议措施": row.get("建议措施", ""),
                "责任部门": row.get("责任部门", ""),
                "时限": row.get("时限", ""),
                "状态": row.get("状态", "")
            })
        return normalized

    def _append_attendees(self, minutes: str, transcripts: List[TranscriptSegment]) -> str:
        grouped = self._group_attendees_by_role(transcripts)
        lines = [minutes.rstrip(), "", "**参会人员**（按角色分组）:"]
        if not grouped:
            lines.append("- （无发言记录，参会人员待人工补充）")
        else:
            for role, speakers in grouped.items():
                lines.append(f"- {role}: {'、'.join(speakers)}")
        return "\n".join(lines)

    def _group_attendees_by_role(self, transcripts: List[TranscriptSegment]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for t in transcripts:
            role = t.speaker_role or "其他"
            speakers = grouped.setdefault(role, [])
            if t.speaker not in speakers:
                speakers.append(t.speaker)
        return grouped

    def _assemble_markdown(
        self,
        minutes: str,
        cause_analysis: str,
        action_matrix: List[Dict[str, Any]],
        full_transcript: str
    ) -> str:
        sections = [
            "# 交通拥堵听证会报告",
            "",
            "## 一、会议纪要",
            "",
            minutes,
            "",
            "## 二、成因分析",
            "",
            cause_analysis,
            "",
            "## 三、行动矩阵",
            "",
            self._render_action_matrix(action_matrix),
            "",
            "## 四、完整发言记录",
            "",
            full_transcript if full_transcript else "（无发言记录）",
        ]
        return "\n".join(sections)

    def _render_action_matrix(self, action_matrix: List[Dict[str, Any]]) -> str:
        header = "| " + " | ".join(MATRIX_COLUMNS) + " |"
        divider = "|" + "|".join([" --- "] * len(MATRIX_COLUMNS)) + "|"
        lines = [header, divider]
        for row in action_matrix:
            cells = [str(row.get(col, "") or "") for col in MATRIX_COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
        if not action_matrix:
            lines.append("")
            lines.append(f"（{EMPTY_MATRIX_HINT}）")
        return "\n".join(lines)

    def _format_transcripts(self, transcripts: List[TranscriptSegment]) -> str:
        lines = []
        for t in transcripts:
            time_str = f"[{self._format_time(t.start_time)}]"
            speaker = f"{t.speaker}({t.speaker_role})"
            lines.append(f"{time_str} {speaker}: {t.text}")
        return "\n".join(lines)

    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

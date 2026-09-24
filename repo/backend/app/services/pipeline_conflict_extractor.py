"""地下管线迁改会议：冲突实体抽取与兜底摘要。

抽取规则（统计口径）：
- 每个抽取器按 (position, 归一化后的值) 去重：同一处文字被多条正则/多个关键词
  重复命中时只保留一条，position 为命中在原文中的起始偏移。
- 关键词/正则互相包含时取最长匹配优先（如 "燃气管" 命中后，"燃气" 不再重复计入）。
- 桩号统一归一化为 "K<里程>+<偏移>" 形式（K 大写、补 K 前缀、去内部空白）。
- 一段发言（一条 transcript）无论命中多少种管线类型/多少个桩号，都计为 1 处冲突。
- 全部抽取逻辑为纯函数：同一份输入重复运行，结果完全一致。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

COUNTING_RULE = (
    "一段发言（一条 transcript）只要命中至少一个管线类型或桩号，即计为 1 处冲突；"
    "同一段发言命中多种管线类型仍计为 1 处。实体按 (position, 归一化值) 去重，"
    "互相包含的关键词/正则取最长匹配。"
)

DEFAULT_ACTION_ITEM_LEAD_DAYS = 7


@dataclass(frozen=True)
class ExtractedEntity:
    """单个抽取实体。value 为归一化后的值，raw 为原文命中，position 为起始偏移。"""

    value: str
    raw: str
    position: int

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "raw": self.raw, "position": self.position}


def _dedup_entities(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
    """按 (position, 归一化值) 去重，保留首次出现，输出顺序稳定。"""
    seen = set()
    result: List[ExtractedEntity] = []
    for entity in entities:
        key = (entity.position, entity.value)
        if key in seen:
            continue
        seen.add(key)
        result.append(entity)
    return result


def normalize_stake_number(raw: str) -> str:
    """桩号归一化：去空白、K 大写、补 K 前缀。如 "k12+300" -> "K12+300"。"""
    compact = re.sub(r"\s+", "", raw).upper()
    if not compact.startswith("K"):
        compact = "K" + compact
    return compact


class StakeNumberExtractor:
    """桩号抽取。多条正则可能命中同一处桩号，按 (position, 归一化值) 去重。"""

    _STAKE = r"[Kk]?\s*\d+\s*\+\s*\d+(?:\.\d+)?"
    PATTERNS = [
        re.compile(r"桩号\s*[:：]?\s*(" + _STAKE + r")"),
        re.compile(r"(" + _STAKE + r")"),
    ]

    def extract(self, text: str) -> List[ExtractedEntity]:
        entities: List[ExtractedEntity] = []
        for pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(1)
                entities.append(
                    ExtractedEntity(
                        value=normalize_stake_number(raw),
                        raw=raw,
                        position=match.start(1),
                    )
                )
        return _dedup_entities(entities)


# 管线类型：归一化类型 -> 别名（匹配时按长度降序，保证最长匹配优先）
PIPELINE_TYPE_ALIASES: Dict[str, List[str]] = {
    "燃气": ["燃气管道", "燃气管线", "燃气管", "燃气"],
    "给水": ["给水管道", "给水管线", "给水管", "自来水管", "给水"],
    "排水": ["排水管道", "排水管线", "排水管", "排水"],
    "雨水": ["雨水管道", "雨水管", "雨水"],
    "污水": ["污水管道", "污水管", "污水"],
    "电力": ["电力电缆", "电力管线", "电力管", "电缆", "电力"],
    "通信": ["通信光缆", "通信管线", "通信管", "光缆", "通信"],
    "热力": ["热力管道", "热力管", "热力"],
    "中水": ["中水管道", "中水管", "中水"],
}


def _longest_first_alternation(words: Sequence[str], suffix: str = "") -> re.Pattern:
    ordered = sorted(set(words), key=len, reverse=True)
    body = "|".join(re.escape(word) for word in ordered)
    return re.compile(f"(?:{body}){suffix}")


class PipelineTypeExtractor:
    """管线类型抽取。

    关键词按长度降序组成一个 alternation，正则引擎在同一位置优先取最长匹配；
    另有一条通用 "XX管线/XX管道" 正则兜底，二者命中同一段文字时按
    (position, 归一化类型) 去重，"燃气"/"燃气管" 不会各算一条。
    """

    _ALIAS_TO_CANONICAL: Dict[str, str] = {
        alias: canonical
        for canonical, aliases in PIPELINE_TYPE_ALIASES.items()
        for alias in aliases
    }
    # 负向先行：单位名称（如 "燃气公司"）中的字样不算管线类型
    _KEYWORD_PATTERN = _longest_first_alternation(
        list(_ALIAS_TO_CANONICAL), suffix=r"(?!公司|集团)"
    )
    _GENERIC_PATTERN = re.compile(r"([一-龥]{1,6}?)(?:管线|管道)")

    def extract(self, text: str) -> List[ExtractedEntity]:
        entities: List[ExtractedEntity] = []
        for match in self._KEYWORD_PATTERN.finditer(text):
            raw = match.group(0)
            entities.append(
                ExtractedEntity(
                    value=self._ALIAS_TO_CANONICAL[raw],
                    raw=raw,
                    position=match.start(),
                )
            )
        for match in self._GENERIC_PATTERN.finditer(text):
            prefix = match.group(1)
            canonical = self._canonical_for_prefix(prefix)
            if canonical is None:
                continue  # 未知名词（如 "高压"）不单独成类，关键词已覆盖已知类型
            offset = len(prefix) - len(canonical)
            entities.append(
                ExtractedEntity(
                    value=canonical,
                    raw=match.group(0),
                    position=match.start(1) + offset,
                )
            )
        return _dedup_entities(entities)

    @classmethod
    def _canonical_for_prefix(cls, prefix: str) -> Optional[str]:
        for canonical in PIPELINE_TYPE_ALIASES:
            if prefix.endswith(canonical):
                return canonical
        for alias, canonical in cls._ALIAS_TO_CANONICAL.items():
            if prefix.endswith(alias):
                return canonical
        return None


class BurialDepthExtractor:
    """埋深抽取，归一化为 "<数值>m"（米）。"""

    _DEPTH = r"(\d+(?:\.\d+)?)\s*(?:米|m|M)"
    PATTERNS = [
        re.compile(r"埋深\s*[:：]?\s*" + _DEPTH),
        re.compile(r"埋设深度\s*[:：]?\s*" + _DEPTH),
    ]

    def extract(self, text: str) -> List[ExtractedEntity]:
        entities: List[ExtractedEntity] = []
        for pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(1)
                entities.append(
                    ExtractedEntity(
                        value=f"{float(raw):g}m",
                        raw=raw,
                        position=match.start(1),
                    )
                )
        return _dedup_entities(entities)


ORGANIZATION_KEYWORDS: List[str] = [
    "自来水公司", "燃气公司", "供电公司", "电力公司", "热力公司",
    "排水公司", "通信公司", "电信公司", "移动公司", "联通公司",
    "建设单位", "施工单位", "设计单位", "监理单位",
    "市政管理处", "规划局", "住建局", "交通局", "城管局",
    "管线单位", "产权单位",
]


class OrganizationExtractor:
    """涉及单位抽取，最长匹配优先。"""

    _PATTERN = _longest_first_alternation(ORGANIZATION_KEYWORDS)

    def extract(self, text: str) -> List[ExtractedEntity]:
        entities = [
            ExtractedEntity(value=match.group(0), raw=match.group(0), position=match.start())
            for match in self._PATTERN.finditer(text)
        ]
        return _dedup_entities(entities)


class PipelineConflictExtractor:
    """地下管线迁改冲突抽取。

    出参结构：description / speaker / organization / stake_numbers /
    burial_depths / pipeline_types / mentioned_organizations / confidence / timestamp。
    confidence 语义：0.5 起，每多一类命中实体 +0.1，封顶 0.95，
    仅反映抽取证据充分度，为输入的确定性函数。
    """

    def __init__(self) -> None:
        self.stake_extractor = StakeNumberExtractor()
        self.depth_extractor = BurialDepthExtractor()
        self.pipeline_extractor = PipelineTypeExtractor()
        self.organization_extractor = OrganizationExtractor()

    def extract_conflict(
        self,
        description: str,
        speaker: str = "",
        organization: str = "",
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        stake_numbers = self.stake_extractor.extract(description)
        burial_depths = self.depth_extractor.extract(description)
        pipeline_types = self.pipeline_extractor.extract(description)
        mentioned_organizations = self.organization_extractor.extract(description)

        categories_hit = sum(
            1
            for hits in (stake_numbers, burial_depths, pipeline_types, mentioned_organizations)
            if hits
        )
        confidence = min(0.5 + 0.1 * categories_hit, 0.95)

        return {
            "description": description,
            "speaker": speaker,
            "organization": organization,
            "stake_numbers": [e.to_dict() for e in stake_numbers],
            "burial_depths": [e.to_dict() for e in burial_depths],
            "pipeline_types": [e.to_dict() for e in pipeline_types],
            "mentioned_organizations": [e.to_dict() for e in mentioned_organizations],
            "confidence": confidence,
            "timestamp": timestamp,
        }

    def extract_conflicts(self, segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """逐段抽取。一段发言命中至少一个管线类型或桩号即记为 1 处冲突。"""
        conflicts: List[Dict[str, Any]] = []
        for segment in segments:
            record = self.extract_conflict(
                description=segment.get("text", ""),
                speaker=segment.get("speaker", ""),
                organization=segment.get("organization", ""),
                timestamp=segment.get("timestamp"),
            )
            if record["pipeline_types"] or record["stake_numbers"]:
                conflicts.append(record)
        return conflicts

    def summarize(self, conflicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """冲突统计。返回中显式携带 counting_rule，说明统计口径。"""
        pipeline_type_counts: Dict[str, int] = {}
        stake_numbers = set()
        for conflict in conflicts:
            for entity in conflict["pipeline_types"]:
                pipeline_type_counts[entity["value"]] = (
                    pipeline_type_counts.get(entity["value"], 0) + 1
                )
            for entity in conflict["stake_numbers"]:
                stake_numbers.add(entity["value"])
        return {
            "total_conflicts": len(conflicts),
            "counting_rule": COUNTING_RULE,
            "pipeline_type_counts": dict(sorted(pipeline_type_counts.items())),
            "unique_stake_numbers": sorted(stake_numbers),
        }


def _derive_deadline(meeting_date: Optional[Any], lead_days: int) -> str:
    """行动项截止日期：基于会议日期推算；没有真实日期时返回 "待确定"。"""
    parsed: Optional[date] = None
    if isinstance(meeting_date, datetime):
        parsed = meeting_date.date()
    elif isinstance(meeting_date, date):
        parsed = meeting_date
    elif isinstance(meeting_date, str) and meeting_date.strip():
        try:
            parsed = datetime.fromisoformat(meeting_date.strip()).date()
        except ValueError:
            parsed = None
    if parsed is None:
        return "待确定"
    return (parsed + timedelta(days=lead_days)).isoformat()


def generate_fallback_summary(
    conflicts: Sequence[Dict[str, Any]],
    meeting_date: Optional[Any] = None,
    lead_days: int = DEFAULT_ACTION_ITEM_LEAD_DAYS,
) -> Dict[str, Any]:
    """兜底摘要（无 LLM 时使用）。

    行动项截止日期不写死：有会议日期时按会议日期 + lead_days 推算，
    否则为 "待确定"。
    """
    deadline = _derive_deadline(meeting_date, lead_days)
    owners: List[str] = []
    for conflict in conflicts:
        for entity in conflict["mentioned_organizations"]:
            if entity["value"] not in owners:
                owners.append(entity["value"])

    action_items = [
        {
            "action": f"核实全部 {len(conflicts)} 处冲突点位的管线权属与现状资料",
            "owner": owners[0] if owners else "待定",
            "deadline": deadline,
        },
        {
            "action": "组织产权单位现场确认迁改方案并反馈时限",
            "owner": "、".join(owners) if owners else "待定",
            "deadline": deadline,
        },
    ]
    return {
        "total_conflicts": len(conflicts),
        "counting_rule": COUNTING_RULE,
        "action_items": action_items,
    }

"""地下管线迁改会议：冲突实体抽取与兜底摘要。

统计口径（counting_rule，随统计结果返回）：
- 一条发言（一个 transcript 段）只要命中至少一个桩号或管线类型，即计为 1 处
  冲突；同一段发言命中多种管线类型、多个桩号，仍只计 1 处。
- 每个抽取器按 (position, 归一化后的值) 去重：同一处文字被多条正则/多个
  关键词重复命中时只保留一条；position 为命中在原文中的起始偏移，随结果
  返回便于定位。
- 关键词/正则互相包含时取最长匹配（如 "燃气管" 命中后，"燃气" 不再重复
  计入）。
- 桩号统一归一化为 "K<里程>+<偏移>" 形式（补 K 前缀、K 大写、去内部空白）。
- 全部为纯函数式抽取：同一份输入重复运行，结果完全一致。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

COUNTING_RULE = (
    "一条发言（一个 transcript 段）命中至少一个桩号或管线类型即计为 1 处冲突；"
    "同一段发言命中多种管线类型仍计为 1 处。实体按 (position, 归一化值) 去重，"
    "互相包含的关键词/正则取最长匹配。"
)

#: 兜底摘要中行动项相对于会议日期的默认提前天数
DEFAULT_LEAD_DAYS = 7

#: 没有真实会议日期时行动项截止日期的占位值（不写死任何具体日期）
UNDETERMINED_DEADLINE = "待确定"


@dataclass(frozen=True)
class Entity:
    """单个抽取实体。value 为归一化后的值，raw 为原文命中，position 为起始偏移。"""

    value: str
    raw: str
    position: int

    @property
    def end(self) -> int:
        return self.position + len(self.raw)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "raw": self.raw, "position": self.position}


def _resolve_entities(candidates: List[Entity]) -> List[Entity]:
    """去重 + 最长匹配优先。

    1. 按 (position, 归一化值) 去重，同一键保留 raw 最长的一条；
    2. 若某实体的命中区间被另一个归一化值相同的实体完整包含，丢弃较短者
       （覆盖关键词与兜底正则命中同一段文字的情形）。
    输出按 position 升序，顺序稳定。
    """
    best: Dict[tuple, Entity] = {}
    for entity in candidates:
        key = (entity.position, entity.value)
        current = best.get(key)
        if current is None or len(entity.raw) > len(current.raw):
            best[key] = entity
    entities = sorted(best.values(), key=lambda e: (e.position, -len(e.raw)))

    resolved: List[Entity] = []
    for entity in entities:
        contained = any(
            other is not entity
            and other.value == entity.value
            and other.position <= entity.position
            and entity.end <= other.end
            and len(other.raw) > len(entity.raw)
            for other in entities
        )
        if not contained:
            resolved.append(entity)
    return resolved


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

    def extract(self, text: str) -> List[Entity]:
        candidates = [
            Entity(
                value=normalize_stake_number(match.group(1)),
                raw=match.group(1),
                position=match.start(1),
            )
            for pattern in self.PATTERNS
            for match in pattern.finditer(text)
        ]
        return _resolve_entities(candidates)


#: 管线类型：归一化类型 -> 别名（"燃气管"/"燃气管道" 等都归一到 "燃气"）
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

_ALIAS_TO_CANONICAL: Dict[str, str] = {
    alias: canonical
    for canonical, aliases in PIPELINE_TYPE_ALIASES.items()
    for alias in aliases
}


def _longest_first_pattern(words: Sequence[str], suffix: str = "") -> re.Pattern:
    """把关键词按长度降序组成 alternation：同一位置正则引擎优先取最长匹配。"""
    ordered = sorted(set(words), key=len, reverse=True)
    return re.compile("(?:" + "|".join(re.escape(w) for w in ordered) + ")" + suffix)


class PipelineTypeExtractor:
    """管线类型抽取。

    关键词按长度降序匹配（最长匹配优先），另有一条通用 "XX管线/XX管道" 正则
    兜底；二者命中同一段文字时经 _resolve_entities 去重，"燃气"/"燃气管"
    不会各算一条。单位名称（如 "燃气公司"）中的字样不计为管线类型。
    """

    _KEYWORD_PATTERN = _longest_first_pattern(
        list(_ALIAS_TO_CANONICAL), suffix=r"(?!公司|集团|有限)"
    )
    _GENERIC_PATTERN = re.compile(r"([一-龥]{1,6}?)(?:管线|管道)")

    def extract(self, text: str) -> List[Entity]:
        candidates: List[Entity] = []
        for match in self._KEYWORD_PATTERN.finditer(text):
            raw = match.group(0)
            candidates.append(
                Entity(
                    value=_ALIAS_TO_CANONICAL[raw],
                    raw=raw,
                    position=match.start(),
                )
            )
        for match in self._GENERIC_PATTERN.finditer(text):
            prefix = match.group(1)
            canonical = self._canonical_for_prefix(prefix)
            if canonical is None:
                continue  # 未知名词不单独成类，已知类型已被关键词覆盖
            offset = len(prefix) - len(canonical)
            candidates.append(
                Entity(
                    value=canonical,
                    raw=match.group(0),
                    position=match.start(1) + offset,
                )
            )
        return _resolve_entities(candidates)

    @staticmethod
    def _canonical_for_prefix(prefix: str) -> Optional[str]:
        for canonical in PIPELINE_TYPE_ALIASES:
            if prefix.endswith(canonical):
                return canonical
        for alias, canonical in _ALIAS_TO_CANONICAL.items():
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

    def extract(self, text: str) -> List[Entity]:
        candidates = [
            Entity(
                value=f"{float(match.group(1)):g}m",
                raw=match.group(1),
                position=match.start(1),
            )
            for pattern in self.PATTERNS
            for match in pattern.finditer(text)
        ]
        return _resolve_entities(candidates)


ORGANIZATION_KEYWORDS: List[str] = [
    "自来水公司", "燃气公司", "供电公司", "电力公司", "热力公司",
    "排水公司", "通信公司", "电信公司", "移动公司", "联通公司",
    "建设单位", "施工单位", "设计单位", "监理单位",
    "市政管理处", "规划局", "住建局", "交通局", "城管局",
    "管线单位", "产权单位",
]


class OrganizationExtractor:
    """涉及单位抽取，最长匹配优先。"""

    _PATTERN = _longest_first_pattern(ORGANIZATION_KEYWORDS)

    def extract(self, text: str) -> List[Entity]:
        candidates = [
            Entity(value=match.group(0), raw=match.group(0), position=match.start())
            for match in self._PATTERN.finditer(text)
        ]
        return _resolve_entities(candidates)


class PipelineConflictExtractor:
    """地下管线迁改冲突抽取。

    入参：description / speaker / organization / timestamp。
    出参结构：description / speaker / organization / stake_numbers /
    burial_depths / pipeline_types / mentioned_organizations / confidence /
    timestamp。
    confidence 语义：0.5 起，每多一类命中实体 +0.1，封顶 0.95，仅反映抽取
    证据充分度，是输入的确定性函数。
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
        """逐段抽取。一段发言命中至少一个桩号或管线类型即记为 1 处冲突。"""
        conflicts: List[Dict[str, Any]] = []
        for segment in segments:
            record = self.extract_conflict(
                description=segment.get("text", ""),
                speaker=segment.get("speaker", ""),
                organization=segment.get("organization", ""),
                timestamp=segment.get("timestamp"),
            )
            if record["stake_numbers"] or record["pipeline_types"]:
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


def _parse_meeting_date(meeting_date: Optional[Any]) -> Optional[date]:
    if isinstance(meeting_date, datetime):
        return meeting_date.date()
    if isinstance(meeting_date, date):
        return meeting_date
    if isinstance(meeting_date, str) and meeting_date.strip():
        text = meeting_date.strip().replace("/", "-").replace(".", "-")
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    return None


def derive_deadline(meeting_date: Optional[Any], lead_days: int = DEFAULT_LEAD_DAYS) -> str:
    """行动项截止日期：基于会议日期推算；没有真实日期时返回 "待确定"。"""
    parsed = _parse_meeting_date(meeting_date)
    if parsed is None:
        return UNDETERMINED_DEADLINE
    return (parsed + timedelta(days=lead_days)).isoformat()


def generate_fallback_summary(
    conflicts: Sequence[Dict[str, Any]],
    meeting_date: Optional[Any] = None,
    lead_days: int = DEFAULT_LEAD_DAYS,
) -> Dict[str, Any]:
    """兜底摘要（无 LLM 时使用）。

    行动项截止日期不写死：有会议日期时按会议日期 + lead_days 推算，
    否则为 "待确定"。
    """
    deadline = derive_deadline(meeting_date, lead_days)

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

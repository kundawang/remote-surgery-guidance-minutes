"""将转写段与 diarization 说话人段合并的纯逻辑。

规则：
1. 每个 diarization label 独立命名；同一角色存在多个 label 时按首次出现
   顺序加序号（如 张博士1 / 张博士2），单名不加序号。
2. 一个转写段的 speaker 与 speaker_role 必须来自同一条被匹配到的
   diarization 段，不允许分别推断。
3. 时间匹配保持既有语义：先判断转写段中点是否落在某条说话人段区间内；
   没有任何区间包含中点时，取中点与转写段中点距离最近的说话人段。
   多个候选时按 (start_time, label) 做确定性裁决，保证同输入同输出。
4. diarization 段为空（完全匹配不上）时保留原文，speaker 与
   speaker_role 均标注为未知。
"""

from typing import Any, Dict, List, Optional, Tuple

UNKNOWN_SPEAKER = "未知"
UNKNOWN_ROLE = "未知"
DEFAULT_ROLE = "其他"


def _midpoint(start_time: float, end_time: float) -> float:
    return (start_time + end_time) / 2.0


def build_speaker_profile_map(
    speaker_segments: List[Dict[str, Any]],
    role_name_map: Optional[Dict[str, str]] = None,
    default_role: str = DEFAULT_ROLE,
) -> Dict[str, Dict[str, str]]:
    """为每个 diarization label 建立稳定的 {role, name} 档案。

    同一 label 在一次处理中只解析一次，因此始终对应同一个角色和名字。
    同一角色对应多个 label 时，按 label 首次出现顺序加序号区分。
    """
    role_name_map = role_name_map or {}

    label_role: Dict[str, str] = {}
    label_first_start: Dict[str, float] = {}
    for seg in speaker_segments:
        label = seg.get("speaker")
        if label is None:
            continue
        label = str(label)
        if label not in label_role:
            label_role[label] = str(seg.get("role") or default_role)
            label_first_start[label] = float(seg.get("start_time", 0.0))

    # 编号顺序只取决于 label 首次出现的时间与 label 本身，与输入列表顺序无关，
    # 避免同一批 diarization 段以不同顺序传入时编号翻转。
    label_order = sorted(
        label_role,
        key=lambda lbl: (label_first_start[lbl], lbl),
    )

    role_labels: Dict[str, List[str]] = {}
    for label in label_order:
        role_labels.setdefault(label_role[label], []).append(label)

    profile_map: Dict[str, Dict[str, str]] = {}
    for role, labels in role_labels.items():
        base_name = role_name_map.get(role, role)
        multi = len(labels) > 1
        for index, label in enumerate(labels, start=1):
            name = f"{base_name}{index}" if multi else base_name
            profile_map[label] = {"role": role, "name": name}

    return profile_map


def find_matching_segment(
    mid_time: float,
    speaker_segments: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """按中点规则找到唯一、确定的说话人段；无候选时返回 None。"""
    if not speaker_segments:
        return None

    containing = [
        seg for seg in speaker_segments
        if seg["start_time"] <= mid_time <= seg["end_time"]
    ]
    if containing:
        return min(
            containing,
            key=lambda seg: (seg["start_time"], str(seg.get("speaker"))),
        )

    def distance(seg: Dict[str, Any]) -> Tuple[float, float, str]:
        seg_mid = _midpoint(seg["start_time"], seg["end_time"])
        return (
            abs(mid_time - seg_mid),
            seg["start_time"],
            str(seg.get("speaker")),
        )

    return min(speaker_segments, key=distance)


def attach_speaker_info(
    transcript_segments: List[Dict[str, Any]],
    speaker_segments: List[Dict[str, Any]],
    role_name_map: Optional[Dict[str, str]] = None,
    default_role: str = DEFAULT_ROLE,
) -> List[Dict[str, Any]]:
    """把 speaker / speaker_role 挂到转写段上，返回新列表。

    只写入 speaker 与 speaker_role 两个字段，start_time / end_time / text
    等原始内容原样保留。
    """
    profile_map = build_speaker_profile_map(
        speaker_segments,
        role_name_map=role_name_map,
        default_role=default_role,
    )

    attached: List[Dict[str, Any]] = []
    for seg in transcript_segments:
        result = dict(seg)
        matched = find_matching_segment(
            _midpoint(seg["start_time"], seg["end_time"]),
            speaker_segments,
        )
        if matched is None:
            result["speaker"] = UNKNOWN_SPEAKER
            result["speaker_role"] = UNKNOWN_ROLE
        else:
            profile = profile_map[str(matched["speaker"])]
            result["speaker"] = profile["name"]
            result["speaker_role"] = profile["role"]
        attached.append(result)

    return attached

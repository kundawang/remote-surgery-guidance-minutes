"""CSA 农场会员会转写与说话人分离结果合并 / 角色分类。

设计要点：

* ``merge_transcript_and_diarization`` 对转写段与说话人时间段做多对多归属：
  一条转写段横跨多个说话人时，按重叠区域拆成多条，每条携带该区间归属的说话人
  与对应文本；两个及以上说话人时间段与同一区间重叠（插话）时明确标记为
  "多人重叠"，保留所有涉及的说话人，绝不悄悄丢任何一个人；与说话人时间段完全
  没有重叠的转写段保留原文，说话人标记为 ``None``（未知）。
* 重叠归属的所有取舍都是确定性的：按重叠时长排序，相同时依次比较开始时间、
  结束时间、说话人标识，因此相同输入每次运行结果一致。
* 相同说话人的判断一律基于 ``normalize_speaker``：``None``、空串、``"unknown"``
  （大小写不敏感）视为同一类未知，不再出现 ``None`` 与 ``"unknown"`` 混用。
* ``classify_speaker_roles`` 只处理确实归属到某个说话人且文本非空的片段；
  显式 ``None`` / 空文本片段直接跳过，不产生 ``None`` 键，也不会发起空调用。
"""

from typing import Any, Callable, Dict, List, Optional

UNKNOWN_SPEAKER: Optional[str] = None
"""未知说话人的统一归一化标识（None / 空串 / "unknown" 都归到这里）。"""

OVERLAP_SPEAKER = "多人重叠"
"""多个说话人时间段同时覆盖同一区间时使用的标记。"""

ROLE_UNKNOWN = "未知"
ROLE_FARMER = "农人"
ROLE_CONSUMER = "消费者"

_UNKNOWN_TOKENS = {"", "unknown", "none", "null", "n/a", "na", "未知", "未识别"}


def normalize_speaker(speaker: Any) -> Optional[str]:
    """归一化说话人标识。

    ``None``、空串以及 ``"unknown"`` 等占位值统一归一化为 ``None``；
    其它值去除首尾空白后原样保留。
    """
    if speaker is None:
        return None
    if not isinstance(speaker, str):
        speaker = str(speaker)
    speaker = speaker.strip()
    if speaker.lower() in _UNKNOWN_TOKENS:
        return None
    return speaker


def _as_interval(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """兼容 start/end、start_time/end_time 两种键名。"""
    if not isinstance(item, dict):
        return None
    start = item.get("start", item.get("start_time"))
    end = item.get("end", item.get("end_time"))
    if start is None or end is None:
        return None
    start = float(start)
    end = float(end)
    if end <= start:
        return None
    return {"start": start, "end": end, "item": item}


def _sort_key(segment: Dict[str, Any]) -> tuple:
    """时间段的确定性排序键：开始时间、结束时间、归一化说话人。"""
    speaker = normalize_speaker(segment.get("speaker"))
    return (
        float(segment["start"]),
        float(segment["end"]),
        speaker if speaker is not None else "\uffff",
    )


def _speakers_over(interval: Dict[str, Any],
                   speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """返回与给定区间重叠（重叠时长 > 0）的说话人时间段，确定性排序。

    排序规则（全部用于在重叠时长相同时打破平局）：
    1. 重叠时长降序；
    2. 说话人时间段开始时间升序；
    3. 说话人时间段结束时间升序；
    4. 归一化说话人标识升序。
    """
    start, end = interval["start"], interval["end"]
    overlaps: List[tuple] = []
    for seg in speaker_segments:
        parsed = _as_interval(seg)
        if parsed is None:
            continue
        overlap_start = max(start, parsed["start"])
        overlap_end = min(end, parsed["end"])
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue
        speaker = normalize_speaker(parsed["item"].get("speaker"))
        if speaker is None:
            continue
        overlaps.append((
            -overlap,
            parsed["start"],
            parsed["end"],
            speaker,
            speaker,
        ))
    overlaps.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [{"speaker": row[4], "overlap": -row[0]} for row in overlaps]


def _split_text(text: str, lengths: List[float]) -> List[str]:
    """按各切片时长占比把文本确定性地切成若干段。

    使用最大余数法分配字符，短切片至少保留一个字符（文本足够时），
    拼接后与原文逐字一致，不重复也不丢失。
    """
    total = sum(lengths)
    n_chars = len(text)
    if total <= 0 or n_chars == 0:
        return ["" for _ in lengths]

    raw = [n_chars * length / total for length in lengths]
    counts = [int(value) for value in raw]
    positive = [index for index, length in enumerate(lengths) if length > 0]

    if positive:
        min_one = min(1, n_chars // len(positive)) if n_chars >= len(positive) else 0
        for index in positive:
            counts[index] = max(counts[index], min_one)

    allocated = sum(counts)
    remainder_order = sorted(
        positive,
        key=lambda index: (-(raw[index] - counts[index]), -lengths[index], index),
    )
    remainder = n_chars - allocated
    cursor = 0
    while remainder > 0 and remainder_order:
        index = remainder_order[cursor % len(remainder_order)]
        counts[index] += 1
        remainder -= 1
        cursor += 1
    while remainder < 0:
        candidates = [index for index in remainder_order if counts[index] > 0]
        if not candidates:
            break
        index = max(candidates, key=lambda idx: (raw[idx] - counts[idx], counts[idx]))
        counts[index] -= 1
        remainder += 1

    parts: List[str] = []
    position = 0
    for count in counts:
        parts.append(text[position:position + count])
        position += count
    return parts


def merge_transcript_and_diarization(
    transcript_segments: List[Dict[str, Any]],
    speaker_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """合并转写段与说话人分离时间段（多对多归属）。

    返回列表中每项保持 ``{"start", "end", "text", "speaker"}`` 结构：

    * 转写段与说话人时间段完全不重叠：保留原文与原时间边界，``speaker`` 为 ``None``；
    * 区间仅被一个说话人覆盖：归属到该说话人；
    * 区间被两个及以上说话人覆盖（插话）：标记为
      ``"多人重叠:spk1,spk2"``（说话人按重叠时长降序，平局按标识排序），
      所有涉及的说话人都保留；
    * 一条转写段横跨多个说话人 / 覆盖空洞时按边界拆成多条，拆出的边界落在
      原区间内部且彼此不重叠，拼起来恰好覆盖原区间。
    """
    merged: List[Dict[str, Any]] = []

    ordered_speakers = [
        parsed for seg in (speaker_segments or [])
        if (parsed := _as_interval(seg)) is not None
    ]
    ordered_speakers.sort(key=lambda seg: (seg["start"], seg["end"]))

    for raw_segment in transcript_segments or []:
        parsed = _as_interval(raw_segment)
        if parsed is None:
            continue
        seg_start = parsed["start"]
        seg_end = parsed["end"]
        text = str(raw_segment.get("text", "") or "").strip()

        boundaries = {seg_start, seg_end}
        for speaker_seg in ordered_speakers:
            if speaker_seg["end"] <= seg_start or speaker_seg["start"] >= seg_end:
                continue
            boundaries.add(max(seg_start, speaker_seg["start"]))
            boundaries.add(min(seg_end, speaker_seg["end"]))
        cuts = sorted(boundaries)

        slices = [
            {"start": left, "end": right, "length": right - left}
            for left, right in zip(cuts, cuts[1:])
            if right > left
        ]
        if not slices:
            slices = [{"start": seg_start, "end": seg_end, "length": seg_end - seg_start}]

        texts = _split_text(text, [item["length"] for item in slices])

        for sliced, piece in zip(slices, texts):
            covering = _speakers_over(sliced, speaker_segments or [])
            if not covering:
                speaker: Optional[str] = UNKNOWN_SPEAKER
            else:
                names = []
                seen = set()
                for item in covering:
                    if item["speaker"] not in seen:
                        seen.add(item["speaker"])
                        names.append(item["speaker"])
                if len(names) == 1:
                    speaker = names[0]
                else:
                    speaker = f"{OVERLAP_SPEAKER}:" + ",".join(names)

            merged.append({
                "start": sliced["start"],
                "end": sliced["end"],
                "text": piece,
                "speaker": speaker,
            })

    return merged


def _default_llm_classifier(samples: List[Dict[str, str]]) -> Dict[str, str]:
    """未注入 LLM 客户端时的本地兜底分类（不发起任何外部调用）。"""
    roles: Dict[str, str] = {}
    for sample in samples:
        speaker = sample["speaker"]
        text = sample["text"]
        if any(keyword in text for keyword in ("农场", "种植", "采摘", "菜地", "施肥")):
            roles[speaker] = ROLE_FARMER
        elif any(keyword in text for keyword in ("配送", "订购", "会员", "份额", "价格")):
            roles[speaker] = ROLE_CONSUMER
        else:
            roles[speaker] = ROLE_UNKNOWN
    return roles


def classify_speaker_roles(
    merged_segments: List[Dict[str, Any]],
    llm_classifier: Optional[Callable[[List[Dict[str, str]]], Dict[str, str]]] = None,
) -> Dict[str, str]:
    """把已归属片段的说话人分类为 ``农人`` / ``消费者``。

    只处理确实归属到某个说话人（归一化后非 ``None``）且文本非空的片段；
    显式 ``None``、空串、``"unknown"`` 以及纯 "多人重叠" 片段直接跳过：
    既不会在返回字典里产生 ``None`` 键，也不会拿空文本调用 LLM。

    ``llm_classifier`` 接收 ``[{"speaker", "text"}]`` 样本列表（每个说话人
    一条，按标识排序保证确定性），返回 ``{speaker: role}``；未注入时使用
    本地关键词兜底分类。
    """
    samples_by_speaker: Dict[str, str] = {}
    for segment in merged_segments or []:
        speaker = normalize_speaker(segment.get("speaker", "unknown"))
        if speaker is None:
            continue
        if speaker == OVERLAP_SPEAKER or speaker.startswith(OVERLAP_SPEAKER + ":"):
            continue
        text = str(segment.get("text", "") or "").strip()
        if not text:
            continue
        samples_by_speaker.setdefault(speaker, text)

    if not samples_by_speaker:
        return {}

    samples = [
        {"speaker": speaker, "text": samples_by_speaker[speaker]}
        for speaker in sorted(samples_by_speaker)
    ]

    classifier = llm_classifier or _default_llm_classifier
    raw_roles = classifier(samples) or {}

    roles: Dict[str, str] = {}
    for speaker in samples_by_speaker:
        role = raw_roles.get(speaker)
        roles[speaker] = role if role else ROLE_UNKNOWN
    return roles

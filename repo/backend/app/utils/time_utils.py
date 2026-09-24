from typing import Union


def format_timestamp(value: Union[int, float, str, None]) -> str:
    """统一把时间格式化为 mm:ss，超过 1 小时为 h:mm:ss。

    输入可以是秒数（int/float）、数字字符串，或已格式化的
    "mm:ss"/"h:mm:ss" 字符串；无法识别的输入原样返回。
    """
    if value is None:
        return "00:00"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "00:00"
        if ":" in text:
            return _normalize_colon_text(text)
        try:
            return _format_seconds(float(text))
        except ValueError:
            return text
    if isinstance(value, (int, float)):
        return _format_seconds(value)
    return str(value)


def _format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _normalize_colon_text(text: str) -> str:
    parts = text.split(":")
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return text
    if len(nums) == 2:
        return f"{nums[0]:02d}:{nums[1]:02d}"
    if len(nums) == 3:
        return f"{nums[0]}:{nums[1]:02d}:{nums[2]:02d}"
    return text

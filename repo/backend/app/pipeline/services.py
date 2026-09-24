"""流水线各步骤的外部服务适配层。

所有重依赖（librosa / whisper / openai / smtp）都在函数内部延迟导入：
- 编排模块（workflow.py）保持轻量，luigi 任务图可以独立加载；
- 测试可以用 monkeypatch 整体替换某个步骤，无需安装真实依赖。
"""

from typing import Any, Dict, List, Optional, Sequence


def clean_audio(input_path: str, output_path: str) -> Dict[str, Any]:
    """清洗原始音频，把真实 wav 写到 output_path，返回处理状态信息。"""
    import soundfile as sf

    from ..services.audio_processor import AudioProcessor

    processor = AudioProcessor()
    y, sr = processor.load_audio(input_path)
    y = processor.spectral_subtraction(y, sr)
    y = processor.wiener_filter(y)
    sf.write(output_path, y, sr, format="WAV")
    return {
        "sample_rate": int(sr),
        "duration_seconds": round(float(len(y)) / float(sr), 3),
        "noise_reduction": ["spectral_subtraction", "wiener"],
    }


def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """用 Whisper 转写音频文件，返回结构化转写结果。"""
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="zh")
    return {
        "text": result.get("text", ""),
        "language": result.get("language", "zh"),
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.get("segments", [])
        ],
    }


def generate_summary(transcript: Dict[str, Any], meeting: Dict[str, Any]) -> Dict[str, Any]:
    """基于转写文本生成会议纪要摘要；未配置 OpenAI 时退化为截取原文。"""
    from ..core.config import settings

    text = transcript.get("text", "")
    if settings.OPENAI_API_KEY:
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = (
            f"你是伦理委员会秘书。请为会议《{meeting['meeting_title']}》"
            f"（{meeting['meeting_date']}）的转写内容生成中文纪要摘要：\n{text}"
        )
        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        summary_text = resp.choices[0].message.content or ""
    else:
        summary_text = text[:500]
    return {
        "meeting_title": meeting["meeting_title"],
        "meeting_date": meeting["meeting_date"],
        "summary_text": summary_text,
        "key_points": [s["text"] for s in transcript.get("segments", [])[:5]],
    }


def render_review_email(meeting: Dict[str, Any], summary: Dict[str, Any]) -> str:
    """渲染复审邮件 HTML，会议标题、委员会邮箱等均取自传入的会议参数。"""
    key_points = "".join(f"<li>{p}</li>" for p in summary.get("key_points", []))
    reviewers = ", ".join(meeting.get("reviewer_emails", [])) or "（无）"
    return f"""<html lang="zh-CN">
<body>
  <h1>伦理复审：{meeting['meeting_title']}</h1>
  <p>会议日期：{meeting['meeting_date']}</p>
  <p>送审委员会：{meeting['ethics_committee_email']}</p>
  <p>复审人：{reviewers}</p>
  <h2>纪要摘要</h2>
  <p>{summary.get('summary_text', '')}</p>
  <h2>要点</h2>
  <ul>{key_points}</ul>
</body>
</html>"""


def send_email(
    subject: str,
    html_body: str,
    to: Sequence[str],
    cc: Optional[Sequence[str]] = None,
) -> None:
    """通过 SMTP 发送复审邮件。"""
    import smtplib
    from email.mime.text import MIMEText

    from ..core.config import settings

    cc = list(cc or [])
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM_EMAIL
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        if settings.SMTP_USERNAME:
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD or "")
        server.sendmail(settings.SMTP_FROM_EMAIL, list(to) + cc, msg.as_string())

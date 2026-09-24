# 基因伦理研讨会纪要流水线（luigi）

任务链：`CleanAudioTask -> TranscribeAudioTask -> GenerateSummaryTask -> SendReviewEmailTask`

## 运行

Python 入口：

```python
from app.pipeline.workflow import run_workflow

run_workflow(
    meeting_id="ethics-2026-0924",
    meeting_title="CRISPR 生殖系编辑伦理研讨会",
    meeting_date="2026-09-24",
    input_audio_path="/data/raw.wav",
    output_dir="/data/out",
    ethics_committee_email="ethics-board@example.org",
    reviewer_emails=["alice@example.org", "bob@example.org"],
)
```

命令行入口（与 `run_workflow` 行为一致）：

```bash
python -m app.pipeline.workflow \
  --meeting-id ethics-2026-0924 \
  --meeting-title "CRISPR 生殖系编辑伦理研讨会" \
  --meeting-date 2026-09-24 \
  --input-audio-path /data/raw.wav \
  --output-dir /data/out \
  --ethics-committee-email ethics-board@example.org \
  --reviewer-emails alice@example.org,bob@example.org
```

## 产物（`<output_dir>/<meeting_id>_*`，各步骤产物互不覆盖）

| 文件 | 说明 |
| --- | --- |
| `*_cleaned_audio.wav` | 清洗后的音频 |
| `*_cleaned_audio_status.json` | 清洗步骤的状态（独立文件，不写入音频路径） |
| `*_transcript.json` | Whisper 转写结果 |
| `*_summary.json` | 纪要摘要 |
| `*_review_email.json` | 复审邮件发送记录（归档） |

重跑幂等：任务是否执行由 output 目标是否存在决定，归档已存在时不会重复发邮件。

## 测试

```bash
cd backend && python -m pytest tests/test_pipeline_workflow.py
```

import glob
import os
import tempfile

from app.api.summary import summary_generator
from app.api.transcript import transcription_service
from app.models.schemas import SurgerySummaryResponse, TranscriptSegment


def _fake_segments(self, audio_path):
    return [
        TranscriptSegment(
            speaker="张医生",
            speaker_role="主刀医生",
            start_time=0.0,
            end_time=5.0,
            text="开始分离胆囊三角，注意避免损伤胆总管。",
            confidence=0.95,
        )
    ]


def _fake_summary(self, session_id, transcripts, session_info):
    return SurgerySummaryResponse(
        key_points=["要点一"],
        surgical_steps=[{"time": 0.0, "step": "分离", "description": "分离胆囊三角"}],
        anatomical_landmarks=["胆囊三角"],
        technical_improvements=["建议轻柔操作"],
        complications=["出血风险"],
        overall_assessment="手术过程顺利。",
    )


def test_background_transcribe_saves_with_own_session(client, session_id, monkeypatch):
    """后台任务使用自己开的 session 写库，响应返回后数据真实落库。"""
    monkeypatch.setattr(type(transcription_service), "transcribe_segment", _fake_segments)

    response = client.post(
        f"/api/transcript/transcribe/{session_id}",
        params={"audio_file_path": "/tmp/fake.wav"},
    )
    assert response.status_code == 200

    transcripts = client.get(f"/api/transcript/{session_id}")
    assert transcripts.status_code == 200
    data = transcripts.json()
    assert len(data) == 1
    assert data[0]["text"] == "开始分离胆囊三角，注意避免损伤胆总管。"


def test_summary_failure_is_queryable_and_retryable(client, session_id, monkeypatch):
    """生成失败：接口返回错误、状态可查（failed + 原因）；修复后重试可成功。"""
    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom: database is locked")

    monkeypatch.setattr(type(summary_generator), "generate_summary", _boom)

    response = client.post(f"/api/summary/generate/{session_id}")
    assert response.status_code == 500

    detail = client.get(f"/api/surgery/{session_id}").json()
    assert detail["summary_status"] == "failed"
    assert "boom" in detail["summary_error"]

    listed = client.get("/api/surgery").json()
    latest = next(s for s in listed if s["session_id"] == session_id)
    assert latest["summary_status"] == "failed"
    assert "boom" in latest["summary_error"]

    assert client.get(f"/api/summary/{session_id}").status_code == 404

    # 重复点生成可以再次触发并成功
    monkeypatch.setattr(type(summary_generator), "generate_summary", _fake_summary)
    retry = client.post(f"/api/summary/generate/{session_id}")
    assert retry.status_code == 200

    detail = client.get(f"/api/surgery/{session_id}").json()
    assert detail["summary_status"] == "completed"
    assert detail["summary_error"] is None

    saved = client.get(f"/api/summary/{session_id}")
    assert saved.status_code == 200
    assert saved.json()["key_points"] == ["要点一"]


def test_download_summary_leaves_no_temp_files(client, session_id, monkeypatch):
    """下载接口流式返回，不在临时目录残留 summary_*.md 文件。"""
    monkeypatch.setattr(type(summary_generator), "generate_summary", _fake_summary)
    assert client.post(f"/api/summary/generate/{session_id}").status_code == 200

    pattern = os.path.join(tempfile.gettempdir(), "summary_*")
    before = set(glob.glob(pattern))

    response = client.get(f"/api/summary/{session_id}/download")
    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert f'filename="summary_{session_id}_' in disposition
    assert disposition.endswith('.md"')
    assert "手术纪要" in response.text

    after = set(glob.glob(pattern))
    assert after == before


def test_download_summary_not_found(client, session_id):
    assert client.get(f"/api/summary/{session_id}/download").status_code == 404

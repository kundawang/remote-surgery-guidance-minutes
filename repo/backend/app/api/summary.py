import logging

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from ..core.database import get_db, SessionLocal
from ..services.summary_generator import SummaryGenerator
from ..models.schemas import SurgerySummaryRequest, SurgerySummaryResponse, TranscriptSegment
from ..core.database import SurgerySession, SurgerySummary, Transcript

logger = logging.getLogger(__name__)

router = APIRouter()

summary_generator = SummaryGenerator()


def _update_summary_status(session_pk: int, status: str, error: str = None):
    """在独立的数据库 session 中更新摘要生成状态，任务结束后确保关闭。"""
    db = SessionLocal()
    try:
        db_session = db.query(SurgerySession).filter(SurgerySession.id == session_pk).first()
        if db_session:
            db_session.summary_status = status
            db_session.summary_error = error
            db.commit()
    except Exception:
        logger.exception("更新摘要生成状态失败: session_pk=%s status=%s", session_pk, status)
        db.rollback()
    finally:
        db.close()


@router.post("/generate/{session_id}", response_model=SurgerySummaryResponse)
async def generate_summary(
    session_id: str,
    db: Session = Depends(get_db)
):
    session = db.query(SurgerySession).filter(SurgerySession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="手术会话未找到")

    session.summary_status = "processing"
    session.summary_error = None
    db.commit()

    try:
        transcripts = db.query(Transcript).filter(
            Transcript.session_id == session.id
        ).order_by(Transcript.start_time).all()

        transcript_segments = [
            TranscriptSegment(
                speaker=t.speaker,
                speaker_role=t.speaker_role,
                start_time=t.start_time,
                end_time=t.end_time,
                text=t.text,
                confidence=t.confidence,
                is_anatomical_term=t.is_anatomical_term,
                anatomical_terms=t.anatomical_terms,
                is_surgery_step=t.is_surgery_step,
                surgery_step=t.surgery_step
            )
            for t in transcripts
        ]

        summary = summary_generator.generate_summary(
            session_id=session_id,
            transcripts=transcript_segments,
            session_info={
                "patient_name": session.patient_name,
                "surgery_type": session.surgery_type,
                "primary_surgeon": session.primary_surgeon,
                "remote_expert": session.remote_expert,
                "operating_room": session.operating_room,
                "start_time": session.start_time.isoformat() if session.start_time else None,
                "end_time": session.end_time.isoformat() if session.end_time else None
            }
        )

        existing_summary = db.query(SurgerySummary).filter(SurgerySummary.session_id == session.id).first()
        if existing_summary:
            existing_summary.key_points = summary.key_points
            existing_summary.surgical_steps = summary.surgical_steps
            existing_summary.anatomical_landmarks = summary.anatomical_landmarks
            existing_summary.technical_improvements = summary.technical_improvements
            existing_summary.complications = summary.complications
            existing_summary.overall_assessment = summary.overall_assessment
        else:
            db_summary = SurgerySummary(
                session_id=session.id,
                key_points=summary.key_points,
                surgical_steps=summary.surgical_steps,
                anatomical_landmarks=summary.anatomical_landmarks,
                technical_improvements=summary.technical_improvements,
                complications=summary.complications,
                overall_assessment=summary.overall_assessment
            )
            db.add(db_summary)

        session.summary_status = "completed"
        session.summary_error = None
        db.commit()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("生成手术摘要失败: session_id=%s", session_id)
        db.rollback()
        _update_summary_status(session.id, "failed", str(e))
        raise HTTPException(status_code=500, detail=f"手术摘要生成失败: {e}")

    return summary


@router.get("/{session_id}", response_model=SurgerySummaryResponse)
def get_summary(
    session_id: str,
    db: Session = Depends(get_db)
):
    session = db.query(SurgerySession).filter(SurgerySession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="手术会话未找到")
    
    summary = db.query(SurgerySummary).filter(SurgerySummary.session_id == session.id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="手术摘要未找到")
    
    return SurgerySummaryResponse(
        key_points=summary.key_points,
        surgical_steps=summary.surgical_steps,
        anatomical_landmarks=summary.anatomical_landmarks,
        technical_improvements=summary.technical_improvements,
        complications=summary.complications,
        overall_assessment=summary.overall_assessment
    )


@router.post("/regenerate/{session_id}", response_model=SurgerySummaryResponse)
async def regenerate_summary(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    session = db.query(SurgerySession).filter(SurgerySession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="手术会话未找到")
    
    return await generate_summary(session_id, db)


def _render_summary_markdown(session: SurgerySession, summary: SurgerySummary) -> str:
    lines = [
        f"# 手术纪要 - {session.session_id}",
        "",
        "## 手术信息",
        "",
        f"- 患者姓名: {session.patient_name}",
        f"- 手术类型: {session.surgery_type}",
        f"- 主刀医生: {session.primary_surgeon}",
        f"- 远程指导专家: {session.remote_expert}",
        f"- 手术室: {session.operating_room}",
        "",
        "## 手术要点",
        "",
    ]
    for point in summary.key_points or []:
        lines.append(f"- {point}")
    lines += ["", "## 手术步骤", ""]
    for step in summary.surgical_steps or []:
        if isinstance(step, dict):
            lines.append(f"- [{step.get('time', '')}] {step.get('step', '')}: {step.get('description', '')}")
        else:
            lines.append(f"- {step}")
    lines += ["", "## 解剖标识", ""]
    for landmark in summary.anatomical_landmarks or []:
        lines.append(f"- {landmark}")
    lines += ["", "## 技术改进建议", ""]
    for item in summary.technical_improvements or []:
        lines.append(f"- {item}")
    lines += ["", "## 并发症风险", ""]
    for item in summary.complications or []:
        lines.append(f"- {item}")
    lines += ["", "## 总体评估", "", summary.overall_assessment or "", ""]
    return "\n".join(lines)


@router.get("/{session_id}/download")
def download_summary(
    session_id: str,
    db: Session = Depends(get_db)
):
    session = db.query(SurgerySession).filter(SurgerySession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="手术会话未找到")

    summary = db.query(SurgerySummary).filter(SurgerySummary.session_id == session.id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="手术摘要未找到")

    content = _render_summary_markdown(session, summary)
    filename = f"summary_{session.session_id}_{summary.id}.md"

    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

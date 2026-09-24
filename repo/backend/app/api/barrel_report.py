from fastapi import APIRouter

from ..models.schemas import (
    BarrelReportEmailRequest,
    BarrelReportEmailResponse,
    BarrelTastingRequest,
    BarrelTastingReportResponse,
)
from ..services.barrel_report_generator import BarrelReportGenerator
from ..services.barrel_report_mailer import BarrelReportMailer

router = APIRouter()

report_generator = BarrelReportGenerator()
report_mailer = BarrelReportMailer()


@router.post("/generate", response_model=BarrelTastingReportResponse)
async def generate_report(request: BarrelTastingRequest):
    return report_generator.generate_report(request)


@router.post("/email", response_model=BarrelReportEmailResponse)
async def email_report(request: BarrelReportEmailRequest):
    result = report_mailer.send_report(
        report=request.report,
        recipient_email=request.recipient_email,
    )
    return BarrelReportEmailResponse(**result)

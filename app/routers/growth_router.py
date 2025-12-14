from fastapi import APIRouter, Depends, Cookie, HTTPException, status
from sqlalchemy.orm import Session
from app.schemas.growth_schema import ReportSchema
from app.routers.dependencies import get_db
from app.services.growth_service import get_full_report 
from app.utils.jwt_provider import verify_access_token

router = APIRouter(prefix="/growth", tags=["성장 기록"])

@router.get("/report", response_model=ReportSchema, summary="성장 기록 불러오기")
def getReport(
    access_token: str | None = Cookie(default=None),
    session: Session = Depends(get_db)
):
    """
    현재 유저의 1년치 성장 기록과 마지막 달 월간 레포트를 함께 반환
    """

    user_id = None

    if access_token:
        payload = verify_access_token(access_token)  
        if payload:
            user_id = int(payload.get("sub")) 

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
        )

    report_data = get_full_report(user_id=user_id, session=session)
    return report_data
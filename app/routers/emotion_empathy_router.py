from fastapi import APIRouter, Form, File, Request, Depends, Cookie
from app.routers.dependencies import get_db as get_session

from sqlmodel import Session
# from app.core.db import get_session

from app.schemas.emotion_empathy_schema import (
    SelectedEmotionQuery,
    SituationCreateResponse,
    EmpathyEvaluateRequest,
    EmpathyEvaluateResponse,
)
from app.services.emotion_empathy_service import (
    create_empathy_scenario_service,
    evaluate_empathy_message_service,
)

from app.routers.dependencies import SessionDep, CurrentUserDep
from app.models.empathy_training_result import *

router = APIRouter(
    prefix="/empathy",
    tags=["Empathy"]
)


# -------------------------------------------------------
# ⭐ 1) 공감 시나리오 생성 API
# -------------------------------------------------------
@router.get("/scenario", response_model=SituationCreateResponse)
async def create_empathy_scenario(
    query: SelectedEmotionQuery = Depends(),
    access_token: str | None = Cookie(default=None),   
):
    """
    감정 선택 기반 공감 상황 생성 API
    """

    result = await create_empathy_scenario_service(
        query=query,
        token=access_token,
    )

    # 서비스는 emotion과 scenario만 반환하므로 바로 매핑
    return SituationCreateResponse(
        emotion=result["emotion"],
        scenario=result["scenario"],
    )


# -------------------------------------------------------
# ⭐ 2) 공감 메시지 평가 API
# -------------------------------------------------------
@router.post("/submit", response_model=EmpathyEvaluateResponse)
async def evaluate_empathy_message(
    body: EmpathyEvaluateRequest,                          # ✔ 바디 받기
    access_token: str | None = Cookie(default=None),       # ✔ 쿠키에서 액세스토큰 받기
    session: Session = Depends(get_session),               # ✔ DB 세션 주입
):
    """
    사용자의 공감 메시지를 AI로 평가하는 API + user별로 기록을 DB에 저장
    """

    # 서비스 호출 (body + access_token + session 전달)
    result = await evaluate_empathy_message_service(
        body=body,
        token=access_token,
        session=session
    )

    # 서비스에서 score + feedback 반환됨
    return EmpathyEvaluateResponse(
        score=result["score"],
        feedback=result["feedback"]
    )

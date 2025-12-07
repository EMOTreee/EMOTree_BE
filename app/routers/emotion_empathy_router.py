from fastapi import APIRouter, Depends, Header

#from sqlmodel import Session
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
):
    """
    감정 선택 기반 공감 상황 생성 API
    """

    result = await create_empathy_scenario_service(query=query)

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
    body: EmpathyEvaluateRequest,
    session: SessionDep,
    user: CurrentUserDep, # 로그인한 User 자동 주입
):
    """
    사용자의 공감 메시지를 AI로 평가하는 API + user별로 기록을 DB에 저장
    """

    result = await evaluate_empathy_message_service(
        body=body,
    )

    score = result["score"]
    feedback = result["feedback"]

    # 2) DB 저장 -> 이게 맞나 몰겠다..
    history = EmpathyTrainingResult(
        user_id=user.id,
        emotion_label=body.emotion,       # EmotionLabel enum
        scenario_text=body.scenario,      # 시나리오 텍스트
        user_reply=body.userMessage,      # 사용자가 작성한 메시지
        empathy_score=score,              # 점수
        feedback=feedback                 # 피드백
    )

    session.add(history)
    session.commit()

    #응답
    return EmpathyEvaluateResponse(
        score=result["score"],
        feedback=result["feedback"]
    )

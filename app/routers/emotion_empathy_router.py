from fastapi import APIRouter, Depends, Header

# from sqlmodel import Session
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
):
    """
    사용자의 공감 메시지를 AI로 평가하는 API
    """

    result = await evaluate_empathy_message_service(
        body=body,
    )

    return EmpathyEvaluateResponse(
        score=result["score"],
        feedback=result["feedback"]
    )

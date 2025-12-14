from fastapi import APIRouter, HTTPException, Depends, Cookie
from sqlmodel import Session
from openai import OpenAI

from app.schemas.emotion_quiz_schema import (
    QuizGenerateResponse,
    QuizSubmitRequest,
    QuizSubmitResponse,
    EMOTION_CHOICES,
)
from app.services.emotion_quiz_service import (
    generate_question,
    submit_emotion_quiz_service,
)
from app.core.config import settings
from app.routers.dependencies import get_db as get_session


router = APIRouter(prefix="/quiz", tags=["퀴즈"])


# 선택: DALLE 사용 여부를 쿼리로 제어. 기본은 STATIC
@router.get("/generate", response_model=QuizGenerateResponse, summary="퀴즈 생성")
def generate(use_dalle: bool = False):
    # DALLE 쓰려면 openai client 준비
    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        organization=settings.OPENAI_ORGANIZATION_ID,
    )

    source = "DALLE" if use_dalle else "STATIC"
    qid, image_url, summary = generate_question(source, client)

    return QuizGenerateResponse(
        quizId=qid,
        quizImageUrl=image_url,
        summary=summary,
    )


@router.post("/submit", response_model=QuizSubmitResponse, summary="정답 제출")
async def submit(
    payload: QuizSubmitRequest,
    access_token: str | None = Cookie(default=None),
    session: Session = Depends(get_session),
):
    """
    - 캐시에 있는 퀴즈 채점
    - access_token이 유효하면 EmotionQuizResult에 저장
    - 만료된 퀴즈는 410 에러
    """
    try:
        result_dict = await submit_emotion_quiz_service(
            quiz_id=payload.quizId,
            user_answer=payload.userAnswer,
            token=access_token,
            session=session,
        )
    except ValueError:
        # 캐시에서 사라진 / 만료된 퀴즈
        raise HTTPException(
            status_code=410,
            detail="문항이 만료되었거나 존재하지 않습니다.",
        )

    # 서비스에서 dict로 주는 값을 스키마로 감싸서 리턴
    return QuizSubmitResponse(**result_dict)

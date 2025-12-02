from fastapi import APIRouter, HTTPException, Depends
from app.schemas.emotion_quiz_schema import (
    QuizGenerateResponse, QuizSubmitRequest, QuizSubmitResponse, EMOTION_CHOICES
)
from app.services.emotion_quiz_service import generate_question, grade
from app.core.config import settings
from openai import OpenAI

router = APIRouter(prefix="/quiz", tags=["퀴즈"])

# 선택: DALLE 사용 여부를 쿼리로 제어. 기본은 STATIC
@router.get("/generate", response_model=QuizGenerateResponse, summary="퀴즈 생성")
def generate(use_dalle: bool = False):
    # DALLE 쓰려면 openai client 준비    
    client = OpenAI(api_key=settings.OPENAI_API_KEY, organization=settings.OPENAI_ORGANIZATION_ID)

    source = "DALLE" if use_dalle else "STATIC"
    qid, image_url, summary = generate_question(source, client)
    return QuizGenerateResponse(
        questionId=qid,
        imageUrl=image_url,
        summary=summary
    )

@router.post("/submit", response_model=QuizSubmitResponse, summary="정답 제출")
def submit(payload: QuizSubmitRequest):
    result = grade(payload.questionId, payload.userAnswer)
    if result is None:
        raise HTTPException(status_code=410, detail="문항이 만료되었거나 존재하지 않습니다.")
    is_correct, correct, feedback = result
    return QuizSubmitResponse(
        isCorrect=is_correct,
        correctEmotion=correct,
        feedback=feedback
    )

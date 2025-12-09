from fastapi import APIRouter, UploadFile, Form, File, Depends, Cookie
from fastapi.responses import JSONResponse
from typing import Literal

from app.routers.dependencies import CurrentUserDep, SessionDep
from app.services.emotion_voice_service import analyze_voice_emotion_service
from app.schemas.emotion_voice_schema import VoiceEmotionResponse


router = APIRouter(
    prefix="/emotion-expression",
    tags=["emotion-expression"]
)

@router.post("/voice", response_model=VoiceEmotionResponse)
async def analyze_voice_emotion(
    current_user: CurrentUserDep,
    session: SessionDep,
    file: UploadFile = File(...),
    targetEmotion: Literal["JOY", "SADNESS", "ANGER", "SURPRISE", "ANXIETY"] = Form(...),
    resetFlag: bool = Form(False),
):
    # 음성 파일 읽기
    audio_bytes = await file.read()
    
    # 음성 감정 분석 서비스 호출
    result = await analyze_voice_emotion_service(
        audio_bytes=audio_bytes,
        target_emotion=targetEmotion,
        user_id=current_user.id,
        session=session,
        reset_flag=resetFlag,
    )
    
    return result

from sqlmodel import Session

from app.services.voice_emotion_chains import voice_emotion_pipeline
from app.schemas.emotion_voice_schema import VoiceEmotionResponse
from app.models.emotion_expression_result import EmotionExpressionResult
from app.models.enums import EmotionLabel
from app.crud.emotion_expression import create_emotion_expression_result


async def analyze_voice_emotion_service(
    *,
    audio_bytes: bytes,
    target_emotion: str,
    user_id: int,
    session: Session,
    reset_flag: bool = False
) -> VoiceEmotionResponse:
    
    # 파이프라인 실행 (user_id와 reset_flag 전달)
    result = voice_emotion_pipeline(audio_bytes, target_emotion, user_id, reset_flag)
    
    # EmotionLabel enum으로 변환
    target_emotion_enum = EmotionLabel[result["targetEmotion"]]
    detected_emotion_enum = EmotionLabel[result["detectedEmotion"]]
    
    # DB에 결과 저장
    emotion_result = EmotionExpressionResult(
        user_id=user_id,
        target_emotion=target_emotion_enum,
        detected_emotion=detected_emotion_enum,
        expression_score=result["score"],
        feedback=result["feedback"]
    )
    
    create_emotion_expression_result(session=session, emotion_result=emotion_result)
    
    # 응답 스키마 생성
    response = VoiceEmotionResponse(
        targetEmotion=target_emotion_enum,
        detectedEmotion=detected_emotion_enum,
        score=result["score"],
        feedback=result["feedback"],
        isCorrect=result["isCorrect"]
    )
    
    return response

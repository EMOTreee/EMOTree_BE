from pydantic import BaseModel, Field
from app.models.enums import EmotionLabel


# 음성 감정 표현 평가 결과 (Response)
class VoiceEmotionResponse(BaseModel):
    targetEmotion: EmotionLabel
    detectedEmotion: EmotionLabel
    score: int = Field(ge=0, le=100)
    feedback: str
    isCorrect: bool

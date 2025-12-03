from pydantic import BaseModel, Field
from app.models.enums import EmotionLabel

EMOTION_CHOICES = [
    EmotionLabel.JOY, EmotionLabel.ANGER, EmotionLabel.SADNESS,
    EmotionLabel.SURPRISE, EmotionLabel.ANXIETY
]

class QuizGenerateResponse(BaseModel):
    quizId: str = Field(..., description="제출 시 사용할 ID")
    quizImageUrl: str = Field(..., description="객관식 퀴즈 이미지")
    summary: str | None = Field(None, description="이미지 설명(선택)")

class QuizSubmitRequest(BaseModel):
    quizId: str
    userAnswer: EmotionLabel

class QuizSubmitResponse(BaseModel):
    isCorrect: bool
    correctEmotion: EmotionLabel
    feedback: str | None = None

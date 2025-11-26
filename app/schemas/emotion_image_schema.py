from pydantic import BaseModel

class EmotionCheckResponse(BaseModel):
    targetEmotion: str
    detectedEmotion: str
    score: int
    feedback: str

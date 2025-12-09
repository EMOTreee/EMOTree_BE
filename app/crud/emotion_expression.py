from sqlmodel import Session
from app.models.emotion_expression_result import EmotionExpressionResult


def create_emotion_expression_result(
    *,
    session: Session,
    emotion_result: EmotionExpressionResult
) -> EmotionExpressionResult:
    session.add(emotion_result)
    session.commit()
    session.refresh(emotion_result)
    return emotion_result

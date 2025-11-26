from fastapi import APIRouter, UploadFile, Form, File
from fastapi.responses import JSONResponse

from app.schemas.emotion_image_schema import EmotionCheckResponse
from app.services.emotion_image_service import analyze_emotion_service

router = APIRouter(
    prefix="/emotion-expression",
    tags=["emotion-expression"]
)

@router.post("/image", response_model=EmotionCheckResponse)
async def emotion_image_check(
    file: UploadFile = File(...),
    targetEmotion: str = Form(...)
):
    image_bytes = await file.read()

    result = await analyze_emotion_service(image_bytes, targetEmotion)

    return result

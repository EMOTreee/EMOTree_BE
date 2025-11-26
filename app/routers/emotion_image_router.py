from fastapi import APIRouter, UploadFile, Form, File, Request, Depends, Cookie
from fastapi.responses import JSONResponse

from sqlmodel import Session
from app.routers.dependencies import get_db


from app.schemas.emotion_image_schema import EmotionCheckResponse
from app.services.emotion_image_service import analyze_emotion_service

router = APIRouter(
    prefix="/emotion-expression",
    tags=["emotion-expression"]
)

@router.post("/image", response_model=EmotionCheckResponse)
async def emotion_image_check(
    access_token: str | None = Cookie(default=None),
    file: UploadFile = File(...),
    targetEmotion: str = Form(...)
):
    image_bytes = await file.read()

    result = await analyze_emotion_service(
        image_bytes=image_bytes,
        target_emotion=targetEmotion, 
        token=access_token,           
    )

    return result

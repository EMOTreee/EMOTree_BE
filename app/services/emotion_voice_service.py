from sqlmodel import Session
from io import BytesIO
import subprocess
import tempfile
import os
from fastapi import HTTPException

from app.services.voice_emotion_chains import voice_emotion_pipeline
from app.schemas.emotion_voice_schema import VoiceEmotionResponse
from app.models.emotion_expression_result import EmotionExpressionResult
from app.models.enums import EmotionLabel
from app.crud.emotion_expression import create_emotion_expression_result

# FFmpeg 경로 설정
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"


def convert_webm_to_wav(audio_bytes: bytes) -> bytes:
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
            webm_file.write(audio_bytes)
            webm_path = webm_file.name
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
        
        try:
            # FFmpeg 명령어 실행
            command = [
                FFMPEG_PATH,
                "-i", webm_path,          # 입력 파일
                "-ar", "16000",           # 샘플링 레이트 16kHz
                "-ac", "1",               # 모노
                "-y",                     # 덮어쓰기
                wav_path                  # 출력 파일
            ]
            
            # FFmpeg 실행
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"[ERROR] FFmpeg 변환 실패: {result.stderr.decode()}")
                return audio_bytes
            
            # 변환된 WAV 파일 읽기
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
            
            print("[INFO] WEBM → WAV 변환 완료")
            return wav_bytes
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(webm_path):
                os.unlink(webm_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        
    except Exception as e:
        print(f"[ERROR] WEBM → WAV 변환 실패: {e}")
        return audio_bytes


async def analyze_voice_emotion_service(
    *,
    audio_bytes: bytes,
    target_emotion: str,
    user_id: int | None,
    session: Session,
    reset_flag: bool = False
) -> VoiceEmotionResponse:
    
    try:
        # WEBM을 WAV로 변환
        wav_audio_bytes = convert_webm_to_wav(audio_bytes)
        
        # 파이프라인 실행 (user_id와 reset_flag 전달)
        # 비로그인 사용자는 user_id=0으로 메모리 관리
        pipeline_user_id = user_id if user_id else 0
        result = voice_emotion_pipeline(wav_audio_bytes, target_emotion, pipeline_user_id, reset_flag)
        
        # EmotionLabel enum으로 변환
        target_emotion_enum = EmotionLabel[result["targetEmotion"]]
        detected_emotion_enum = EmotionLabel[result["detectedEmotion"]]
        
        # 로그인한 경우에만 DB에 결과 저장
        if user_id:
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
    
    except Exception as e:
        print(f"[ERROR] 음성 감정 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        # DB 저장하지 않고 에러 응답
        raise HTTPException(
            status_code=500,
            detail="음성 감정 분석 중 오류가 발생했습니다"
        )

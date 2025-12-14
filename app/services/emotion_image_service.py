import base64
import json
from fastapi import Request
from sqlmodel import Session
from openai import OpenAI
import os
from dotenv import load_dotenv
from app.models.emotion_expression_result import EmotionExpressionResult
from app.models.enums import EmotionLabel
from app.utils.jwt_provider import verify_access_token
load_dotenv()


async def analyze_emotion_service(    
    *,
    image_bytes: bytes,
    target_emotion: str,
    token: str | None,
    session: Session):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    # targetEmotion 대문자로 통일
    target_emotion = target_emotion.upper()

    # Base64 변환
    base64_image = base64.b64encode(image_bytes).decode()

    # ----------------------------
    # 🔥 최종 개선된 프롬프트
    # ----------------------------

    system_prompt = """
    당신은 감정 인식과 감정 표현을 어려워하는 사람들을 돕는 감정 표현 코치입니다.

    업로드된 얼굴 표정을 분석한 뒤 반드시 아래 JSON 형식으로만 출력하세요:

    {
    "detectedEmotion": "JOY" | "SADNESS" | "ANGER" | "SURPRISE" | "ANXIETY",
    "feedback": "한국어로 된 상세 피드백"
    }

    규칙:
    1. detectedEmotion은 위 5개 중 하나만 선택.
    2. 감정은 한국어로 표현할 것 (기쁨, 슬픔, 분노, 놀람, 불안).
    3. feedback은 반드시 한국어만 사용.
    4. 얼굴 특징(눈, 입, 눈썹 등)을 근거로 감정을 분석할 것.
    5. detectedEmotion과 targetEmotion이 같을 경우:
    - 잘 표현된 요소를 칭찬
    - 더 자연스럽거나 강하게 표현하는 방법 제안
    6. detectedEmotion과 targetEmotion이 다를 경우:
    - 왜 다른 감정으로 보였는지 설명
    - targetEmotion에 가까워지기 위한 구체적 조언
    7. 전체 톤은 친절하고 코칭하듯 할 것.
    8. JSON 이외의 출력 금지.
    9. 코드블록 사용 금지.
    10. '사진'이라는 단어 사용 금지.
    """

    user_prompt = f"""
    사용자가 표현하려고 하는 목표 감정(targetEmotion)은 다음과 같습니다:
    - targetEmotion: "{target_emotion}"
    """

    # ----------------------------
    # 🔥 OpenAI Vision 호출
    # ----------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # 이미지 분석 가능한 최신 소형 모델
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )

    # ----------------------------
    # 🔥 GPT 응답 파싱
    # ----------------------------
    result_text = response.choices[0].message.content

    try:
        gpt_json = json.loads(result_text)
    except Exception:
        raise ValueError(f"GPT JSON 파싱 실패: {result_text}")

    detected = gpt_json["detectedEmotion"]
    feedback = gpt_json["feedback"]

    # 점수 계산
    score = 100 if detected == target_emotion else 40

    #  access_token → user_id 파싱
    # -------------------------------------------------------
    user_id = None

    if token:
        payload = verify_access_token(token)  
        if payload:
            user_id = int(payload.get("sub"))       

    # 🔥 user_id 있으면 DB 저장

    if user_id:
        record = EmotionExpressionResult(
            user_id=user_id,
            target_emotion=EmotionLabel[target_emotion],
            detected_emotion=EmotionLabel[detected],
            expression_score=score,
            feedback=feedback
        )
        session.add(record)
        session.commit()
        session.refresh(record)


    # 최종 반환

    return {
        "targetEmotion": target_emotion,
        "detectedEmotion": detected,
        "score": score,
        "feedback": feedback
    }

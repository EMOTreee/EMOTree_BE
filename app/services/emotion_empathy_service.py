import json
import random
import os
from sqlmodel import Session
from fastapi import Request
from sqlmodel import Session
from openai import OpenAI
from dotenv import load_dotenv

from app.models.enums import EmotionLabel
from app.schemas.emotion_empathy_schema import (
    SelectedEmotionQuery,
    EmpathyEvaluateRequest,
)
from app.utils.jwt_provider import verify_access_token
from app.models.empathy_training_result import EmpathyTrainingResult

#서비스

load_dotenv()


# -------------------------------------------------------
# ⭐ 1) 공감 시나리오 생성 서비스
# -------------------------------------------------------
async def create_empathy_scenario_service(
    *,
    query: SelectedEmotionQuery,
):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Emotion이 RANDOM이면 랜덤 선택
    if query.option == EmotionLabel.RANDOM:
        emotions = [e for e in EmotionLabel if e != EmotionLabel.RANDOM]
        chosen_emotion = random.choice(emotions)
    else:
        chosen_emotion = query.option

    # -----------------------------
    # 🔥 Prompt 설계
    # -----------------------------
    prompt = f"""
    당신은 공감 능력을 기르는 연습을 돕는 "상황 생성 전문가"입니다.

    아래 감정에 해당하는, 사용자가 공감 연습에 사용할 짧고 구체적인 상황 설명을 만들어주세요.

    감정:
    - {chosen_emotion.name}

    출력 형식(JSON):
    {{
        "emotion": "JOY" | "SAD" | "ANGER" | "LOVE" | "FEAR",
        "scenario": "공감이 필요한 채팅 텍스트"
    }}

    규칙:
    1. 감정을 직접적으로 언급하지 말고 메시지 내용으로 감정이 드러나게 표현할 것.
    2. 현실적이고 공감 가능한 카톡/메신저 스타일 대화만 작성할 것.
    3. 모든 메시지는 한국어로.
    4. JSON만 출력. 코드 블록 금지.
    """

    # -----------------------------
    # 🔥 OpenAI 호출
    # -----------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # GPT 응답 텍스트
    result_text = response.choices[0].message.content

    try:
        gpt_json = json.loads(result_text)
    except Exception:
        raise ValueError(f"GPT JSON 파싱 실패: {result_text}")

    scenario_text = gpt_json["scenario"]

    return {
    "emotion": chosen_emotion.name,
    "scenario": scenario_text
}

# -------------------------------------------------------
# ⭐ 2) 공감 메시지 평가 서비스
# -------------------------------------------------------
async def evaluate_empathy_message_service(
    *,
    body: EmpathyEvaluateRequest,
    token: str | None,
    session: Session
):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    emotion = body.emotion # 선택한 감정도 저장해야하기때문에 추가
    scenario = body.scenario
    user_message = body.userMessage

    # -----------------------------
    # 🔥 Prompt 설계
    # -----------------------------
    prompt = f"""
    당신은 "공감 능력 코칭 전문가"입니다.

    아래 시나리오 상황과 사용자의 공감 메시지를 평가해주세요.

    시나리오:
    "{scenario}"

    사용자의 메시지:
    "{user_message}"

    출력(JSON) 형식:
    {{
        "score": 0~100 숫자,
        "feedback": "한국어 상세 피드백"
    }}

    규칙:
    1. score는 숫자만.
    2. feedback은 다음 내용을 포함할 것:
        - 공감이 잘 된 부분
        - 부족한 부분
        - 개선을 위한 구체적 조언
    3. 전체 피드백은 친절하게.
    4. JSON만 출력, 코드블록 금지.
    5. 한국어만 사용.
    """

    # -----------------------------
    # 🔥 OpenAI 호출
    # -----------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    result_text = response.choices[0].message.content

    # GPT응답 JSON 파싱
    try:
        gpt_json = json.loads(result_text)
    except Exception:
        raise ValueError(f"GPT JSON 파싱 실패: {result_text}")

    score = gpt_json["score"]
    feedback = gpt_json["feedback"]

    #  access_token → user_id 파싱
    # -------------------------------------------------------
    user_id = None

    if token:
        payload = verify_access_token(token)  
        if payload:
            user_id = int(payload.get("sub"))  

    # 🔥 user_id 있으면 DB 저장
    if user_id:
        history = EmpathyTrainingResult(
            user_id=user_id,
            emotion_label=emotion,
            scenario_text=scenario,
            user_reply=user_message,
            empathy_score = score,
            feedback=feedback
        )
        session.add(history)
        session.commit()
        session.refresh(history)

    # 점수와 gpt피드백 최종 반환
    return {
    "score": score,
    "feedback": feedback
}


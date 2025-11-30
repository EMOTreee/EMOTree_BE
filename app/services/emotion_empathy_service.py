import json
import random
import os
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
        "scenario": "한국어 시나리오 텍스트"
    }}

    규칙:
    1. 감정을 직접적으로 언급하지 말고 '상황 묘사'로 표현할 것.
    2. 시나리오는 현실적이고 공감 가능한 내용으로 작성할 것.
    3. scenario는 반드시 한국어로.
    4. JSON만 출력. 코드블록 금지.
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
):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    # JSON 파싱
    try:
        gpt_json = json.loads(result_text)
    except Exception:
        raise ValueError(f"GPT JSON 파싱 실패: {result_text}")

    score = gpt_json["score"]
    feedback = gpt_json["feedback"]

    return {
    "score": score,
    "feedback": feedback
}


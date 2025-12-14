import json
import random
import os
from sqlmodel import Session
from fastapi import Request
from sqlmodel import Session
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

from app.models.enums import EmotionLabel
from app.schemas.emotion_empathy_schema import (
    SelectedEmotionQuery,
    EmpathyEvaluateRequest,
)
from app.utils.jwt_provider import verify_access_token
from app.models.empathy_training_result import EmpathyTrainingResult
from app.models.empathy_type import EmpathyType

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
    system_prompt = """
    당신은 공감 능력 코칭 전문가입니다.

    아래 시나리오와 사용자의 공감 메시지를 평가한 뒤,
    반드시 아래 JSON 형식으로만 출력하세요:

    {
    "score": 0~100,
    "feedback": "한국어 상세 피드백"
    }

    규칙:
    1. score는 0에서 100 사이의 숫자만 출력할 것.
    2. feedback에는 다음을 포함할 것:
    - 공감이 잘 된 부분
    - 개선을 위한 구체적인 조언
    - 만약 부족한 부분이 있다면 부족한 부분도 포함
    4. 전체 피드백은 친절하고 코칭하듯 작성할 것.
    5. JSON 이외의 내용은 절대 출력하지 말 것.
    6. 코드블록 사용 금지.
    7. 한국어만 사용할 것.
    """
    
    user_prompt = f"""
    시나리오:
    "{scenario}"

    사용자의 메시지:
    "{user_message}"
    """

    # -----------------------------
    # 🔥 OpenAI 호출
    # -----------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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


    predicted_label = None
    
    # 🔥 user_id 있으면 DB 저장
    if user_id:
        predicted_label = classify_empathy(client, user_message)
        type_history = EmpathyType(
            user_id=user_id,
            empathy_category=predicted_label
        )
        session.add(type_history)
        session.commit()
        session.refresh(type_history)
        
        training_history = EmpathyTrainingResult(
            user_id=user_id,
            emotion_label=emotion,
            scenario_text=scenario,
            user_reply=user_message,
            empathy_score = score,
            feedback=feedback
        )
        session.add(training_history)
        session.commit()
        session.refresh(training_history)

    # 점수와 gpt피드백 최종 반환
    return {
        "score": score,
        "feedback": feedback
    }


def embed(client:OpenAI, text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return resp.data[0].embedding

# 👉 이미 만들어진 DB만 사용 (인덱싱은 다른 파일에서)
chroma_client = chromadb.PersistentClient(path="./chroma/db")
collection = chroma_client.get_or_create_collection(
    name="empathy_training",
)

def classify_empathy(client:OpenAI, user_text: str) -> str:
    user_vec = embed(client, user_text)

    results = collection.query(
        query_embeddings=[user_vec],
        n_results=5,
    )

    labels = [m["label"] for m in results["metadatas"][0]]
    return max(set(labels), key=labels.count)
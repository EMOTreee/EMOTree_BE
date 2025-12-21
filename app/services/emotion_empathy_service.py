import random
import os
from typing import Dict, Tuple

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

from langchain.memory import ConversationBufferWindowMemory

from app.services.emotion_empathy_chain import build_empathy_multi_chain

load_dotenv()

empathy_multi_chain = build_empathy_multi_chain(model_name="gpt-4o-mini")

empathy_user_memories: Dict[
    Tuple[int, str],
    ConversationBufferWindowMemory
] = {}


def get_or_create_empathy_memory(
    user_id: int,
    emotion: str
) -> ConversationBufferWindowMemory:
    key = (user_id, emotion)

    if key not in empathy_user_memories:
        empathy_user_memories[key] = ConversationBufferWindowMemory(
            k=5,  # 감정별 최대 5개
            input_key="user_message",
            output_key="feedback",
            memory_key="chat_history",
            return_messages=False
        )
        print(f"[INFO] 공감 메모리 생성 - user:{user_id}, emotion:{emotion}")

    return empathy_user_memories[key]


def reset_empathy_memory(user_id: int, emotion: str):
    key = (user_id, emotion)
    if key in empathy_user_memories:
        empathy_user_memories[key].clear()
        print(f"[INFO] 공감 메모리 초기화 - user:{user_id}, emotion:{emotion}")


# -------------------------------------------------------
# 1) 공감 시나리오 생성 서비스 (멀티체인 적용)
# -------------------------------------------------------
async def create_empathy_scenario_service(
    *,
    query: SelectedEmotionQuery,
    token: str | None
):

    user_id = None
    if token:
        payload = verify_access_token(token)
        if payload:
            user_id = int(payload.get("sub"))
            reset_empathy_memory(user_id=user_id, emotion=query.option)

    # Emotion이 RANDOM이면 랜덤 선택
    if query.option == EmotionLabel.RANDOM:
        emotions = [e for e in EmotionLabel if e not in (EmotionLabel.RANDOM, EmotionLabel.NEUTRAL)]
        chosen_emotion = random.choice(emotions)
    else:
        chosen_emotion = query.option

    # ✅ 수정: OpenAI 호출 → 멀티체인 호출
    gpt_json = empathy_multi_chain.invoke(
        {"task": "scenario", "emotion": chosen_emotion.name}
    )

    # 에러 방어
    if "error" in gpt_json:
        raise ValueError(gpt_json["error"])

    scenario_text = gpt_json["scenario"]

    return {
        "emotion": chosen_emotion.name,
        "scenario": scenario_text
    }


# -------------------------------------------------------
# ⭐ 2) 공감 메시지 평가 서비스 (멀티체인 적용)
# -------------------------------------------------------
async def evaluate_empathy_message_service(
    *,
    body: EmpathyEvaluateRequest,
    token: str | None,
    session: Session
):
    # ✅ 유지: embedding/Chroma 분류용 OpenAI client는 필요
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    emotion = body.emotion
    scenario = body.scenario
    user_message = body.userMessage

    user_id = None
    if token:
        payload = verify_access_token(token)
        if payload:
            user_id = int(payload.get("sub"))

    memory = None
    chat_history = "이전 시도 이력: 없음"

    if user_id:
        memory = get_or_create_empathy_memory(user_id, emotion)
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history") or "이전 시도 이력: 없음"

    print(chat_history)

    gpt_json = empathy_multi_chain.invoke(
        {
            "task": "evaluate",
            "chat_history": chat_history,
            "scenario": scenario,
            "user_message": user_message,
        }
    )

    # 에러 처리
    if "error" in gpt_json:
        raise ValueError(gpt_json["error"])

    # 점수/피드백 추출
    score = gpt_json["score"]
    feedback = gpt_json["feedback"]

    # 메모리 저장
    if memory:
        memory.save_context(
            {"user_message": user_message},
            {"feedback": f"점수: {score}, 피드백: {feedback}"}
        )

    # user_id 있으면 DB 저장
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
            empathy_score=score,
            feedback=feedback
        )
        session.add(training_history)
        session.commit()
        session.refresh(training_history)

    return {
        "score": score,
        "feedback": feedback
    }


# -------------------------------------------------------
# Embedding / Chroma 분류 (기존 유지)
# -------------------------------------------------------
def embed(client: OpenAI, text: str):
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


def classify_empathy(client: OpenAI, user_text: str) -> str:
    user_vec = embed(client, user_text)

    results = collection.query(
        query_embeddings=[user_vec],
        n_results=5,
    )

    labels = [m["label"] for m in results["metadatas"][0]]
    return max(set(labels), key=labels.count)

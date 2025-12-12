from sqlmodel import Session, select, func
from sqlalchemy import Float
from datetime import datetime
from app.models.emotion_quiz_result import EmotionQuizResult
from app.models.empathy_training_result import EmpathyTrainingResult
from app.models.emotion_expression_result import EmotionExpressionResult
from openai import OpenAI
import os
import json
import re

def generate_monthly_report(session: Session, user_id: int, prev_year, prev_month, year, month):
    
    # 지난 달
    quiz_stats_now = get_quiz_stats_by_month(session, user_id, year, month)
    empathy_stats_now = get_empathy_stats_by_month(session, user_id, year, month)
    expression_stats_now = get_expression_stats_by_month(session, user_id, year, month)

    # 그 전 달
    quiz_stats_prev = get_quiz_stats_by_month(session, user_id, prev_year, prev_month)
    empathy_stats_prev = get_empathy_stats_by_month(session, user_id, prev_year, prev_month)
    expression_stats_prev = get_expression_stats_by_month(session, user_id, prev_year, prev_month)

    # LLM에 전달할 prompt 생성
    prompt = f"""
    유저의 활동 보고서를 생성해줘.

    📌 [이번 달: {year}년 {month}월]

    ▶ 감정 인지 퀴즈
    - 전체 평균: {quiz_stats_now["overall"]}
    - 감정별: {quiz_stats_now["by_emotion"]}

    ▶ 감정 표현
    - 전체 평균: {expression_stats_now["overall"]}
    - 감정별: {expression_stats_now["by_emotion"]}

    ▶ 공감 훈련
    - 전체 평균: {empathy_stats_now["overall"]}
    - 감정별: {empathy_stats_now["by_emotion"]}

    -------------------------------------------------

    📌 [지난 달: {prev_year}년 {prev_month}월]

    ▶ 감정 인지 퀴즈
    - 전체 평균: {quiz_stats_prev["overall"]}
    - 감정별: {quiz_stats_prev["by_emotion"]}

    ▶ 감정 표현
    - 전체 평균: {expression_stats_prev["overall"]}
    - 감정별: {expression_stats_prev["by_emotion"]}

    ▶ 공감 훈련
    - 전체 평균: {empathy_stats_prev["overall"]}
    - 감정별: {empathy_stats_prev["by_emotion"]}

    -------------------------------------------------

    위 자료를 기반으로:
    - 월간 분석
    - 성장/감소 포인트
    - 부족한 감정/강점 감정
    - 다음 달 개선 목표
    를 상세하게 작성해줘.

    점수와 같은 정확한 수치를 넣지 말고 감정별 피드백을 친절하게 설명해줘.
    감정 라벨은 영어가 아닌 한글로 표기해줘.
    지난 달과 그 전 달에 대한 언급을 하며 피드백 해줘.
    감정적 약점이 있다면 해당 부분을 자세히 설명하고 계획도 제공해줘.

    1. 해당 항목에 데이터가 없으면 문자열 대신 **null**로 표시해줘.
    2. 데이터가 존재하면 실제 분석 내용을 문자열로 작성해줘.
    3. JSON 출력만 해주고, 추가 텍스트나 예시는 포함하지 마.

    출력은 반드시 아래 JSON 형식을 그대로 지켜서 반환해줘:
    {{
        "quiz_analysis": "string" | "null",
        "empathy_analysis": "string" | "null",
        "expression_analysis": "string" | "null"
    }}
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    result_text = response.choices[0].message.content

    try:
        gpt_json = parse_gpt_json(result_text)
    except json.JSONDecodeError:
        raise ValueError(f"GPT JSON 파싱 실패: {result_text}")

    return {
        "quiz_analysis": gpt_json.get("quiz_analysis") if quiz_stats_now["overall"] is not None else None,
        "empathy_analysis": gpt_json.get("empathy_analysis") if empathy_stats_now["overall"] is not None else None,
        "expression_analysis": gpt_json.get("expression_analysis") if expression_stats_now["overall"] is not None else None
    }


def get_expression_stats_by_month(session, user_id, year, month):
    start_date = datetime(year, month, 1)
    end_month = month + 1 if month < 12 else 1
    end_year = year if month < 12 else year + 1
    end_date = datetime(end_year, end_month, 1)

    query = (
        select(
            EmotionExpressionResult.target_emotion,
            func.avg(EmotionExpressionResult.expression_score)
        )
        .where(
            EmotionExpressionResult.user_id == user_id,
            EmotionExpressionResult.created_at >= start_date,
            EmotionExpressionResult.created_at < end_date
        )
        .group_by(EmotionExpressionResult.target_emotion)
    )

    emotion_avgs = session.exec(query).all()

    overall_avg = session.exec(
        select(func.avg(EmotionExpressionResult.expression_score))
        .where(
            EmotionExpressionResult.user_id == user_id,
            EmotionExpressionResult.created_at >= start_date,
            EmotionExpressionResult.created_at < end_date
        )
    ).one()

    return {
        "overall": overall_avg,
        "by_emotion": {emotion.value: avg for emotion, avg in emotion_avgs}
    }

def get_quiz_stats_by_month(session, user_id, year, month):
    start_date = datetime(year, month, 1)
    end_month = month + 1 if month < 12 else 1
    end_year = year if month < 12 else year + 1
    end_date = datetime(end_year, end_month, 1)

    emotion_avgs = session.exec(
        select(
            EmotionQuizResult.emotion_label,
            func.avg(func.cast(EmotionQuizResult.is_correct, Float))
        )
        .where(
            EmotionQuizResult.user_id == user_id,
            EmotionQuizResult.created_at >= start_date,
            EmotionQuizResult.created_at < end_date
        )
        .group_by(EmotionQuizResult.emotion_label)
    ).all()

    overall_avg = session.exec(
        select(func.avg(func.cast(EmotionQuizResult.is_correct, Float)))
        .where(
            EmotionQuizResult.user_id == user_id,
            EmotionQuizResult.created_at >= start_date,
            EmotionQuizResult.created_at < end_date
        )
    ).one()

    return {
        "overall": overall_avg,
        "by_emotion": {emotion.value: avg for emotion, avg in emotion_avgs}
    }

def get_empathy_stats_by_month(session, user_id, year, month):
    start_date = datetime(year, month, 1)
    end_month = month + 1 if month < 12 else 1
    end_year = year if month < 12 else year + 1
    end_date = datetime(end_year, end_month, 1)

    emotion_avgs = session.exec(
        select(
            EmpathyTrainingResult.emotion_label,
            func.avg(EmpathyTrainingResult.empathy_score)
        )
        .where(
            EmpathyTrainingResult.user_id == user_id,
            EmpathyTrainingResult.created_at >= start_date,
            EmpathyTrainingResult.created_at < end_date
        )
        .group_by(EmpathyTrainingResult.emotion_label)
    ).all()

    overall_avg = session.exec(
        select(func.avg(EmpathyTrainingResult.empathy_score))
        .where(
            EmpathyTrainingResult.user_id == user_id,
            EmpathyTrainingResult.created_at >= start_date,
            EmpathyTrainingResult.created_at < end_date
        )
    ).one()

    return {
        "overall": overall_avg,
        "by_emotion": {emotion.value: avg for emotion, avg in emotion_avgs}
    }

def parse_gpt_json(result_text: str):
    # ```json ... ``` 코드블록 제거
    cleaned = re.sub(r"^```json\s*|\s*```$", "", result_text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)

def check_data(stats):
    if stats["overall"] is None:
        return None
    return stats
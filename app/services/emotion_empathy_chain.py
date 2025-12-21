import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """수정: LLM 출력이 JSON만 오도록 유도하지만 섞여오면 마지막 방어."""
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def build_empathy_multi_chain(model_name: str = "gpt-4o-mini"):
    """
    멀티체인 입력:
      - {"task":"scenario", "emotion":"JOY"}
      - {"task":"evaluate", "chat_history":"...", "scenario":"...", "user_message":"..."}

    출력: dict(JSON)
    """

    # -----------------------------
    # 1) Scenario chain
    # -----------------------------
    scenario_llm = ChatOpenAI(model=model_name, temperature=0.8)

    scenario_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """
                당신은 공감 능력을 기르는 연습을 돕는 "상황 생성 전문가"입니다.

                아래 감정에 해당하는, 사용자가 공감 연습에 사용할 짧고 구체적인 상황 설명을 만들어주세요.

                감정:
                - {emotion}

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
                """.strip(),
            )
        ]
    )

    scenario_chain = (
        scenario_prompt
        | scenario_llm
        | StrOutputParser()
        | RunnableLambda(_safe_json_loads)
    )

    # -----------------------------
    # 2) Evaluate chain
    # -----------------------------
    eval_llm = ChatOpenAI(model=model_name, temperature=0.2)

    eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 공감 능력 코칭 전문가입니다.

                [이전 공감 시도 이력]
                {chat_history}

                아래 시나리오와 사용자의 공감 메시지를 평가한 뒤,
                반드시 아래 JSON 형식으로만 출력하세요:

                {{
                "score": 0~100,
                "feedback": "한국어 상세 피드백"
                }}

                규칙:
                1. score는 0에서 100 사이의 숫자만 출력할 것.
                2. feedback에는 다음을 포함할 것:
                - 공감이 잘 된 부분
                - 이전 시도가 있다면 대비 개선/악화된 점
                - 개선을 위한 구체적인 조언
                - 부족한 부분이 있다면 부족한 부분도 포함
                3. 전체 피드백은 친절하고 코칭하듯 작성할 것.
                4. JSON 이외의 내용은 절대 출력하지 말 것.
                5. 코드블록 사용 금지.
                6. 한국어만 사용할 것.
                """.strip(),
            ),
            (
                "user",
                """
                시나리오:
                "{scenario}"

                사용자의 메시지:
                "{user_message}"
                """.strip(),
            ),
        ]
    )

    evaluate_chain = (
        eval_prompt
        | eval_llm
        | StrOutputParser()
        | RunnableLambda(_safe_json_loads)
    )

    # -----------------------------
    # 3) Router (멀티체인)
    # -----------------------------
    router = RunnableBranch(
        (lambda x: x.get("task") == "scenario", scenario_chain),
        (lambda x: x.get("task") == "evaluate", evaluate_chain),
        RunnableLambda(lambda _: {"error": "Invalid task. Use 'scenario' or 'evaluate'."}),
    )

    return router

import base64
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


async def analyze_emotion_service(image_bytes: bytes, target_emotion: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # targetEmotion 대문자로 통일
    target_emotion = target_emotion.upper()

    # Base64 변환
    base64_image = base64.b64encode(image_bytes).decode()

    # ----------------------------
    # 🔥 최종 개선된 프롬프트
    # ----------------------------
    prompt = f"""
당신은 감정 인식과 감정 표현을 어려워하는 사람들을 돕는 “감정 표현 코치”입니다.  

사용자가 표현하려고 하는 목표 감정(targetEmotion)은 다음과 같습니다:
- targetEmotion: "{target_emotion}"

업로드된 얼굴 표정을 분석한 뒤 아래 JSON 형식으로만 출력하세요:

{{
  "detectedEmotion": "JOY" | "SADNESS" | "ANGER" | "SURPRISE" | "ANXIETY",
  "feedback": "한국어로 된 상세 피드백"
}}

규칙:
1. detectedEmotion은 위 5개 중 하나만 선택.
2. feedback은 다음 내용을 반드시 포함해야 함:
   - 사진 속 얼굴 특징(눈, 입, 눈썹 등)을 기준으로 어떤 감정처럼 보였는지 분석
   - detectedEmotion과 targetEmotion이 같다면:
        - 어떤 요소 덕분에 감정이 잘 표현되었는지 칭찬
        - 더욱 자연스럽게 / 더 강하게 표현하려면 어떻게 해야 하는지 구체적 조언
   - detectedEmotion과 targetEmotion이 다르다면:
        - 어떤 요소 때문에 다른 감정으로 보였는지 설명
        - targetEmotion에 더 가까워지기 위해 어떤 조정이 필요한지 조언
3. 전체 피드백은 친절하고 코칭하듯 말할 것.
4. JSON 이외의 내용은 절대 출력하지 말 것.
5. 코드블록(```json```) 사용 금지.
6. 사진이라는 단어 사용 금지.
7. 감정을 한국어로 표현할 것(기쁨, 슬픔, 분노, 놀람, 불안).
8. feedback은 반드시 한국어만 사용
"""

    # ----------------------------
    # 🔥 OpenAI Vision 호출
    # ----------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # 이미지 분석 가능한 최신 소형 모델
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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

    # 점수 계산
    score = 100 if detected == target_emotion else 40

    # 최종 반환
    return {
        "targetEmotion": target_emotion,
        "detectedEmotion": detected,
        "score": score,
        "feedback": gpt_json["feedback"]
    }

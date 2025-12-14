import base64, os, random, uuid, time
from typing import Literal, Tuple

from cachetools import TTLCache
from sqlmodel import Session

from app.core.config import settings
from app.models.enums import EmotionLabel
from app.models.emotion_quiz_result import EmotionQuizResult
from app.utils.jwt_provider import verify_access_token

# 캐시: 1000문항, TTL 30분
_quiz_cache: TTLCache = TTLCache(maxsize=1000, ttl=60 * 30)


def _save_base64_image(b64_png: str, emotion: EmotionLabel) -> str:
    # static/images/GENERATED/<emotion>/<uuid>.png
    out_dir = os.path.join(settings.IMAGE_ROOT, emotion.value)

    os.makedirs(out_dir, exist_ok=True)
    name = f"{uuid.uuid4().hex}.png"
    raw = base64.b64decode(b64_png)

    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(raw)

    rel_dir = f"GENERATED/{emotion.value}"
    image_url = f"{settings.SERVER_URL}/static/images/{rel_dir}/{name}"
    return image_url


def _prompt_for(emotion: EmotionLabel) -> str:
    m = {
        EmotionLabel.JOY: (
            "A person naturally showing signs of joy. The corners of the mouth are lifted, "
            "the eyes gently narrow with subtle crow’s-feet forming, reflecting the facial muscle movements "
            "typically seen in positive emotions."
        ),
        EmotionLabel.ANGER: (
            "A person reflecting anger. The eyebrows are tightly drawn together, the eyes narrow sharply, "
            "and the jaw and mouth muscles appear tense—facial signals commonly associated with negative emotions."
        ),
        EmotionLabel.SADNESS: (
            "A person showing signs of sadness. The gaze is lowered, the eyelids droop slightly, "
            "and the muscles around the mouth relax downward, presenting features that express a subdued emotional state."
        ),
        EmotionLabel.SURPRISE: (
            "A person displaying sudden surprise. The eyes are opened wide, the eyebrows lift upward, "
            "and the mouth slightly opens—an immediate facial reaction indicating unexpected emotion."
        ),
        EmotionLabel.ANXIETY: (
            "A person expressing anxiety. The eyebrows draw closer with a slight upward pinch, "
            "the lips tighten or press together subtly, and the eyes convey tension and uneasiness—"
            "delicate facial cues that indicate inner worry."
        ),
    }

    return (
        f"Photo, realistic portrait. {m[emotion]}. "
        "The emotion should be conveyed through natural, unexaggerated facial muscle movements. "
        "Neutral background, natural skin tone, no distortion or excessive stylization."
    )


def _summary_for(emotion: EmotionLabel) -> str:
    s = {
        EmotionLabel.JOY: (
            """입술이 크게 벌어지면서 상하 치아가 드러나는 활짝 웃는 입 모양이 중심적이며, 입꼬리가 자연스럽게 위로 올라가 있습니다. 
            눈은 살짝 좁혀지고 눈가 주변에 미소로 인한 잔주름이 생겨 눈웃음을 형성하며, 시선은 카메라를 향해 집중되어 있어 생동감 있는 즐거움을 전달합니다. 
            어깨와 목 주변 근육도 긴장되지 않고 자연스럽게 풀려 있어 전체적으로 편안하면서도 행복한 자세를 보여줍니다."""
        ),
        EmotionLabel.ANGER: (
            """이마와 눈 사이의 깊게 잡힌 주름은 눈썹을 아래로 모아 찌푸린 상태를 보여주며, 눈은 강하게 뜨이고 시선이 정면을 응시하여 날카로운 긴장감을 줍니다. 
            또한 입술은 아래로 굽어져 단단히 다물려 있으며, 턱 근육이 긴장되어 있어 전반적으로 분노와 적대감을 강조합니다."""
        ),
        EmotionLabel.SADNESS: (
            """눈썹이 안쪽으로 모이며 위로 올라가 얼굴 중앙에 주름이 깊게 생겨 전형적인 걱정·슬픔 표정을 형성하고 있습니다. 
            눈은 약간 아래로 처져 있고 시선은 땅이나 아래쪽을 향해 있어 우울한 느낌을 강조합니다. 
            입꼬리는 아래로 처져 있으며, 입술이 굳어 있지 않고 약간 풀려 있어 슬픈 감정을 나타내고 있고, 어깨는 살짝 내려가 있어 몸 전체가 무거운 감정을 반영하고 있습니다."""
        ),
        EmotionLabel.SURPRISE: (
            """눈이 크게 뜨여 있으며, 눈썹이 위로 올라가 있어 시선이 넓게 펼쳐진 듯한 인상을 줍니다.
            입은 살짝 벌어져 있고, 턱이 약간 내려가 있어 갑작스러운 충격이나 놀람을 표현합니다.
            어깨가 살짝 위로 올라가고, 몸이 약간 뒤로 젖혀져 있어 긴장과 반사적인 반응이 드러납니다."""
        ),
        EmotionLabel.ANXIETY: (
            """눈썹이 강하게 위로 모이며 이마에 깊은 주름이 잡힌 모습은 내면의 긴장감과 불안함을 반영합니다. 
            눈동자가 흔들리듯 움직이며 시선이 불안정하게 흔들리는 점과, 입술이 단단히 다물어진 채 입꼬리가 약간 아래로 내려간 모습은 마음 속 초조함과 걱정을 짙게 드러냅니다. 
            전체적으로 얼굴과 목 주변 근육이 긴장되어 있어 몸도 함께 경직된 듯한 인상을 줍니다."""
        ),
    }
    return s[emotion]


def _analyze_emotion_features_from_base64(
    openai_client,
    b64_image: str,
    emotion: EmotionLabel,
    mime: str = "image/png",
) -> str:
    """
    표정 이미지를 분석해서
    '어떤 표정/얼굴 특징 때문에 이 감정처럼 보이는지'를 한국어로 설명.
    """

    # client 없으면 그냥 기존 요약으로 fallback
    if openai_client is None:
        return _summary_for(emotion)

    system_prompt = """
    당신은 얼굴 표정과 신체 자세를 기반으로 감정의 시각적 단서를 설명하는 전문가입니다.

    규칙:
    - 감정 이름(기쁨, 분노 등)을 직접 언급하지 말 것.
    - 입꼬리, 눈 모양, 시선, 얼굴 근육, 주름, 어깨나 자세 등
    구체적인 시각적 특징을 중심으로 설명할 것.
    - 설명은 2~3문장의 한국어 문장으로 작성할 것.
    - 해석은 단정적이지 않고 관찰 기반으로 서술할 것.
    - 예: '입꼬리가 아래로 처지고 눈꺼풀이 무겁게 내려앉은 모습은, 마음속 무거움이 얼굴 표정에 고스란히 묻어날 때 보이는 특징입니다. 시선이 쉽게 떨어지고 눈가의 근육이 힘없이 풀려 있는 모습이, 감정이 전반적으로 가라앉아 있음을 드러냅니다.'
    """

    user_prompt = f"""
    다음 인물 이미지에서,
    '{emotion.name}' 감정이 드러난다고 볼 수 있는 시각적 단서를 설명해주세요.
    """

    resp = openai_client.responses.create(
        model="gpt-4o-mini",  # vision 지원되는 경량 모델 가정
        input=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_prompt,
                },
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{b64_image}",
                },
            ],
        }],
    )
    return resp.output_text.strip()


def pick_static(openai_client) -> Tuple[str, EmotionLabel, str]:
    emotions = [e for e in EmotionLabel if e != "RANDOM"]
    emotion = random.choice(emotions)
    folder_path = os.path.join(settings.IMAGE_ROOT, emotion)

    print(folder_path)

    if not os.path.isdir(folder_path):
        return ("/static/quiz/placeholder.png", emotion, _summary_for(emotion))

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not files:
        # 이미지가 없으면 fallback
        return ("/static/images/placeholder.png", emotion, _summary_for(emotion))

    # 이미지 랜덤 선택
    chosen = random.choice(files)
    abs_path = os.path.join(folder_path, chosen)
    image_url = f"{settings.SERVER_URL}/static/images/GENERATED/{emotion.name}/{chosen}"

    with open(abs_path, "rb") as f:
        png_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 비용 문제로 임시 요약만 사용
    summary = _summary_for(emotion)

    # 필요하면 Vision 분석 사용
    # summary = _analyze_emotion_features_from_base64(
    #     openai_client=openai_client,
    #     b64_image=png_b64,
    #     emotion=emotion,
    #     mime="image/png",
    # )

    return (image_url, emotion, summary)


def pick_dalle(openai_client, emotion: EmotionLabel | None = None) -> Tuple[str, EmotionLabel, str]:
    if emotion is None:
        emotion = random.choice([
            e for e in EmotionLabel
            if e != EmotionLabel.RANDOM
        ])
    prompt = _prompt_for(emotion)

    # OpenAI 이미지 생성
    png_b64 = openai_client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size="1024x1024",
        quality="low",
        n=1,
    ).data[0].b64_json

    rel_url = _save_base64_image(png_b64, emotion)

    summary = _analyze_emotion_features_from_base64(
        openai_client=openai_client,
        b64_image=png_b64,
        emotion=emotion,
        mime="image/png",
    )

    return (rel_url, emotion, summary)


def generate_question(
    source: Literal["STATIC", "DALLE"] = "STATIC",
    openai_client=None,
):
    if source == "DALLE" and openai_client is not None:
        image_url, answer, summary = pick_dalle(openai_client)
    else:
        image_url, answer, summary = pick_static(openai_client)

    qid = uuid.uuid4().hex
    _quiz_cache[qid] = {"answer": answer, "ts": time.time(), "summary": summary}
    return qid, image_url, summary


def grade(question_id: str, user_answer: EmotionLabel) -> Tuple[bool, EmotionLabel, str] | None:
    data = _quiz_cache.get(question_id)
    if not data:
        return None  # 만료됨 / 존재하지 않음

    correct = data["answer"]
    is_correct = (user_answer == correct)
    feedback = data["summary"]
    return is_correct, correct, feedback


# ---------------------------------------------------------
# 제출 + DB 저장 서비스 (로그인 상태일 때만 기록)
# ---------------------------------------------------------
async def submit_emotion_quiz_service(
    *,
    quiz_id: str,
    user_answer: EmotionLabel,
    token: str | None,
    session: Session,
):
    """
    - quiz_id / user_answer로 채점
    - access_token이 유효하면 EmotionQuizResult 테이블에 저장
    - 프론트 응답 형식: QuizSubmitResponse와 동일
    """

    graded = grade(quiz_id, user_answer)
    if graded is None:
        # 캐시에서 사라진 경우 (TTL 만료 등)
        raise ValueError("유효하지 않거나 만료된 퀴즈입니다.")

    is_correct, correct_emotion, feedback = graded

    # access_token → user_id 파싱
    user_id: int | None = None
    if token:
        payload = verify_access_token(token)
        if payload:
            try:
                user_id = int(payload.get("sub"))
            except (TypeError, ValueError):
                user_id = None

    # 로그인 상태면 결과 DB 저장
    if user_id is not None:
        record = EmotionQuizResult(
            user_id=user_id,
            emotion_label=correct_emotion,
            is_correct=is_correct,
        )
        session.add(record)
        session.commit()
        session.refresh(record)

    # QuizSubmitResponse 스펙에 맞춰 반환
    return {
        "isCorrect": is_correct,
        "correctEmotion": correct_emotion,
        "feedback": feedback,
    }

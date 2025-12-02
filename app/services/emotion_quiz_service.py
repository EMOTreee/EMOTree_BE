import base64, os, random, uuid, time
from typing import Literal, Tuple
from cachetools import TTLCache
from app.models.enums import EmotionLabel
from app.core.config import settings

# 캐시: 1000문항, TTL 30분
_quiz_cache: TTLCache = TTLCache(maxsize=1000, ttl=60*30)
_STATIC_SET: list[tuple[str, EmotionLabel]] = []

def _load_static_set_once() -> None:
    global _STATIC_SET
    if _STATIC_SET:
        return
    root = settings.STATIC_IMAGE_DIR
    plan = [
        ("JOY/1.jpg", EmotionLabel.JOY),
        ("JOY/2.jpg", EmotionLabel.JOY),
        ("ANGER/1.jpg", EmotionLabel.ANGER),
        ("ANGER/2.jpg", EmotionLabel.ANGER),
        ("SADNESS/1.jpg", EmotionLabel.SADNESS),
        ("SADNESS/2.jpg", EmotionLabel.SADNESS),
        ("SURPRISE/1.jpg", EmotionLabel.SURPRISE),
        ("SURPRISE/2.jpg", EmotionLabel.SURPRISE),
        ("ANXIETY/1.jpg", EmotionLabel.ANXIETY),
        ("ANXIETY/2.jpg", EmotionLabel.ANXIETY),
    ]
    _STATIC_SET = [(f"/static/quiz/{p}", lab) for (p, lab) in plan if os.path.exists(os.path.join(root, p))]

def _save_base64_image(b64_png: str, emotion: EmotionLabel) -> str:
    # static/quiz/GENERATED/<emotion>/<uuid>.png
    rel_dir = f"GENERATED/{emotion.value}"
    out_dir = os.path.join(settings.STATIC_IMAGE_DIR, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    name = f"{uuid.uuid4().hex}.png"
    raw = base64.b64decode(b64_png)
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(raw)
    return f"/static/quiz/{rel_dir}/{name}"

def _prompt_for(emotion: EmotionLabel) -> str:
    m = {
        EmotionLabel.JOY: "a person smiling brightly, eyes slightly squinted, warm lighting, candid portrait",
        EmotionLabel.ANGER: "a person with furrowed brows and tense jaw, dramatic contrast, cinematic portrait",
        EmotionLabel.SADNESS: "a person looking down with teary eyes, soft cold lighting, moody portrait",
        EmotionLabel.SURPRISE: "a person with widened eyes and open mouth, mid-motion, high shutter, portrait",
        EmotionLabel.ANXIETY: "a person with uneasy expression, biting lip, low contrast, grainy film look",
    }
    # 안전하게 얼굴 디테일 강조, 과한 왜곡 금지
    return f"Photo, realistic portrait, {m[emotion]}. Neutral background, natural skin tone."

def _summary_for(emotion: EmotionLabel) -> str:
    s = {
        EmotionLabel.JOY: "입꼬리가 위로 올라가고 눈가 주름이 잡힌다.",
        EmotionLabel.ANGER: "미간이 좁아지고 턱이 굳는다.",
        EmotionLabel.SADNESS: "눈썹 안쪽이 위로 치켜 올라가고 시선이 아래로 떨어진다.",
        EmotionLabel.SURPRISE: "눈이 커지고 입이 벌어진다.",
        EmotionLabel.ANXIETY: "시선이 흔들리고 입술을 깨물거나 굳힌다.",
    }
    return s[emotion]

def _gpt_summary_fallback(emotion: EmotionLabel) -> str:
    # 필요 시 GPT 붙여서 한 줄 설명 생성하는 자리.
    return _summary_for(emotion)

def pick_static() -> Tuple[str, EmotionLabel, str]:
    _load_static_set_once()
    if not _STATIC_SET:
        # 세팅 전이면 임시로 요약만
        return ("/static/quiz/placeholder.png", EmotionLabel.JOY, _summary_for(EmotionLabel.JOY))
    img, lab = random.choice(_STATIC_SET)
    return (img, lab, _gpt_summary_fallback(lab))

def pick_dalle(openai_client, emotion: EmotionLabel | None = None) -> Tuple[str, EmotionLabel, str]:
    if emotion is None:
        emotion = random.choice(list(EmotionLabel))
    prompt = _prompt_for(emotion)
    
    # OpenAI 이미지 생성
    png_b64 = openai_client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        n=1,
    ).data[0].b64_json
    rel_url = _save_base64_image(png_b64, emotion)
    return (rel_url, emotion, _gpt_summary_fallback(emotion))

def generate_question(source: Literal["STATIC","DALLE"]="STATIC", openai_client=None):
    if source == "DALLE" and openai_client is not None:
        image_url, answer, summary = pick_dalle(openai_client)
    else:
        image_url, answer, summary = pick_static()
    qid = uuid.uuid4().hex
    _quiz_cache[qid] = {"answer": answer, "ts": time.time()}
    return qid, image_url, summary

def grade(question_id: str, user_answer: EmotionLabel):
    data = _quiz_cache.get(question_id)
    if not data:
        return None  # 만료됨
    correct = data["answer"]
    is_correct = (user_answer == correct)
    feedback = "정답입니다." if is_correct else f"정답은 {correct.value}."
    return is_correct, correct, feedback

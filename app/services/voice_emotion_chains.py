import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import torch
import soundfile as sf
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableMap
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

from preprocessing.audio_preprocessing import preprocess_audio
from app.utils.acoustic_features import (
    extract_all_features,
    calculate_z_scores_with_interpretation,
    format_acoustic_features_for_llm
)

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Baseline JSON 로드
BASELINE_PATH = PROJECT_ROOT / "document" / "emotion_baseline.json"
with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    EMOTION_BASELINE = json.load(f)

# Wav2Vec2 감정 인식 모델 로드
MODEL_PATH = PROJECT_ROOT / "models" / "ehcalabres_emotion"
emotion_model = AutoModelForAudioClassification.from_pretrained(str(MODEL_PATH))
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_PATH))

# 감정 라벨 매핑
EMOTION_MAPPING = {
    "angry": "ANGER",
    "happy": "JOY",
    "sad": "SADNESS",
    "surprise": "SURPRISE",
    "fear": "ANXIETY",
    "neutral": "NEUTRAL",
    "disgust": "ANGER"
}

# 감정 라벨 한글 매핑 (피드백용)
EMOTION_KOREAN = {
    "ANGER": "분노",
    "JOY": "기쁨",
    "SADNESS": "슬픔",
    "SURPRISE": "놀람",
    "ANXIETY": "불안",
    "NEUTRAL": "중립"
}

# LLM 모델 초기화
chat_openai = ChatOpenAI(
    temperature=0.7,
    max_tokens=600,
    model="gpt-4o-mini",
)

# 사용자별 메모리 저장소 (인메모리)
user_memories: Dict[int, ConversationBufferWindowMemory] = {}

# 메모리 관리 함수
def get_or_create_memory(user_id: int) -> ConversationBufferWindowMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(
            k=5,
            input_key="user_attempt",
            output_key="feedback_result",
            memory_key="chat_history",
            return_messages=False
        )
        print(f"[INFO] 사용자 {user_id}의 새 메모리 생성")
    return user_memories[user_id]

# 메모리 초기화 함수
def reset_memory(user_id: int) -> None:
    if user_id in user_memories:
        user_memories[user_id].clear()
        print(f"[INFO] 사용자 {user_id}의 메모리 초기화")
    else:
        print(f"[INFO] 사용자 {user_id}의 메모리가 존재하지 않음")

# 1. 전처리 체인
def preprocess_audio_chain(audio_bytes: bytes) -> np.ndarray:
    try:
        preprocessed = preprocess_audio(audio_bytes, sr=16000)
        return preprocessed
    except Exception as e:
        print(f"[ERROR] 전처리 실패: {e}")
        raise

# 2-1. 감정 감지 체인
def detect_emotion_chain(audio_array: np.ndarray) -> Dict[str, any]:
    try:
        # 특징 추출
        inputs = feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # 모델 추론
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
        
        # 확률값 계산
        probs = torch.softmax(logits, dim=-1)[0]
        
        # 상위 3개 감정 가져오기
        top_k = torch.topk(probs, k=min(3, len(probs)))
        top_indices = top_k.indices.tolist()
        top_probs = top_k.values.tolist()
        
        # 상위 3개 결과 구성
        top3_results = []
        for idx, prob in zip(top_indices, top_probs):
            original_label = emotion_model.config.id2label[idx].lower()
            mapped_emotion = EMOTION_MAPPING.get(original_label, "NEUTRAL")
            top3_results.append({
                "emotion": mapped_emotion,
                "confidence": round(prob * 100, 2)  # 백분율로 변환
            })
        
        # 1위 감정
        primary_emotion = top3_results[0]["emotion"]
        
        print(f"[INFO] 감정 감지 결과:")
        for i, result in enumerate(top3_results, 1):
            print(f"  {i}위: {result['emotion']} ({result['confidence']}%)")
        
        # top3를 프롬프트용 문자열로 포맷팅
        top3_formatted = "\n".join([
            f"{i}위: {item['emotion']} ({item['confidence']}%)"
            for i, item in enumerate(top3_results, 1)
        ])
        
        return {
            "primary": primary_emotion,
            "top3": top3_results,
            "top3_formatted": top3_formatted
        }
        
    except Exception as e:
        print(f"[ERROR] 감정 감지 실패: {e}")
        return {
            "primary": "NEUTRAL",
            "top3": [{"emotion": "NEUTRAL", "confidence": 0.0}],
            "top3_formatted": "1위: NEUTRAL (0.0%)"
        }

# 2-2-1. STT 체인
def perform_stt(audio_array: np.ndarray) -> Optional[str]:
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_array, 16000, format='WAV')
            audio_file_path = tmp_file.name
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            with open(audio_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            text = transcription.text
            print(f"[INFO] STT 결과: {text}")
            return text
            
        finally:
            import os as os_module
            if os_module.path.exists(audio_file_path):
                os_module.unlink(audio_file_path)
        
    except Exception as e:
        print(f"[ERROR] STT 실패: {e}")
        return None

# 2-2-2. 텍스트 평가 체인
def build_text_evaluation_chain():
    evaluation_prompt = ChatPromptTemplate.from_template("""
            당신은 음성 감정 표현 전문가입니다.

            사용자가 "{target_emotion}" 감정을 표현하려고 음성으로 말한 내용이 다음과 같습니다:
            "{text}"

            이 텍스트가 목표 감정인 "{target_emotion}"을 표현하기에 적절한지 평가해주세요.
            평가 내용에는 다음을 포함해주세요:
            - 단어 선택의 적절성
            - 감정 표현의 명확성
            - 개선이 필요한 부분

            평가 결과를 자연스러운 문장으로 작성해주세요.
""")
    
    return evaluation_prompt | chat_openai | StrOutputParser()


# 2-3. 음향 특징 추출 체인
def extract_acoustic_features(audio_array: np.ndarray, target_emotion: str) -> Optional[str]:
    try:
        # 음향 특징 추출
        features = extract_all_features(audio_array)
        if features is None:
            return None
        
        # Target emotion의 baseline 가져오기
        target_baseline = EMOTION_BASELINE.get(target_emotion)
        if not target_baseline:
            print(f"[WARN] {target_emotion}의 baseline 없음")
            return None
        
        # Z-score 계산 및 자연어 해석
        z_scores = calculate_z_scores_with_interpretation(
            features,
            target_baseline["mean"],
            target_baseline["std"]
        )
        
        # 전체 특징을 구조화하여 포맷
        formatted_features = format_acoustic_features_for_llm(z_scores)
        
        analysis_text = f"""
            [목표 감정 '{target_emotion}'과의 음향 특징 비교]

            {formatted_features}
        """
        
        print(f"[INFO] 음향 특징 분석 완료")
        return analysis_text
        
    except Exception as e:
        print(f"[ERROR] 음향 특징 추출 실패: {e}")
        return None


# 3. 종합 평가 및 피드백 생성 체인
def build_feedback_generation_chain():
    prompt_template = """
당신은 따뜻하고 친절한 음성 감정 표현 코치입니다.

### [역할 및 청중]
1. **청중:** 당신의 피드백은 음성 훈련을 처음 시작하는 일반인을 대상으로 합니다.
2. **톤앤매너:** 긍정적이고 건설적인 언어를 사용하며, 잘한 점과 개선할 점을 균형 있게 제시하십시오.

### [핵심 지침: 기술 용어 사용 금지] 🚨
다음과 같은 전문 음향 특징 용어를 **절대 사용하지 마십시오:**
- mfcc, rms, f0, spectral centroid, z-score, pitch, energy, speaking_rate, pause_ratio 등

모든 분석 내용은 일반인이 이해할 수 있는 언어로 해석하여 설명해야 합니다.

### [기술 용어 해석 가이드]
분석 결과에 나타나는 기술 용어를 다음과 같이 해석하여 일반 언어로 치환하십시오:
- **Pitch (f0, mean_f0, std_f0, range_f0):** "목소리 톤의 높낮이", "목소리가 얼마나 높은지/낮은지", "목소리 톤의 변화"
- **Energy (rms, mean_rms, std_rms, max_rms):** "목소리의 크기", "목소리에 실린 힘", "목소리의 강약"
- **Rate & Pause (speaking_rate, pause_ratio, mean_voiced_segment_length):** "말의 빠르기", "말하는 속도", "말 사이의 쉼", "말의 흐름"
- **Timbre (mfcc, spectral_centroid):** "목소리의 울림", "목소리 색깔", "목소리의 밝기/어둡기", "목소리에 담긴 떨림"

### [감정 라벨 한글 표기]
피드백 작성 시 감정 라벨은 반드시 다음과 같이 한글로 표기하십시오:
- ANGER → "분노"
- JOY → "기쁨"
- SADNESS → "슬픔"
- SURPRISE → "놀람"
- ANXIETY → "불안"
- NEUTRAL → "중립"

사용자가 "{target_emotion_korean}" 감정을 음성으로 표현했습니다.

[이전 시도 이력]
{chat_history}

[감정 인식 모델 분석 결과 (상위 3개)]
{detected_emotion_top3}

[음향 특징 분석]
{acoustic_analysis}

[텍스트 평가]
{text_evaluation}

위 정보를 종합하여 다음을 수행해주세요:

1. **최종 감정 판단**: 
   - 감정 인식 모델 결과와 음향 특징 분석을 종합적으로 고려하여 사용자가 실제로 표현한 감정을 판단하세요
   - 모델의 1위 결과가 음향 특징과 일치하지 않으면 음향 특징을 더 우선시하세요
   - 판단한 감정은 반드시 다음 중 하나여야 합니다: JOY, SADNESS, ANGER, SURPRISE, ANXIETY, NEUTRAL
   - NEUTRAL은 사용자가 특정 감정을 명확히 표현하지 못한 일부 경우에만 선택하고 가능하면 피하세요

2. **점수 산정** (0~100):
   - 판단된 감정이 목표 감정과 일치: 가장 중요 (약 40%)
   - 음향 특징이 목표 감정의 baseline과 유사: 중요 (약 40%)
   - 텍스트 적절성: 중요 (약 20%)

3. **상세 피드백 작성 (일반인 친화적 언어 필수):**
   - 잘한 점과 개선할 점을 구체적이지만 쉬운 말로 제시
   - 기술 용어를 사용하지 말고, 위 해석 가이드를 참고하여 일반 언어로 설명
   - 예: "목소리 톤이 적절했어요", "말의 빠르기를 조절해보세요", "목소리에 힘을 더 실어보세요"
   - **감정 라벨은 반드시 한글로 표기하세요** (예: "슬픔", "불안", "기쁨" 등)
   - **이전 시도 이력이 있다면 반드시 참고하여 이번 시도의 점수가 상승/하락했는지, 어떤 부분(목소리 톤, 크기, 말의 빠르기 등)이 개선되었거나 악화되었는지 쉽게 언급하세요**

다음 형식의 JSON으로만 응답하세요:
{{
    "detectedEmotion": "판단된 감정 (JOY, SADNESS, ANGER, SURPRISE, ANXIETY, NEUTRAL 중 하나)",
    "score": 0~100 사이의 정수,
    "feedback": "사용자에게 제공할 상세한 피드백 (기술 용어 없이 일반인이 이해할 수 있는 언어로, 감정 라벨은 한글로)"
}}

만약 텍스트 평가나 음향 특징 분석이 "분석 불가"인 경우, 해당 부분을 제외하고 평가해주세요.
이전 시도 이력이 비어있다면("이전 시도 이력: 없음" 또는 빈 문자열) 첫 시도이므로 이력 관련 언급 없이 현재 시도만 평가하세요.

### [피드백 작성 예시]

**예시 1: 목표 감정 달성**
{{
    "detectedEmotion": "JOY",
    "score": 95,
    "feedback": "대단합니다! 이번 시도에서 목표 감정인 '기쁨'을 완벽하게 표현하셨습니다. 이전 기록에서 목소리 톤이 다소 낮게 유지되어 아쉬웠는데, 이번에는 목소리 톤의 높낮이 변화 폭이 크게 느껴져 매우 생동감이 넘칩니다. 목소리 크기도 일관되게 유지되었고, 목소리 울림이 밝고 선명해졌어요. 텍스트 내용도 적절했지만, 무엇보다 음성적 특징이 기준점을 넘어섰습니다. 아주 훌륭해요! 다음 감정으로 넘어가셔도 좋습니다."
}}

**예시 2: 개선 필요**
{{
    "detectedEmotion": "NEUTRAL",
    "score": 58,
    "feedback": "이번 시도에서는 슬픔보다는 평이하고 힘없는 목소리가 감지되었습니다. 텍스트는 슬픔을 표현하기 좋았지만, 말의 빠르기가 너무 빨라서 진지함이 부족했고, 목소리 크기가 너무 낮아 감정이 전달되지 못했습니다. 슬픔을 표현하려면 말의 빠르기를 훨씬 더 느리게 조절하고, 말 사이에 쉼을 의도적으로 늘려보세요. 또한, 목소리 톤을 평균보다 낮게 유지하는 것에 집중하여 감정의 깊이를 더하면 다음에는 점수가 크게 오를 것입니다. 아직은 연습이 더 필요합니다."
}}

위 예시를 참고하여, 긍정적이고 구체적이며 실행 가능한 피드백을 작성하세요.
JSON 형식으로만 응답하세요. 코드 블록이나 다른 텍스트는 포함하지 마세요.
"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    base_chain = prompt | chat_openai | JsonOutputParser()
    
    # 메모리 통합 래퍼 함수
    def create_chain_with_memory(memory: Optional[ConversationBufferWindowMemory]):
        def chain_with_memory(inputs: Dict) -> Dict:
            # 메모리에서 이전 대화 로드
            chat_history = ""
            if memory:
                memory_vars = memory.load_memory_variables({})
                chat_history = memory_vars.get("chat_history", "")
                if not chat_history:
                    chat_history = "이전 시도 이력: 없음"
            else:
                chat_history = "이전 시도 이력: 없음"
            
            # 프롬프트에 chat_history 추가
            inputs_with_history = {**inputs, "chat_history": chat_history}
            
            # 체인 실행
            result = base_chain.invoke(inputs_with_history)
            
            # 메모리에 현재 시도 저장
            if memory:
                user_attempt = f"목표 감정: {inputs['target_emotion']}, 감지된 감정: {result.get('detectedEmotion', 'NEUTRAL')}, 점수: {result.get('score', 0)}"
                feedback_result = result.get('feedback', '')
                memory.save_context(
                    {"user_attempt": user_attempt},
                    {"feedback_result": feedback_result}
                )
            
            return result
        
        return chain_with_memory
    
    return create_chain_with_memory

# 전체 파이프라인
def build_voice_emotion_pipeline():
    
    # 텍스트 평가 체인
    text_evaluation_chain = build_text_evaluation_chain()
    
    # 종합 평가 및 피드백 생성 체인 팩토리
    feedback_chain_factory = build_feedback_generation_chain()
    
    # 1단계: 전처리 체인
    preprocess_chain = RunnableLambda(
        lambda x: {
            "audio_array": preprocess_audio_chain(x["audio_bytes"]),
            "target_emotion": x["target_emotion"]
        }
    )
    
    # 2-2: STT + 텍스트 평가 체인 (순차 실행)
    def stt_and_text_evaluation(x):
        # STT 실행
        transcribed_text = perform_stt(x["audio_array"]) or ""
        
        # 텍스트 평가 (STT 결과 사용)
        text_eval = text_evaluation_chain.invoke({
            "target_emotion": x["target_emotion"],
            "text": transcribed_text
        }) if transcribed_text else "분석 불가"
        
        return text_eval
    
    # 2단계: 병렬 분석 체인 (감정 감지 + STT→텍스트평가 + 음향 특징)
    parallel_analysis_chain = RunnableMap({
        "detected_emotion": RunnableLambda(
            lambda x: detect_emotion_chain(x["audio_array"])
        ),
        "text_evaluation": RunnableLambda(stt_and_text_evaluation),
        "acoustic_analysis": RunnableLambda(
            lambda x: extract_acoustic_features(x["audio_array"], x["target_emotion"]) or "분석 불가"
        ),
        "target_emotion": lambda x: x["target_emotion"]
    })
    
    def pipeline(audio_bytes: bytes, target_emotion: str, user_id: int, reset_flag: bool = False) -> Dict:
        try:
            print(f"[PIPELINE] 사용자 {user_id} - 전처리|병렬분석|피드백 체인 실행 시작...")
            
            # 메모리 초기화 처리
            if reset_flag:
                reset_memory(user_id)
            
            # 사용자 메모리 가져오기
            memory = get_or_create_memory(user_id)
            
            # 3단계: 피드백 생성 체인 (메모리 주입)
            def feedback_step(analysis_result: Dict) -> Dict:
                target_emotion_korean = EMOTION_KOREAN.get(analysis_result["target_emotion"], analysis_result["target_emotion"])
                
                feedback_input = {
                    "target_emotion": analysis_result["target_emotion"],
                    "target_emotion_korean": target_emotion_korean,
                    "detected_emotion_top3": analysis_result["detected_emotion"]["top3_formatted"],
                    "text_evaluation": analysis_result["text_evaluation"],
                    "acoustic_analysis": analysis_result["acoustic_analysis"]
                }
                
                feedback_chain = feedback_chain_factory(memory)
                return feedback_chain(feedback_input)
            
            # 4단계: 최종 결과 포맷팅
            def format_result(x: Dict) -> Dict:
                analysis_result = x["analysis"]
                feedback_result = x["feedback"]
                
                detected_emotion = feedback_result.get("detectedEmotion", "NEUTRAL")
                is_correct = (detected_emotion == x["target_emotion"])
                
                result = {
                    "targetEmotion": x["target_emotion"],
                    "detectedEmotion": detected_emotion,
                    "detectedEmotionTop3": analysis_result["detected_emotion"]["top3"],
                    "score": feedback_result.get("score", 50),
                    "feedback": feedback_result.get("feedback", "평가를 완료했습니다."),
                    "isCorrect": is_correct
                }
                
                # 정답 달성 시 메모리 자동 초기화
                if is_correct:
                    print(f"[SUCCESS] 사용자 {user_id} - 목표 감정 달성! 메모리 초기화")
                    reset_memory(user_id)
                
                return result
            
            # 전체 체인 연결: 1단계|2단계|3단계|4단계
            full_chain = (
                preprocess_chain 
                | parallel_analysis_chain 
                | RunnableLambda(lambda analysis: {
                    "analysis": analysis,
                    "feedback": feedback_step(analysis),
                    "target_emotion": analysis["target_emotion"]
                })
                | RunnableLambda(format_result)
            )
            
            # 체인 실행
            result = full_chain.invoke({
                "audio_bytes": audio_bytes,
                "target_emotion": target_emotion
            })
            
            print(f"[SUCCESS] 사용자 {user_id} - 파이프라인 완료 (정답: {result['isCorrect']})")
            return result
            
        except Exception as e:
            print(f"[ERROR] 파이프라인 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            # 예외를 다시 발생시켜 상위 레이어에서 처리하도록 함
            raise
    
    return pipeline

# 파이프라인 인스턴스 생성
voice_emotion_pipeline = build_voice_emotion_pipeline()

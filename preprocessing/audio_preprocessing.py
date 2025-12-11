import librosa
from io import BytesIO
import librosa
import numpy as np
import noisereduce as nr
import webrtcvad
from scipy.signal import lfilter

# 1) 샘플레이트 변환
def load_and_resample_from_bytes(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
    audio_stream = BytesIO(audio_bytes)
    y, _ = librosa.load(audio_stream, sr=sr)
    return y

# 2) 노이즈 제거
def noise_reduce(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    return nr.reduce_noise(y=y, sr=sr)

# 3) RMS 정규화
def rms_normalize(y: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    if len(y) == 0:
        return y

    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-10:
        return y

    current_dbfs = 20 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain = 10 ** (gain_db / 20)

    y_norm = y * gain

    peak = np.max(np.abs(y_norm))
    if peak > 1.0:
        y_norm /= peak

    return y_norm

# 4) 앞뒤 묵음 제거 (WebRTC VAD)
def trim_silence(y: np.ndarray, sr: int = 16000, aggressiveness: int = 2) -> np.ndarray:
    FRAME_MS = 30
    vad = webrtcvad.Vad(aggressiveness)

    frame_len = int(sr * FRAME_MS / 1000)
    num_frames = len(y) // frame_len
    speech_flags = []

    for i in range(num_frames):
        frame = y[i * frame_len:(i + 1) * frame_len]
        pcm = (frame * 32768).astype(np.int16).tobytes()
        speech_flags.append(vad.is_speech(pcm, sr))

    if not any(speech_flags):
        return y

    first_speech = next(i for i, f in enumerate(speech_flags) if f)
    last_speech = len(speech_flags) - 1 - next(i for i, f in enumerate(reversed(speech_flags)) if f)

    start_sample = first_speech * frame_len
    end_sample = (last_speech + 1) * frame_len

    return y[start_sample:end_sample]

# 5) 프리엠퍼시스
def pre_emphasis_lfilter(y: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    if len(y) < 2:
        return y

    b = np.array([1.0, -alpha])
    a = np.array([1.0])
    return lfilter(b, a, y)

# 전처리 파이프라인
def preprocess_audio(
    audio_bytes: bytes,
    sr: int = 16000,
    apply_noise_reduction: bool = True,
    apply_trim: bool = True,
    apply_normalize: bool = True,
    apply_pre_emphasis: bool = True,
) -> np.ndarray:

    # 로드 + 리샘플링
    y = load_and_resample_from_bytes(audio_bytes, sr)

    # 노이즈 제거
    if apply_noise_reduction:
        y = noise_reduce(y, sr)

    # 앞뒤 묵음 제거
    if apply_trim:
        y = trim_silence(y, sr)

    # RMS 정규화
    if apply_normalize:
        y = rms_normalize(y)

    # 프리엠퍼시스
    if apply_pre_emphasis:
        y = pre_emphasis_lfilter(y)

    return y


# # 감정 분석용 전처리 (최소한의 전처리만 적용)
# def preprocess_audio_for_emotion(
#     audio_bytes: bytes,
#     sr: int = 16000,
# ) -> np.ndarray:
#     """
#     감정 분석을 위한 최소한의 전처리.
#     에너지, 피치 등 감정 특징을 보존합니다.
#     """
#     # 로드 + 리샘플링만 적용
#     y = load_and_resample_from_bytes(audio_bytes, sr)
    
#     # 묵음 제거만 적용 (감정 특징 보존)
#     y = trim_silence(y, sr)
    
#     # 정규화하지 않음 (에너지 특징 보존)
#     # 노이즈 제거하지 않음 (음성 질감 보존)
#     # 프리엠퍼시스 적용하지 않음 (스펙트럼 보존)
    
#     return y

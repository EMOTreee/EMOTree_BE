import librosa
import numpy as np
from typing import Dict, Optional, Tuple

SR: int = 16000
HOP_LENGTH: int = 512
FRAME_LENGTH: int = 2048

FMIN: float = 50.0
FMAX: float = 800.0

RMS_SILENCE_THRESH: float = 0.02
MIN_PAUSE_SEC: float = 0.1

FEATURE_NAMES = [
    "mean_f0", "std_f0", "range_f0", "slope_f0",
    "mean_rms", "std_rms", "max_rms",
    "mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3", "mfcc_mean_4",
    "mfcc_mean_5", "mfcc_mean_6", "mfcc_mean_7", "mfcc_mean_8",
    "mfcc_mean_9", "mfcc_mean_10", "mfcc_mean_11", "mfcc_mean_12",
    "mfcc_mean_13",
    "spectral_centroid_mean",
    "speaking_rate", "pause_ratio", "mean_voiced_segment_length",
]


def estimate_f0(y: np.ndarray, sr: int = SR) -> Optional[np.ndarray]:
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=FMIN, fmax=FMAX, sr=sr,
            frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )
        f0 = np.array(f0, dtype=np.float32)
    except Exception:
        f0 = None

    # fallback
    if f0 is None:
        try:
            f0 = librosa.yin(
                y, fmin=FMIN, fmax=FMAX, sr=sr,
                frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
            )
            f0 = np.array(f0, dtype=np.float32)
        except:
            return None

    valid = f0[~np.isnan(f0)]
    if valid.size < 3:
        return None

    return valid


def extract_pitch_features(y: np.ndarray, sr: int = SR) -> Optional[Dict[str, float]]:
    f0 = estimate_f0(y, sr=sr)
    if f0 is None:
        return None

    mean_f0 = float(np.mean(f0))
    std_f0 = float(np.std(f0))
    range_f0 = float(np.max(f0) - np.min(f0))

    t = np.linspace(0, len(f0) * HOP_LENGTH / sr, len(f0), endpoint=False)
    try:
        slope, _ = np.polyfit(t, f0, 1)
    except:
        slope = 0.0

    return {
        "mean_f0": mean_f0,
        "std_f0": std_f0,
        "range_f0": range_f0,
        "slope_f0": float(slope),
    }


def extract_energy_features(y: np.ndarray, sr: int = SR) -> Optional[Dict[str, float]]:
    try:
        rms = librosa.feature.rms(
            y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]
    except:
        return None
    
    if rms.size == 0:
        return None

    return {
        "mean_rms": float(np.mean(rms)),
        "std_rms": float(np.std(rms)),
        "max_rms": float(np.max(rms)),
    }


def extract_spectral_features(y: np.ndarray, sr: int = SR) -> Optional[Dict[str, float]]:
    try:
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13,
            hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH
        )
        cent = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH
        )[0]
    except:
        return None

    features = {}
    for i in range(13):
        features[f"mfcc_mean_{i+1}"] = float(np.mean(mfcc[i]))

    features["spectral_centroid_mean"] = float(np.mean(cent))
    return features


def compute_pause_stats(rms: np.ndarray) -> Tuple[float, float]:
    n = rms.size
    frame_sec = HOP_LENGTH / SR
    total = n * frame_sec

    silence_mask = rms < RMS_SILENCE_THRESH

    pause_sec = 0.0
    voiced_list = []

    in_sil = silence_mask[0]
    seg_start = 0

    for i in range(1, n):
        if silence_mask[i] != in_sil:
            seg_end = i
            seg_len = (seg_end - seg_start) * frame_sec

            if in_sil:
                if seg_len >= MIN_PAUSE_SEC:
                    pause_sec += seg_len
            else:
                voiced_list.append(seg_len)

            seg_start = i
            in_sil = silence_mask[i]

    seg_end = n
    seg_len = (seg_end - seg_start) * frame_sec
    if in_sil:
        if seg_len >= MIN_PAUSE_SEC:
            pause_sec += seg_len
    else:
        voiced_list.append(seg_len)

    pause_ratio = pause_sec / total if total > 0 else 0.0
    mean_voiced = float(np.mean(voiced_list)) if voiced_list else 0.0

    return pause_ratio, mean_voiced


def extract_prosody_features(y: np.ndarray, sr: int = SR) -> Optional[Dict[str, float]]:
    duration = len(y) / sr
    if duration <= 0:
        return None

    try:
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=HOP_LENGTH, units="frames"
        )
        speaking_rate = float(len(onset_frames) / duration)
    except:
        speaking_rate = 0.0

    try:
        rms = librosa.feature.rms(
            y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]
    except:
        return None

    pause_ratio, mean_voiced = compute_pause_stats(rms)

    return {
        "speaking_rate": speaking_rate,
        "pause_ratio": pause_ratio,
        "mean_voiced_segment_length": mean_voiced,
    }

# 모든 음향 특징 추출
def extract_all_features(y: np.ndarray) -> Optional[Dict[str, float]]:

    pitch = extract_pitch_features(y)
    if pitch is None:
        return None

    energy = extract_energy_features(y)
    if energy is None:
        return None

    spec = extract_spectral_features(y)
    if spec is None:
        return None

    prosody = extract_prosody_features(y)
    if prosody is None:
        return None

    feat = {}
    feat.update(pitch)
    feat.update(energy)
    feat.update(spec)
    feat.update(prosody)

    # 모든 필수 특징이 있는지 확인
    for f in FEATURE_NAMES:
        if f not in feat:
            return None

    return feat

# Z-score를 자연어로 해석
def interpret_z_score(z: float) -> str:

    abs_z = abs(z)
    
    if abs_z < 0.5:
        level = "평균 수준"
    elif abs_z < 1.0:
        level = "약간 높음" if z > 0 else "약간 낮음"
    elif abs_z < 1.5:
        level = "다소 높음" if z > 0 else "다소 낮음"
    elif abs_z < 2.0:
        level = "높음" if z > 0 else "낮음"
    else:
        level = "매우 높음" if z > 0 else "매우 낮음"
    
    return f"{level} (z={z:.2f})"

# 추출된 특징과 baseline을 비교하여 z-score 계산 및 자연어 해석

def calculate_z_scores_with_interpretation(
    features: Dict[str, float],
    baseline_mean: Dict[str, float],
    baseline_std: Dict[str, float]
) -> Dict[str, str]:
    
    z_scores = {}
    
    for feature_name in FEATURE_NAMES:
        if feature_name in features and feature_name in baseline_mean and feature_name in baseline_std:
            value = features[feature_name]
            mean = baseline_mean[feature_name]
            std = baseline_std[feature_name]
            
            # 표준편차가 0인 경우 처리
            if std == 0 or std is None:
                z_scores[feature_name] = "비교 불가 (표준편차 0)"
            else:
                z = (value - mean) / std
                z_scores[feature_name] = interpret_z_score(z)
        else:
            z_scores[feature_name] = "데이터 없음"
    
    return z_scores

# 모든 음향 특징을 LLM이 이해하기 쉽도록 그룹별로 구조화하여 포맷
def format_acoustic_features_for_llm(z_scores: Dict[str, str]) -> str:
    lines = []
    
    # 그룹 1: 음높이 특징
    lines.append("## 음높이(Pitch) 특징:")
    pitch_features = ["mean_f0", "std_f0", "range_f0", "slope_f0"]
    for key in pitch_features:
        if key in z_scores:
            lines.append(f"  - {key}: {z_scores[key]}")
    
    # 그룹 2: 에너지 특징
    lines.append("\n## 에너지(Energy) 특징:")
    energy_features = ["mean_rms", "std_rms", "max_rms"]
    for key in energy_features:
        if key in z_scores:
            lines.append(f"  - {key}: {z_scores[key]}")
    
    # 그룹 3: MFCC 특징
    lines.append("\n## 음색(MFCC) 특징:")
    mfcc_features = [f"mfcc_mean_{i}" for i in range(1, 14)]
    for key in mfcc_features:
        if key in z_scores:
            lines.append(f"  - {key}: {z_scores[key]}")
    
    # 그룹 4: 스펙트럼 특징
    lines.append("\n## 스펙트럼 특징:")
    if "spectral_centroid_mean" in z_scores:
        lines.append(f"  - spectral_centroid_mean: {z_scores['spectral_centroid_mean']}")
    
    # 그룹 5: 운율(Prosody) 특징
    lines.append("\n## 운율(Prosody) 특징:")
    prosody_features = ["speaking_rate", "pause_ratio", "mean_voiced_segment_length"]
    for key in prosody_features:
        if key in z_scores:
            lines.append(f"  - {key}: {z_scores[key]}")
    
    return "\n".join(lines) if lines else "음향 특징 분석 결과 없음"

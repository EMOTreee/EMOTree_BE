from email import parser
import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import numpy as np
import pandas as pd
import librosa

from preprocessing.audio_preprocessing import preprocess_audio_for_emotion

SR: int = 16000
HOP_LENGTH: int = 512
FRAME_LENGTH: int = 2048

FMIN: float = 50.0
FMAX: float = 800.0

RMS_SILENCE_THRESH: float = 0.02
MIN_PAUSE_SEC: float = 0.1
MIN_DURATION_SEC: float = 0.2

# 처리할 모든 speaker
SPEAKER = ["F0001", "F0002", "F0003", "F0004",
           "M0001", "M0002", "M0003", "M0004"]

EMOTION_LABELS: List[str] = [
    "ANGER", "ANXIETY", "JOY", "NEUTRAL", "SADNESS", "SURPRISE"
]

# CSV 라벨 정규화용 : 구버전용
EMOTION_CANON_MAP: Dict[str, str] = {e: e for e in EMOTION_LABELS}

# 특징 이름
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


def load_and_preprocess_audio(file_path: str, sr: int = SR) -> Optional[np.ndarray]:
    if not os.path.isfile(file_path):
        print(f"[WARN] Audio not found: {file_path}")
        return None

    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        y = preprocess_audio_for_emotion(audio_bytes=audio_bytes, sr=sr)
        if y is None or len(y) == 0:
            return None

        duration = len(y) / sr
        if duration < MIN_DURATION_SEC:
            return None

        return y.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] preprocess failed for {file_path}: {e}")
        return None


def estimate_f0(y: np.ndarray, sr: int = SR) -> Optional[np.ndarray]:
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=FMIN, fmax=FMAX, sr=sr,
            frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )
        f0 = np.array(f0, dtype=np.float32)
    except Exception:
        f0 = None

    # fallback → yin
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


def extract_pitch_features(y, sr=SR):
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


def extract_energy_features(y, sr=SR):
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


def extract_spectral_features(y, sr=SR):
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


def compute_pause_stats(rms):
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


def extract_prosody_features(y, sr=SR):
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


def extract_all_features(y):
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

    for f in FEATURE_NAMES:
        if f not in feat:
            return None

    return feat


def canonical_emotion(label: str) -> Optional[str]:
    if label is None:
        return None
    key = label.strip().upper()
    return EMOTION_CANON_MAP.get(key)


def calculate_baseline(features_by_emotion: Dict[str, List[np.ndarray]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    baseline = {}
    
    for emo in EMOTION_LABELS:
        vecs = features_by_emotion[emo]
        
        if len(vecs) == 0:
            # 데이터 없으면 None
            print(f"[WARN] {emo} 감정에 대한 데이터가 없습니다.")
            mean_dict = {f: None for f in FEATURE_NAMES}
            std_dict = {f: None for f in FEATURE_NAMES}
        else:
            # 벡터들을 배열로 변환
            arr = np.stack(vecs, axis=0)  # shape: (n_samples, n_features)
            
            # 평균과 표준편차 계산
            means = np.mean(arr, axis=0)
            stds = np.std(arr, axis=0)
            
            # 딕셔너리로 변환
            mean_dict = {FEATURE_NAMES[i]: float(means[i]) for i in range(len(FEATURE_NAMES))}
            std_dict = {FEATURE_NAMES[i]: float(stds[i]) for i in range(len(FEATURE_NAMES))}
            
            print(f"[INFO] {emo}: {len(vecs)}개 샘플로 baseline 계산 완료")
        
        baseline[emo] = {
            "mean": mean_dict,
            "std": std_dict
        }
    
    return baseline


def save_baseline_to_json(baseline: Dict, output_path: str) -> None:
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 기준점 JSON 저장 완료: {output_path}")

# 메인: 모든 speaker 처리
def process_speaker_emotion(
    args: Tuple[str, str, str, str]
) -> Tuple[str, List[np.ndarray]]:
    speaker, emotion, audio_root_dir, scripts_root_dir = args
    feature_vectors = []
    
    # 감정별로 분리된 CSV 파일 경로
    csv_path = os.path.join(scripts_root_dir, f"{speaker}_{emotion}_script.csv")
    
    if not os.path.isfile(csv_path):
        print(f"[WARN] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return (emotion, feature_vectors)

    # CSV 파일 ID를 문자열로 로드
    df = pd.read_csv(csv_path, dtype={"음성파일명": str}, encoding="utf-8-sig")
    total_rows = len(df)
    
    # 임시: 30개만 처리
    df = df.head(30)
    
    print(f"[INFO] {speaker}_{emotion}: 처리 시작 (총 {len(df)}개, 원본: {total_rows}개)")

    for idx, row in enumerate(df.iterrows(), 1):
        _, row = row
        
        # 파일명 zero-padding (6자리)
        file_id = str(row["음성파일명"]).zfill(6)

        # 실제 wav 파일명
        audio_filename = f"{speaker}_{file_id}.wav"

        # 실제 파일 경로 구조 반영
        audio_path = os.path.join(
            audio_root_dir,
            speaker,
            speaker,
            "wav_48000",
            audio_filename
        )

        # 오디오 로드 + 전처리
        y = load_and_preprocess_audio(audio_path)
        if y is None:
            continue

        # 특징 추출
        feat = extract_all_features(y)
        if feat is None:
            continue

        # vector로 저장
        vec = np.array([feat[f] for f in FEATURE_NAMES], dtype=np.float32)
        feature_vectors.append(vec)
        
        # 50개마다 진행상황 출력
        if idx % 10 == 0:
            print(f"[INFO] {speaker}_{emotion}: {idx}/{total_rows} 처리 중... (성공: {len(feature_vectors)})")
    
    success_count = len(feature_vectors)
    print(f"[INFO] {speaker}_{emotion}: 완료 - {success_count}/{total_rows}개 샘플 처리 성공")
    return (emotion, feature_vectors)


def generate_emotion_baseline(
    audio_root_dir: str,
    scripts_root_dir: str = "transscripts",
    output_path: str = "document/emotion_baseline.json",
    n_jobs: int = -1,
):
    features_by_emotion = {e: [] for e in EMOTION_LABELS}

    # 처리할 작업 목록 생성
    tasks = []
    for speaker in SPEAKER:
        for emotion in EMOTION_LABELS:
            tasks.append((speaker, emotion, audio_root_dir, scripts_root_dir))
    
    # 프로세스 수 결정
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    print(f"[INFO] {n_jobs}개의 프로세스로 병렬 처리 시작 (총 {len(tasks)}개 작업)")
    
    # 병렬 처리
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_speaker_emotion, tasks)
    
    # 결과 취합
    for emotion, feature_vectors in results:
        features_by_emotion[emotion].extend(feature_vectors)
    
    print(f"[INFO] 모든 작업 완료. 감정별 샘플 수:")
    for emo in EMOTION_LABELS:
        print(f"  - {emo}: {len(features_by_emotion[emo])}개")

    # baseline 계산
    baseline = calculate_baseline(features_by_emotion)
    
    # JSON 저장
    save_baseline_to_json(baseline, output_path)


if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_root_dir', type=str)
    args = parser.parse_args()
    generate_emotion_baseline(audio_root_dir=args.audio_root_dir)

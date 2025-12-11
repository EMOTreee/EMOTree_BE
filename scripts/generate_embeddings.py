import os
import csv
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from chromadb import PersistentClient

# 기존 Wav2Vec2 임베딩용 import
# from transformers import AutoModel, AutoFeatureExtractor

# SpeechBrain 임베딩용 import
from speechbrain.pretrained import EncoderClassifier
from preprocessing.audio_preprocessing import preprocess_audio

# MODEL_DIR = "models/wav2vec_emotion"
MODEL_DIR = "models/speechbrain_emotion"

SPEAKER = ["F0001", "F0002", "F0003", "F0004",
           "M0001", "M0002", "M0003", "M0004"]
# 기존 Wav2Vec2 임베딩용
# CHROMA_DIR = "vector_store/chroma_emotion_db"
# COLLECTION_NAME = "emotion_audio_embeddings"

CHROMA_DIR = "vector_store/speechbrain_emotion_db_v3"
COLLECTION_NAME = "speechbrain_emotion_audio_embeddings_v3"

_model = None
# 기존 Wav2Vec2 임베딩용
# _feature_extractor = None

# def init_model(model_dir: str) -> None:
#     global _model, _feature_extractor

#     print("모델과 특징 추출기를 불러오는 중...")
#     _feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
#     _model = AutoModel.from_pretrained(model_dir)
#     _model.eval()
#     print("모델 초기화 완료.")

# 모델 로딩: SpeechBrain Emotion 모델
def init_model(model_dir: str) -> None:
    global _model

    print("SpeechBrain Emotion 모델 로드 중...")
    _model = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir=model_dir
    )

   # _model.mods.wav2vec2.model.float()

    print("SpeechBrain 모델 초기화 완료.")


def load_csv_metadata(csv_path: str) -> Dict[str, Dict[str, str]]:
    metadata_map: Dict[str, Dict[str, str]] = {}

    print(f"CSV 파일을 불러옵니다: {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["음성파일명"].strip()
            emotion_main = row["감정대분류"].strip()
            emotion_sub = row["감정소분류"].strip()
            transcription = row["대본(표기식)"].strip()

            metadata_map[file_id] = {
                "emotion_main": emotion_main,
                "emotion_sub": emotion_sub,
                "transcription": transcription,
            }
    print(f"총 {len(metadata_map)}개의 메타데이터 로드됨")
    return metadata_map

def prepare_embedding_tasks(audio_root: str,
               metadata_map: Dict[str, Dict[str, str]],
               speaker: str) -> List[Tuple[str, str, Dict[str, Any]]]:

    tasks: List[Tuple[str, str, Dict[str, Any]]] = []

    print(f"{speaker}의 작업 목록을 준비하는 중...")
    for file_id, meta in metadata_map.items():
        audio_filename = f"{speaker}_{file_id}.wav"
        file_path = os.path.join(
            audio_root,
            speaker,
            speaker,
            "wav_48000",
            audio_filename,
        )

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {file_path}")

        payload = {
            "emotion_main": meta["emotion_main"],
            "emotion_sub": meta["emotion_sub"],
            "transcription": meta["transcription"],
            "audio_filename": audio_filename,
        }

        tasks.append((file_path, audio_filename, payload))
        
    print(f"{speaker} 작업 {len(tasks)}개 준비됨")
    return tasks

def compute_embedding(file_path: str, audio_filename: str,
                      metadata: Dict[str, Any]) -> Tuple[str, List[float], Dict[str, Any]]:
    global _model, _feature_extractor

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    y = preprocess_audio(audio_bytes)
    if y is None or len(y) == 0:
        raise ValueError(f"전처리 후 오디오가 비어 있습니다: {file_path}")

    # 기존 Wav2Vec2 임베딩 계산
    # inputs = _feature_extractor(
    #     y,
    #     sampling_rate=16000,
    #     return_tensors="pt",
    # )

    # with torch.no_grad():
    #     outputs = _model(**inputs)
    #     hidden = outputs.last_hidden_state

    # embedding = hidden.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)

    # ---- SpeechBrain Emotion 임베딩 계산 ----
    signal = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        emb = _model.encode_batch(signal)  # (1, 1, 192)

    embedding = emb.squeeze().cpu().numpy().astype(np.float32)
    print("Embedding dimension:", len(embedding))

    return audio_filename, embedding, metadata

def process_speaker_embeddings(
    speaker: str,
    audio_root: str,
    collection
) -> None:
    print(f"{speaker}의 메타데이터를 불러오는 중...")
    csv_path = f"scripts/{speaker}_script.csv"

    metadata_map = load_csv_metadata(csv_path)

    print(f"{speaker}의 임베딩 작업을 생성하는 중...")
    tasks = prepare_embedding_tasks(audio_root, metadata_map, speaker)

    batch_size = 32
    batch_ids, batch_embeddings, batch_metadatas, batch_docs = [], [], [], []

    for idx, job in enumerate(tasks , start=1):
        audio_id, embedding, meta = compute_embedding(*job)

        # 이미 존재하는 ID는 건너뜀
        existing = collection.get(ids=[audio_id])
        if existing and existing["ids"]:
            print(f"이미 존재하는 ID라 건너뜀 → {audio_id}")
            continue

        print(f"{speaker} - {idx}/{len(tasks)} 처리 중...  ({idx/len(tasks)*100:.2f}%)  파일: {audio_id}")

        batch_ids.append(audio_id)
        batch_embeddings.append(embedding)
        batch_metadatas.append(meta)
        batch_docs.append(meta["transcription"])

        # 배치 저장
        if len(batch_ids) >= batch_size:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_docs,
            )
            print(f"{speaker} - {idx}번째까지 저장됨.")
            batch_ids, batch_embeddings, batch_metadatas, batch_docs = [], [], [], []

    # 마지막 배치
    if batch_ids:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_docs,
        )
        print(f"{speaker} - 마지막 배치 저장 완료.")

def generate_embeddings(audio_root: str, mode: str = "ALL") -> None:

    print(f"ChromaDB 초기화 중... ({CHROMA_DIR})")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(path=CHROMA_DIR)

    # 컬렉션 불러오기 또는 생성
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"기존 컬렉션 사용: {COLLECTION_NAME}")
    except Exception:
        collection = client.create_collection(name=COLLECTION_NAME)
        print(f"새 컬렉션 생성됨: {COLLECTION_NAME}")

    # 스피커 선택 로직
    mode = mode.lower()
    
    if mode.upper() == "all":
        target_speakers = SPEAKER

    elif mode.lower() == "mini":
        target_speakers = ["F0001", "M0001"]

    else:
        # 특정 스피커 이름이 들어온 경우
        speaker = mode.upper()
        if speaker not in SPEAKER:
            raise ValueError(f"존재하지 않는 스피커 이름입니다: {mode}")
        target_speakers = [speaker]

    print(f"선택된 스피커: {target_speakers}")

    init_model(MODEL_DIR)

    for speaker in target_speakers:
        process_speaker_embeddings(speaker, audio_root, collection)

    print("모든 임베딩 작업이 완료되었습니다.")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument("audio_root")
    parser.add_argument("mode", nargs="?", default="mini")

    args = parser.parse_args()
    generate_embeddings(args.audio_root, args.mode)

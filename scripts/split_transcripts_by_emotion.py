import os
import pandas as pd
from pathlib import Path

# 설정
SPEAKERS = ["F0001", "F0002", "F0003", "F0004",
            "M0001", "M0002", "M0003", "M0004"]

# 감정 매핑: 원본 감정 -> 변환된 감정
EMOTION_MAPPING = {
    "JOY": "JOY",
    "ANGRY": "ANGER",
    "SAD": "SADNESS",
    "SURPRISE": "SURPRISE",
    "ANXIOUS": "ANXIETY",
    "NEUTRAL": "NEUTRAL"
}

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).parent.parent
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"


def split_transcript_by_emotion(speaker: str):
    input_file = TRANSCRIPTS_DIR / f"{speaker}_script.csv"
    
    if not input_file.exists():
        print(f"파일을 찾을 수 없습니다: {input_file}")
        return
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(input_file)
        print(f"{speaker}_script.csv 읽기 완료 (총 {len(df)}개 행)")
    except Exception as e:
        print(f"{speaker}_script.csv 읽기 실패: {e}")
        return
    
    # 감정 대분류별로 그룹화하여 파일 생성
    emotion_counts = {}
    
    for original_emotion, mapped_emotion in EMOTION_MAPPING.items():
        # 해당 감정에 해당하는 행만 필터링
        emotion_df = df[df['감정대분류'] == original_emotion].copy()
        
        if len(emotion_df) == 0:
            print(f"{original_emotion} 감정 데이터 없음")
            continue
        
        # 감정대분류 컬럼 값을 변환된 감정명으로 변경
        emotion_df['감정대분류'] = mapped_emotion
        
        # 출력 파일 생성
        output_file = TRANSCRIPTS_DIR / f"{speaker}_{mapped_emotion}_script.csv"
        
        try:
            emotion_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            emotion_counts[mapped_emotion] = len(emotion_df)
            print(f"{speaker}_{mapped_emotion}_script.csv 생성 ({len(emotion_df)}개 행)")
        except Exception as e:
            print(f"{speaker}_{mapped_emotion}_script.csv 생성 실패: {e}")
    
    # 결과 요약
    if emotion_counts:
        total = sum(emotion_counts.values())
        print(f"  📊 {speaker} 총 {total}개 행 처리 완료")


def main():
    print("감정 대분류별 Transcript CSV 분할 작업 시작")
    
    for speaker in SPEAKERS:
        print(f"[{speaker}] 처리 중...")
        split_transcript_by_emotion(speaker)
        print()
    
    print("csv 분할 작업 완료")


if __name__ == "__main__":
    main()

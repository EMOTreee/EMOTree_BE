import csv
import re
import os
import argparse
from typing import List, Dict, Optional

# 대본에서 불필요한 기호 제거 함수
def clean_transcript(transcript: str) -> str:
    # 1. 모든 '||', '|||', '||||' 기호 제거
    cleaned = re.sub(r'\|{2,4}', '', transcript)
    
    # 2. 'M', 'HL', 'LH', 'LHL' 제거
    cleaned = re.sub(r'(LHL|HL|LH|M)', '', cleaned)

    # 3. 앞뒤 공백 및 여러 개의 공백을 하나로 정리
    cleaned = ' '.join(cleaned.split()).strip()
    return cleaned

# 감정 태깅 텍스트 파일 처리 함수
def process_emotion_file(file_path: str) -> Optional[List[Dict[str, str]]]:
    # 파일 존재 여부 확인 후 읽기
    if not os.path.isfile(file_path):
        print(f"오류: 파일을 찾을 수 없습니다 → {file_path}")
        return None
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    except Exception as e:
        print(f"오류: 파일을 읽는 중 문제가 발생했습니다 → {file_path}")
        print(f"세부 내용: {e}")
        return None

    data_list: List[Dict[str, str]] = []
    
    # 4줄씩 하나의 데이터 블록 처리
    for i in range(0, len(lines), 4):
        if i + 2 >= len(lines):
            # 완전한 데이터 블록이 아닐 경우 스킵합
            continue

        # 첫 번째 줄: 음성파일명 감정대분류 감정소분류
        line1 = lines[i].strip()
        parts = line1.split(' ')
        
        if len(parts) < 3:
            print(f"형식이 올바르지 않아 건너뜁니다: {line1}")
            continue
            
        filename = parts[0].split('_')[-1]
        emotion_main = parts[1]
        
        # 감정소분류는 '#'을 제거하고 추출
        emotion_sub_raw = ' '.join(parts[2:])
        emotion_sub = emotion_sub_raw.lstrip('#').strip()

        # 두 번째 줄: 대본(표기식)을 전처리
        transcript_raw = lines[i+1].strip()
        transcript_cleaned = clean_transcript(transcript_raw)
        
        # 데이터를 리스트에 추가
        data_list.append({
            '음성파일명': filename,
            '감정대분류': emotion_main,
            '감정소분류': emotion_sub,
            '대본(표기식)': transcript_cleaned
        })
        
    return data_list

def process_all_speakers(base_dir: str):
    SPEAKERS = ["F0001", "F0002", "F0003", "F0004",
                "M0001", "M0002", "M0003", "M0004"]

    for spk in SPEAKERS:
        input_file_path = os.path.join(base_dir, spk, spk, "script.txt")
        output_filename = f"transcripts/{spk}_script.csv"

        print(f"\n처리 중: {input_file_path}")

        final_data_list = process_emotion_file(input_file_path)
        if not final_data_list:
            print(f"{spk}에 유효한 데이터가 없습니다. 건너뜁니다.")
            continue

        fieldnames = ['음성파일명', '감정대분류', '감정소분류', '대본(표기식)']

        try:
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            with open(output_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(final_data_list)

            print(f"{spk} 변환 완료 → {output_filename}")

        except Exception as e:
            print(f"CSV 파일 저장 중 오류가 발생했습니다: {output_filename}")
            print(f"오류 내용: {e}")

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=str)
    args = parser.parse_args()
    print(f"\n=== [시작] txt → CSV 변환 작업 실행: base_dir='{args.base_dir}' ===")
    process_all_speakers(args.base_dir)
    print("\n=== [완료] txt → CSV 변환 작업이 정상적으로 종료되었습니다. ===")


if __name__ == "__main__":
    main()
from huggingface_hub import snapshot_download

def download_ehcalabres_emotion():
    snapshot_download(
        repo_id="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        local_dir="./models/ehcalabres_emotion",
        ignore_patterns=["*.msgpack"]
    )
    print("다운로드 완료: ./models/ehcalabres_emotion")

if __name__ == "__main__":
    download_ehcalabres_emotion()
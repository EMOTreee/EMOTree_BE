### 1. 가상환경 생성 및 활성화
```powershell
# 가상환경 생성
python -m venv .venv
# 가상환경 활성화(powerShell 기준)
.venv\Scripts\Activate.ps1
```

### 2. 의존성 다운로드
```powershell
# pip 업그레이드
python -m pip install --upgrade pip
# 의존성 다운로드
pip install -r requirements.txt
```

### 3. Wav2Vec2 감정 인식 모델 다운로드
```powershell
python scripts/download_models.py
```

### 4. [FFmpeg](https://ffmpeg.org/download.html) 설치 (Windows)

1. Windows 아이콘 클릭
2. "Windows builds from gyan.dev" 선택
3. release builds → ffmpeg-release-full.7z 다운로드
4. ffmpeg로 폴더명 변경하고 c:/로 이동

### 5. 환경변수 설정
```env
# .env
SERVER_URL=http://localhost:8000

OPENAI_API_KEY=
OPENAI_ORGANIZATION_ID=
KAKAO_CLIENT_ID=
KAKAO_REDIRECT_URI=http://localhost:8000/auth/kakao/login2
MYSQL_PASSWORD=
JWT_SECRET_KEY=
```

### 6. FastAPI 서버 실행
```powershell
fastapi dev ./app/main.py
```

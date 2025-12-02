from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, user, emotion_image_router
from app.routers.emotion_empathy_router import router as empathy_router # 감정 공감 라우터
from app.db.init_db import init_db
from contextlib import asynccontextmanager
from dotenv import load_dotenv


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    init_db()
    yield
    # shutdown 

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://emotreee.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(auth.router)
app.include_router(user.router)
app.include_router(emotion_image_router.router)
app.include_router(empathy_router)

# 루트 라우트 추가 -> 테스트용
@app.get("/")
async def root():
    return {"message": "FastAPI server is running"}

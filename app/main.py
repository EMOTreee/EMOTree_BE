from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, user, emotion_image_router, emotion_empathy_router
from app.db.init_db import init_db
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from app.routers import emotion_quiz_router
from fastapi.staticfiles import StaticFiles
from app.core.config import settings

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
    allow_origins=["http://localhost:5173",
                   "https://emotreee.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.mount("/static", StaticFiles(directory=settings.STATIC_ROOT), name="static")

app.include_router(auth.router)
app.include_router(user.router)
app.include_router(emotion_image_router.router)
app.include_router(emotion_quiz_router.router)
app.include_router(emotion_empathy_router.router)

# 루트 라우트 추가 -> 테스트용
@app.get("/")
async def root():
    return {"message": "FastAPI server is running"}

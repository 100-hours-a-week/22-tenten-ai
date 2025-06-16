import argparse
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints.youtube_summary_router import router as youtube_router
from api.endpoints.bot_posts_router import router as bot_post_router
from api.endpoints.bot_recomments_router import router as bot_recomment_router
from api.endpoints.bot_chats_router import router as bot_chat_router
from models.model_loader import ModelLoader 

from api.endpoints.discord_webhook_router import router as discord_router ## for Discord Webhook
from utils.logger_discord import setup_logging
from utils.exception_handler import register_exception_handlers

# CLI 인자 파싱 함수 추가
def parse_args():
    parser = argparse.ArgumentParser(description="텐텐 GPU 사용 모드 선택")
    parser.add_argument(
        "--mode",
        choices=["colab", "gcp", "api-dev", "api-prod"],
        default="colab",
        help="LLM inference 모드 선택 (colab: Ngrok/Colab, gcp: GCP 서버 직접 추론, api-dev: gemini 2.0 flash api 사용 및 로컬 환경변수 사용, api-prod: gemini 2.0 flash api 사용 및 GCP 환경변수 사용)"
    )
    return parser.parse_args()

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="텐텐 AI API",
    description="kakaobase 플랫폼을 위한 AI 기능",
    version="1.0.0"
)

# === [수정] 서버 시작 시 ModelLoader를 FastAPI state에 싱글턴으로 등록 ===
# 서버 실행 모드(LLM_MODE)는 CLI 인자에서 받아오므로, 아래는 임시 기본값
llm_mode = os.environ.get("LLM_MODE", "colab")
app.state.model = ModelLoader(mode=llm_mode)
# 이 코드는 서버 프로세스가 시작될 때 단 한 번만 실행되어,
# app.state.model에 ModelLoader 인스턴스(즉, vLLM 모델)가 로드됩니다.
# 이후 모든 요청에서 이 인스턴스를 재사용하므로, 매번 모델을 다시 로드하지 않습니다.

# origins 설정
origins = [
    "http://www.kakaobase.com",
    "https://www.kakaobase.com",
    "http://localhost:8080",
    "http://localhost:3000"
]

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 디스코드 웹훅 로깅 설정: 파일 + 콘솔 + Discord
setup_logging("ai-log.log")
# 디스코드 웹훅 예외 핸들러 등록
register_exception_handlers(app)


# 라우터 등록
app.include_router(youtube_router, prefix="/posts/youtube", tags=["youtube"])
app.include_router(bot_post_router, prefix="/posts/bot", tags=["bot-posts"])
app.include_router(bot_recomment_router, prefix="/recomments/bot", tags=["bot-recomments"])
app.include_router(bot_chat_router, prefix="/chats/bot", tags=["bot-chats"])
app.include_router(discord_router, prefix="/error_log", tags=["discord-webhook"]) # Discord Webhook router

# 서버 구동을 위한 설정
if __name__ == "__main__":
    args = parse_args()
    os.environ["LLM_MODE"] = args.mode

    reload_flag = True
    if os.environ["LLM_MODE"] in ["gcp", "api-dev", "api-prod"]:
        reload_flag = False

    print(f"실행 모드: {args.mode}, reload : {reload_flag}")
    

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_flag   ## 개발 환경에서만 사용
    )
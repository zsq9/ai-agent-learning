import os
import logging
from datetime import datetime
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel
from dotenv import load_dotenv



# 加载环境变量
load_dotenv()

# 日志配置（企业级标准）
logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL","INFO")),
    format="%(asctime)s | %(levelname)-6s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(title=os.getenv("APP_NAME"))

class ChatRequest(BaseModel):
    user_id:str
    question:str

class ApiResponse(BaseModel):
    code: int=200
    msg: str="success"
    data: dict | None = None
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.get("/health")
async def health():
    logger.info("健康检查接口被调用")
    return ApiResponse(data={"status":"ok"})

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        logger.info(f"用户ID: {req.user_id} | 提问: {req.question}")

        if not req.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")

        return ApiResponse(
            data={
                "user_id": req.user_id,
                "question": req.question,
                "answer": "已收到你的问题,AI 引擎待接入"
            }
        )

    except Exception as e:
        logger.error(f"接口异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    return response
    

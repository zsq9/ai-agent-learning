import os
import logging
import httpx
from datetime import datetime
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel
from dotenv import load_dotenv


#加载环境变量
load_dotenv()


#日志配置
logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_ LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)-6s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(title=os.getenv("APP_NAME"))

class ChatRequest(BaseModel):
    user_id: str
    question: str

class ApiResponse(BaseModel):
    code: int = 200
    msg: str = "success"
    data: dict | None = None
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def call_llm(prompt: str) -> str:
    api_url = "http://127.0.0.1:11434/v1/chat/completions"
    # 本地模型不需要Key，留空即可
    api_key = "empty"
    model = "qwen2.5:7b"

    if not api_key or api_key == "your_api_key_here":
        raise HTTPException(status_code=500,detail="请先配置LLM_API_KEY")
    headers = {
        "Authorization": f"Bearer{api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages":[
            {"role":"system","content":"你是一个专业AI助手,回答简洁准确"},
            {"role":"user","content":prompt}
        ],
        "temperature":0.7
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(api_url, headers=headers, json=payload)
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()

    except httpx.TimeoutException:
        logger.error("大模型接口超时")
        raise HTTPException(status_code=504, detail="AI 服务超时")
    except Exception as e:
        logger.error(f"LLM调用异常: {str(e)}")
        raise HTTPException(status_code=500, detail="AI 服务异常")

# 接口
@app.get("/health", response_model=ApiResponse)
async def health():
    return ApiResponse(data={"status": "ok"})

@app.post("/api/chat", response_model=ApiResponse)
async def chat(req: ChatRequest):
    try:
        logger.info(f"用户 {req.user_id} 提问: {req.question}")

        if not req.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")

        answer = await call_llm(req.question)

        return ApiResponse(data={
            "user_id": req.user_id,
            "question": req.question,
            "answer": answer
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"接口异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误")

# 中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    return response
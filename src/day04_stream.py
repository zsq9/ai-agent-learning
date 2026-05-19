import httpx
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging

#---日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

#---FastAPI实例
app = FastAPI(title="Day04 流式输出接口",version="1.0")

#---请求体模型
class ChatRequest(BaseModel):
    prompt:str
    model: str="qwen2.5:7b"

#---核心异步生成器
async def llm_stream_generator(prompt:str,model:str):
    """
    流式调用本地 Ollama 模型，逐字返回内容
    异步生成器:async def + yield 组合
    """
    api_url = "http://127.0.0.1:11434/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer empty"  # Ollama 不校验 key
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个专业AI助手,回答简洁有条理"},
            {"role": "user", "content": prompt}
        ],
        "stream": True,  # 开启流式输出，关键参数
        "temperature": 0.7,
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            #发送请求并以流的方式接收响应
            async with client.stream(
                "POST",
                api_url,
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                logger.info("流式连接已建立，开始接收数据")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str=="[DONE]":
                            yield f"data:[DONE]\n\n"
                            break
                        try:
                            data=json.loads(data_str)
                            token = data["choices"][0]["delta"].get("content","")
                            if token:
                                # 严格遵循 SSE 格式
                                yield f"data: {json.dumps({'token': token},ensure_ascii=False)}\n\n"
                        except json.JSONDecodeError:
                            continue

    except httpx.TimeoutException:
        logger.error("流式请求超时")
        yield f"data: {json.dumps({'error': '请求超时'})}\n\n"
    except Exception as e:
        logger.error(f"流式异常: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# -------------------------- 对外接口 --------------------------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口
    返回: StreamingResponse,媒体类型 text/event-stream
    """
    return StreamingResponse(
        llm_stream_generator(req.prompt, req.model),
        media_type="text/event-stream"  # SSE 必须的类型
    )

# -------------------------- 启动 --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)       

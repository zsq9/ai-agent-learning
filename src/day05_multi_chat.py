# day05_multi_chat_final.py
# Day05 多轮对话 + 上下文记忆 + 企业级标准代码
# 逐行精讲 + 样例输入输出 + 启动模块完整版

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json

# ==========================================
# 1. 创建 FastAPI 应用实例
# 作用：整个后端服务的入口
# ==========================================
app = FastAPI(title="Day05 多轮对话（企业级）")

# ==========================================
# 2. 全局变量：存储对话历史（记忆功能）
# 样例输入输出：
# 初始：[]
# 一轮后：
# [
#    {"role":"user","content":"你好"},
#    {"role":"assistant","content":"你好！"}
# ]
# ==========================================
history = []

# ==========================================
# 3. 请求体模型：前端传给后端的数据格式
# 样例输入：{ "query": "我刚才说什么了？" }
# ==========================================
class ChatRequest(BaseModel):
    query: str

# ==========================================
# 4. 核心接口：多轮流式对话
# ==========================================
@app.post("/chat/multi-round")
async def chat_multi_round(req: ChatRequest):
    # 把用户的新问题加入历史
    history.append({"role": "user", "content": req.query})

    async def response_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url="http://127.0.0.1:11434/v1/chat/completions",
                json={
                    "model": "qwen2.5:7b",
                    "messages": history,
                    "stream": True
                },
                timeout=None  # 永不超时
            ) as response:

                # ==========================================
                # 检查 Ollama 是否正常响应
                # 异常：服务挂了、模型不存在、地址错误
                # ==========================================
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    raise HTTPException(status_code=500, detail="Ollama 服务异常")

                full_answer = ""

                async for line in response.aiter_lines():
                    # 过滤空行，防止 JSON 解析崩溃
                    if not line:
                        continue

                    # 去掉 SSE 协议的 data: 前缀
                    if line.startswith("data: "):
                        line = line[6:]

                    # 结束标志
                    if line.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                        # 企业标准取值：choices[0].delta.content
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")

                        if content:
                            full_answer += content
                            yield f"data: {content}\n\n"
                    except Exception:
                        continue

                # 把 AI 回答存入历史，实现记忆
                history.append({"role": "assistant", "content": full_answer})

    # ==========================================
    # 返回流式响应
    # media_type="text/event-stream"：SSE 标准格式
    # 作用：前端逐字显示（打字机效果）
    # ==========================================
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream"
    )

# ==========================================
# 清空历史对话
# ==========================================
@app.post("/clear/history")
async def clear_history():
    global history
    history.clear()
    return {"status": "success", "msg": "历史已清空"}

# ==========================================
# ✅ 你要的：启动模块代码（终于补上！）
# 作用：直接运行这个文件即可启动服务
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="day05_multi_chat:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
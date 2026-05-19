# day05_multi_chat_final_plus.py
# Day05 + 历史长度限制 + 多用户隔离（企业增强版）

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json

app = FastAPI(title="Day05 多轮对话增强版")

# 多用户记忆：key=userId, value=对话列表
history_map = {}
# 紧跟解释：存储每个用户独立的对话历史
# 样例输入：user001发消息
# 样例结果：history_map["user001"] = [用户消息]

class ChatRequest(BaseModel):
    query: str
    userId: str
    # 紧跟解释：用户ID，用于区分不同人的记忆
    # 样例输入：{"query":"你好","userId":"user001"}

@app.post("/chat/multi-round")
async def chat_multi_round(req: ChatRequest):
    # 取出当前用户的历史，没有则为空列表
    history = history_map.get(req.userId, [])
    # 紧跟解释：根据userId拿到自己的记忆

    # 把用户问题加入历史
    history.append({"role": "user", "content": req.query})

    # 限制最多保留最近10条对话
    history = history[-10:]
    # 紧跟解释：防止对话过长，超过模型token限制
    # 样例：原长度15 → 处理后10

    async def response_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/v1/chat/completions",
                json={
                    "model": "qwen2.5:7b",
                    "messages": history,
                    "stream": True
                },
                timeout=None
            ) as response:

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    raise HTTPException(status_code=500, detail="Ollama 异常")

                full_answer = ""

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            full_answer += content
                            yield f"data: {content}\n\n"
                    except Exception:
                        continue

                # 把AI回复存入当前用户历史
                history.append({"role": "assistant", "content": full_answer})
                # 写回全局记忆字典
                history_map[req.userId] = history

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream"
    )

# 清空某个用户的历史
@app.post("/clear/history")
async def clear_history(userId: str):
    if userId in history_map:
        del history_map[userId]
    return {"status": "ok", "msg": f"用户 {userId} 历史已清空"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "day05_multi_chat_final_plus:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
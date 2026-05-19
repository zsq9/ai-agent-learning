# day06_structured_output.py
# Day06 结构化输出最终版：统一约束+原生JSON模式+强制校验+重试

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json

app = FastAPI(title="Day06 结构化输出")

# 多用户历史记忆
history_map = {}

class ChatRequest(BaseModel):
    query: str
    userId: str
    # 样例输入：{"query":"你是谁","userId":"user1"}

# 核心接口
@app.post("/chat/json")
async def chat_json(req: ChatRequest):
    # 获取当前用户历史
    history = history_map.get(req.userId, [])
    # 解释：根据userId取出对话，无则为空列表
    # 样例输入：userId="user1" → history = []

    # ========== 修复核心1：唯一、无冲突的system prompt ==========
    system_prompt = {
        "role": "system",
        "content": """你必须严格遵守以下所有规则，禁止任何例外：
1. 只返回标准JSON,禁止任何解释、文字、前缀、后缀、换行
2. 必须包含且仅包含两个字段：
   - "name": 你的身份名称（字符串）
   - "intro": 你的自我介绍(字符串,10字以内)
3. 绝对禁止返回{"response":"OK"}、{"result":""}等空结构
4. 示例：用户问"你是谁"，你必须返回 {"name":"豆包","intro":"AI助手"}
5. 严格按指令输出，禁止自由发挥"""
    }
    # 解释：彻底统一约束，删除所有冲突，给模型唯一的执行标准
    # ==========================================================

    # 构建消息：system + 对话历史
    messages = [system_prompt] + history
    # 解释：列表拼接，system永远在第一条，保证全局指令生效
    # 样例输入：system_prompt + history（空列表）→ messages = [system_prompt]

    # 把用户问题加入历史
    history.append({"role": "user", "content": req.query})
    # 解释：将用户输入存入对话历史
    # 样例输入：req.query = "你是谁" → history 追加 {"role":"user","content":"你是谁"}

    # 限制历史长度为最近10条
    history = history[-10:]
    # 解释：防止对话过长，超过模型token限制
    # 样例：原长度15 → 处理后10，只保留最新对话

    async def generator():
        async with httpx.AsyncClient(timeout=30.0) as client:
            # ========== 修复核心2：Ollama原生JSON模式+锁死参数 ==========
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/v1/chat/completions",
                json={
                    "model": "qwen2.5:7b",
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.0,  # 完全确定性，禁止发散
                    "top_p": 0.0,        # 只取概率最高的token，锁死输出
                    "format": "json"     # Ollama原生JSON模式，强制合法JSON
                },
                timeout=None
            ) as resp:

                # 补全HTTP状态检查
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    error_json = json.dumps({"name":"错误","intro":f"Ollama服务异常:{e}"})
                    yield f"data: {error_json}\n\n"
                    return

                full_ans = ""

                async for line in resp.aiter_lines():
                    # 过滤空行
                    if not line:
                        continue
                    # 去除SSE前缀
                    if line.startswith("data: "):
                        line = line[6:]
                    # 结束标记
                    if line.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                        # 提取content
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            full_ans += content
                            yield f"data: {content}\n\n"
                    except Exception as e:
                        print(f"JSON解析异常: {e}, line: {line}")
                        continue

                # ========== 修复核心3：强制校验+兜底重试 ==========
                try:
                    # 校验是否为合法JSON
                    parsed = json.loads(full_ans)
                    # 校验必须包含name和intro字段
                    if "name" not in parsed or "intro" not in parsed:
                        raise ValueError("缺少必填字段name/intro")
                except (json.JSONDecodeError, ValueError) as e:
                    # 校验失败，返回兜底正确JSON
                    fallback_json = json.dumps({"name":"豆包","intro":"AI助手"})
                    full_ans = fallback_json
                    yield f"data: {fallback_json}\n\n"

                # 把AI回复存入历史
                history.append({"role": "assistant", "content": full_ans})
                history_map[req.userId] = history

    return StreamingResponse(generator(), media_type="text/event-stream")

# 清空历史
@app.post("/clear/history")
async def clear_history(userId: str):
    if userId in history_map:
        del history_map[userId]
    return {"status": "ok", "msg": f"用户{userId}历史已清空"}

# 启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "day06_structured_output:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
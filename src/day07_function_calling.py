from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import json

app = FastAPI(title="Day07 AI工具调用 - 计算器")

# 模拟一个计算器工具（真正的函数）
def calculator(a: float, b: float, op: str):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b if b != 0 else "除零错误"
    else:
        return "不支持的操作"

# 请求结构
class ChatRequest(BaseModel):
    query: str
    userId: str

history_map = {}

# 接口
@app.post("/chat/agent")
async def chat_agent(req: ChatRequest):
    history = history_map.get(req.userId, [])

    system = {
        "role": "system",
        "content": """
你是一个智能助手。
如果用户的问题需要计算,你必须返回如下JSON格式:
{"tool": "calculator", "params": {"a": 数字, "b": 数字, "op": "+-*/"}}
如果不需要计算,直接回答,不要JSON。
不要多余文字。
"""
    }

    messages = [system] + history
    messages.append({"role": "user", "content": req.query})

    # 非流式，方便解析
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://127.0.0.1:11434/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": messages,
                "stream": False,
                "temperature": 0.1
            }
        )

    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()

    # ====================== Day07 核心 ======================
    # 后端判断：AI 是否想调用工具
    tool_result = None
    answer = content
    try:
    # 尝试解析 JSON（关键：必须先 try）
        tool_call = json.loads(content)
    
    # 确认是工具调用
        if isinstance(tool_call, dict) and tool_call.get("tool") == "calculator":
            params = tool_call.get("params", {})
            a = params.get("a")
            b = params.get("b")
            op = params.get("op")
            
            # 参数合法性检查
            if all([isinstance(x, (int, float)) for x in [a, b]]):
                tool_result = calculator(a, b, op)
                answer = f"计算结果：{tool_result}"

    except json.JSONDecodeError:
    # 解析失败 → 不是工具调用，直接使用模型自然语言回答
        answer = content
    except Exception as e:
    # 兜底：参数错误/除零等
        answer = f"工具调用失败：{str(e)}"
    # =======================================================

    # 保存历史
    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": answer})
    history_map[req.userId] = history

    return {
        "ai_raw": content,
        "tool_called": tool_result is not None,
        "answer": answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("day07_function_calling:app", host="127.0.0.1", port=8000,reload=True)
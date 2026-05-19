from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json
import time
from datetime import datetime

app = FastAPI(title="Day08 多工具调度 - 计算器+时间+天气")

# ====================== 定义所有工具函数 ======================
# 工具1：计算器
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
        return "不支持的操作：仅支持 +-*/"

# 工具2：获取当前时间
def get_current_time():
    # 返回格式化时间：2026-04-13 11:30:00
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 工具3：查天气（模拟接口，真实场景替换为天气API）
def get_weather(city: str):
    # 模拟不同城市的天气数据
    weather_data = {
        "北京": "晴，温度 18℃，风力 2级",
        "上海": "多云，温度 22℃，风力 3级",
        "广州": "雷阵雨，温度 28℃，风力 4级",
        "深圳": "阴，温度 26℃，风力 2级"
    }
    return weather_data.get(city, f"暂未查询到{city}的天气信息")

# ====================== 工具映射（核心：把工具名和函数绑定） ======================
tool_mapping = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "get_weather": get_weather
}

# 请求结构
class ChatRequest(BaseModel):
    query: str
    userId: str

# 多用户历史
history_map = {}

# ====================== 核心接口：多工具调度 ======================
@app.post("/chat/multi_tool")
async def chat_multi_tool(req: ChatRequest):
    history = history_map.get(req.userId, [])

    # 关键：强化System Prompt，让AI能识别不同工具并返回对应JSON
    system_prompt = {
        "role": "system",
        "content": """
你是一个智能工具调度助手，严格遵守以下规则：
1. 根据用户问题，判断需要调用的工具，返回纯JSON格式，无任何多余文字：
   - 计算问题 → {"tool": "calculator", "params": {"a": 数字, "b": 数字, "op": "+-*/"}}
   - 查时间 → {"tool": "get_current_time", "params": {}}
   - 查天气 → {"tool": "get_weather", "params": {"city": "城市名"}}
2. 如果不需要调用工具，直接用自然语言回答，不要返回JSON。
3. 绝对禁止输出多余文字、解释、换行、空格！
"""
    }

    # 构建消息：系统指令 + 历史 + 当前问题
    messages = [system_prompt] + history
    messages.append({"role": "user", "content": req.query})

    # 调用Ollama（非流式，方便解析）
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "http://127.0.0.1:11434/v1/chat/completions",
                json={
                    "model": "qwen2.5:7b",
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.0,  # 锁死输出，避免幻觉
                    "format": "json"      # 强制JSON格式，减少解析错误
                },
                timeout=30.0
            )
            resp.raise_for_status()  # 检查HTTP请求是否成功
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"调用Ollama失败：{str(e)}")

    # 解析Ollama返回结果
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()

    # ====================== 多工具调度核心逻辑 ======================
    final_answer = content  # 默认答案：AI原始返回
    tool_called = None      # 记录调用的工具名
    tool_result = None      # 记录工具执行结果

    try:
        # 第一步：解析AI返回的JSON
        tool_call = json.loads(content)
        
        # 第二步：校验是否为合法的工具调用格式
        if isinstance(tool_call, dict) and "tool" in tool_call:
            tool_name = tool_call["tool"]
            tool_params = tool_call.get("params", {})

            # 第三步：检查工具是否存在
            if tool_name in tool_mapping:
                tool_func = tool_mapping[tool_name]  # 获取对应的工具函数
                tool_called = tool_name

                # 第四步：根据工具类型处理参数并调用
                if tool_name == "calculator":
                    # 计算器参数校验
                    a = tool_params.get("a")
                    b = tool_params.get("b")
                    op = tool_params.get("op", "+")
                    if all([isinstance(x, (int, float)) for x in [a, b]]):
                        tool_result = tool_func(a, b, op)
                        final_answer = f"【计算器】{a} {op} {b} = {tool_result}"
                    else:
                        final_answer = f"【参数错误】计算器需要数字参数，当前：a={a}, b={b}"

                elif tool_name == "get_current_time":
                    # 查时间无需参数
                    tool_result = tool_func()
                    final_answer = f"【当前时间】{tool_result}"

                elif tool_name == "get_weather":
                    # 查天气参数校验
                    city = tool_params.get("city", "")
                    if city:
                        tool_result = tool_func(city)
                        final_answer = f"【{city}天气】{tool_result}"
                    else:
                        final_answer = "【参数错误】查天气需要传入城市名，例如：{\"tool\":\"get_weather\",\"params\":{\"city\":\"北京\"}}"

        # 如果不是工具调用，final_answer保持AI原始回答
    except json.JSONDecodeError:
        # 解析失败，说明是自然语言回答，无需处理
        pass
    except Exception as e:
        # 其他异常兜底
        final_answer = f"【工具调用失败】{str(e)}"

    # ====================== 保存对话历史 ======================
    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": final_answer})
    history_map[req.userId] = history[-10:]  # 只保留最近10轮

    # ====================== 返回最终结果 ======================
    return {
        "user_query": req.query,
        "ai_raw_response": content,
        "tool_called": tool_called,
        "tool_result": tool_result,
        "final_answer": final_answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("day08_multi_tool_calling:app", host="127.0.0.1", port=8000, reload=True)
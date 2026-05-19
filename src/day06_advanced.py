from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json

app = FastAPI(title="Day06 进阶：后端解析AI的JSON（修复版）")

# 多用户记忆：字典 {userId: [历史列表]}
history_map = {}


class ChatRequest(BaseModel):
    query: str
    userId: str


@app.post("/chat/advanced")
async def chat_advanced(req: ChatRequest):
    # 取出当前用户的对话历史
    history = history_map.get(req.userId, [])

    # 【修复核心1】强化system prompt，给模型明确示例，彻底杜绝空输出
    system_prompt = {
        "role": "system",
        "content": """你必须严格按照以下要求返回，禁止任何例外：
1. 只返回标准JSON,禁止任何解释、文字、前缀、后缀、换行
2. 必须包含且仅包含两个字段：
   - "answer": 问题的答案（字符串类型）
   - "need_tool": true 或 false(布尔类型)
3. 规则：
   - 简单问题（计算、常识、自我介绍）→ need_tool = false
   - 实时信息、复杂查询 → need_tool = true
4. 示例：用户问"1+1等于几"，你必须返回 {"answer":"2","need_tool":false}
5. 绝对禁止返回空值、空JSON、非JSON内容"""
    }

    messages = [system_prompt] + history

    # 把用户问题加入历史
    history.append({"role": "user", "content": req.query})
    history = history[-10:]  # 只保留最近10条

    # 【修复核心2】移除format: "json"，避免与prompt冲突，用低温度锁死输出
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url="http://127.0.0.1:11434/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": messages,
                "stream": False,
                "temperature": 0.0,  # 完全确定性，禁止发散
                "top_p": 0.0         # 只取概率最高的token，锁死输出
            },
            timeout=None
        )

    # 第一层解析：Ollama 接口返回 → 字典
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()  # 【修复核心3】strip()去除首尾空白
        print(f"AI返回原始内容: {content}")  # 调试用，可删除
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama接口解析失败: {str(e)}")

    # 【修复核心4】空内容兜底，避免解析崩溃
    if not content:
        content = '{"answer":"模型未返回有效内容","need_tool":false}'

    # 第二层解析：AI 返回的业务 JSON → 字典
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        # 解析失败兜底，返回标准结构，避免接口崩溃
        content = '{"answer":"JSON格式错误","need_tool":false}'
        result = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"业务JSON解析失败: {str(e)}")

    # 后端根据 need_tool 做逻辑判断
    answer = result.get("answer", "无答案")
    need_tool = result.get("need_tool", False)

    if need_tool:
        final_msg = f"【需要调用工具】{answer}"
    else:
        final_msg = f"【直接回答】{answer}"

    # 保存AI回复到当前用户历史
    history.append({"role": "assistant", "content": content})
    history_map[req.userId] = history

    return {
        "userId": req.userId,
        "ai_raw_json": content,
        "answer": answer,
        "need_tool": need_tool,
        "final_msg": final_msg
    }


@app.post("/clear/history")
async def clear_history(userId: str):
    # 严格修正：删除history_map中对应用户的历史
    if userId in history_map:
        del history_map[userId]
    return {"status": "ok", "msg": f"用户 {userId} 历史已清空"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("day06_advanced:app", host="127.0.0.1", port=8000, reload=True)
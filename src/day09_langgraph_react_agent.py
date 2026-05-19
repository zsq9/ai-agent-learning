"""
Day09: 基于 FastAPI + LangGraph 的完整 ReAct Agent
==================================================
功能：计算器、天气、时间，完整 ReAct 闭环，多用户记忆
运行：python day09_agent.py
"""

import operator
from typing import TypedDict, Annotated, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama


# ===================================================================
# 第一步：定义工具（使用 StructuredTool 避免 @tool 装饰器问题）
# ===================================================================

def _calculator(expression: str) -> str:
    """实际计算逻辑"""
    try:
        allowed_names = {
            "__builtins__": {},
            "abs": abs, "round": round, "pow": pow,
            "int": int, "float": float
        }
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

calculator = StructuredTool.from_function(
    func=_calculator,
    name="calculator",
    description="执行数学计算，支持加减乘除和括号。参数 expression: 数学表达式字符串，例如 '2 + 3 * 4'"
)


def _get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

get_current_time = StructuredTool.from_function(
    func=_get_current_time,
    name="get_current_time",
    description="获取当前日期和时间，返回格式：YYYY-MM-DD HH:MM:SS"
)


def _get_weather(city: str) -> str:
    weather_db = {
        "北京": "晴朗，气温 22°C，湿度 45%，微风",
        "上海": "多云转阴，气温 20°C，湿度 70%，可能有小雨",
        "深圳": "晴间多云，气温 26°C，湿度 65%，东南风 3级",
        "广州": "晴朗，气温 28°C，湿度 60%",
        "成都": "阴天，气温 18°C，湿度 80%，可能有雾",
    }
    city_norm = city.strip()
    for key in weather_db:
        if key in city_norm or city_norm in key:
            return f"{city}天气：{weather_db[key]}"
    return f"抱歉，暂时没有 {city} 的天气数据。支持的默认城市：北京、上海、深圳、广州、成都"

get_weather = StructuredTool.from_function(
    func=_get_weather,
    name="get_weather",
    description="模拟获取指定城市的天气信息。参数 city: 城市名称，例如 '北京'"
)

TOOLS = [calculator, get_current_time, get_weather]


# ===================================================================
# 第二步：定义状态（State）
# ===================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# ===================================================================
# 第三步：初始化 LLM 并绑定工具
# ===================================================================
llm = ChatOllama(model="qwen2.5:3b", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


# ===================================================================
# 第四步：节点函数
# ===================================================================
def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


# ===================================================================
# 第五步：构建图
# ===================================================================
def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        }
    )
    workflow.add_edge("tools", "agent")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = build_agent_graph()


# ===================================================================
# 第六步：FastAPI 应用
# ===================================================================
app = FastAPI(title="LangGraph ReAct Agent")

class ChatRequest(BaseModel):
    user_id: str
    query: str

class ChatResponse(BaseModel):
    user_id: str
    query: str
    answer: str
    tool_calls_made: bool = False

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        config = {"configurable": {"thread_id": request.user_id}}
        result = await graph.ainvoke(initial_state, config=config)
        final_message = result["messages"][-1]
        answer = final_message.content
        tool_calls_made = any(
            isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls
            for msg in result["messages"]
        )
        return ChatResponse(
            user_id=request.user_id,
            query=request.query,
            answer=answer,
            tool_calls_made=tool_calls_made
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwen2.5:3b"}

if __name__ == "__main__":
    import uvicorn
    print("启动服务: http://127.0.0.1:8000/docs")
    uvicorn.run(app="day09_langgraph_react_agent:app", host="127.0.0.1", port=8000, reload=True)
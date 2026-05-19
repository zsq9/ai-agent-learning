"""
Day13: Multi-Agent 协作（Supervisor + Workers with ReAct）
带调试打印，方便追踪执行流程
"""

import operator
from typing import TypedDict, Annotated, List, Literal
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ============================================================
# 第一步：定义全局工具
# ============================================================

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    print(f"[工具] calculator 被调用，参数: {expression}")
    try:
        allowed_names = {
            "__builtins__": {},
            "abs": abs, "round": round, "pow": pow,
            "int": int, "float": float
        }
        result = eval(expression, allowed_names, {})
        print(f"[工具] calculator 返回: {result}")
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def get_weather(city: str) -> str:
    """获取天气"""
    print(f"[工具] get_weather 被调用，城市: {city}")
    weather_db = {
        "北京": "晴朗，气温 22°C，湿度 45%",
        "上海": "多云，气温 20°C，湿度 70%",
        "深圳": "晴，气温 26°C，湿度 65%",
    }
    result = weather_db.get(city, f"暂无{city}天气数据")
    print(f"[工具] get_weather 返回: {result}")
    return result

@tool
def retrieve_documents(query: str) -> str:
    """检索文档知识库"""
    print(f"[工具] retrieve_documents 被调用，查询: {query}")
    # 模拟检索结果
    result = "产品保质期为3年，使用时要先打开包装，按照说明书安装成功后插电使用。"
    print(f"[工具] retrieve_documents 返回: {result[:50]}...")
    return result

# ============================================================
# 第二步：创建 Worker 子图（带 ReAct 循环 + 调试打印）
# ============================================================

def create_worker_graph(worker_name: str, tools: list, llm_model: str = "qwen2.5:3b"):
    """为每个 Worker 创建带 ReAct 循环的子图，并添加调试打印"""
    
    class WorkerState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]

    llm = ChatOllama(model=llm_model, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: WorkerState) -> dict:
        print(f"\n[{worker_name}] agent_node 被调用，当前消息数量: {len(state['messages'])}")
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        print(f"[{worker_name}] LLM 响应: content={repr(response.content)[:100]}, tool_calls={response.tool_calls if hasattr(response, 'tool_calls') else None}")
        return {"messages": [response]}

    def should_continue(state: WorkerState) -> Literal["tools", "__end__"]:
        last_msg = state["messages"][-1]
        has_tool_calls = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
        print(f"[{worker_name}] should_continue 判断: last_msg 类型={type(last_msg).__name__}, has_tool_calls={has_tool_calls}")
        if has_tool_calls:
            return "tools"
        return "__end__"

    workflow = StateGraph(WorkerState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()

# 创建三个 Worker 子图
calc_worker_graph = create_worker_graph("Calculator", [calculator])
weather_worker_graph = create_worker_graph("Weather", [get_weather])
rag_worker_graph = create_worker_graph("RAG", [retrieve_documents])

# ============================================================
# 第三步：Supervisor（任务分发）
# ============================================================

supervisor_prompt = SystemMessage(content="""你是一个任务分发监督者。根据用户问题，选择最合适的 Worker 来回答。
可选的 Worker：
- "rag_worker": 用于回答关于文档、产品、手册、保质期、使用说明等内部知识的问题。
- "calc_worker": 用于执行数学计算、数值运算。
- "weather_worker": 用于查询天气。

请只输出 Worker 名称，不要输出其他内容。例如：
用户问"产品保质期" -> 输出 "rag_worker"
用户问"5*8" -> 输出 "calc_worker"
用户问"北京天气" -> 输出 "weather_worker"
""")

def supervisor_node(state: dict) -> dict:
    print(f"\n[Supervisor] 被调用，用户消息: {state['messages'][-1].content}")
    messages = state["messages"]
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)
    response = llm.invoke([supervisor_prompt] + messages)
    worker_name = response.content.strip().lower()
    print(f"[Supervisor] LLM 输出: {repr(response.content)} -> 解析为 worker: {worker_name}")
    if "calc" in worker_name:
        next_worker = "calc_worker"
    elif "weather" in worker_name:
        next_worker = "weather_worker"
    else:
        next_worker = "rag_worker"
    print(f"[Supervisor] 路由到: {next_worker}")
    return {"next_worker": next_worker, "messages": [response]}

def route_after_supervisor(state: dict) -> Literal["calc_worker", "weather_worker", "rag_worker"]:
    return state["next_worker"]

def route_after_worker(state: dict) -> Literal["__end__"]:
    print("[主图] Worker 执行完毕，结束")
    return "__end__"

# ============================================================
# 第四步：构建主图
# ============================================================

class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_worker: str

def build_master_graph():
    workflow = StateGraph(MainState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("calc_worker", calc_worker_graph)
    workflow.add_node("weather_worker", weather_worker_graph)
    workflow.add_node("rag_worker", rag_worker_graph)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "calc_worker": "calc_worker",
            "weather_worker": "weather_worker",
            "rag_worker": "rag_worker",
        }
    )
    workflow.add_conditional_edges("calc_worker", route_after_worker, {"__end__": END})
    workflow.add_conditional_edges("weather_worker", route_after_worker, {"__end__": END})
    workflow.add_conditional_edges("rag_worker", route_after_worker, {"__end__": END})
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = build_master_graph()

# ============================================================
# 第五步：FastAPI 应用
# ============================================================
app = FastAPI(title="Multi-Agent Collaboration (Day13)")

class ChatRequest(BaseModel):
    user_id: str
    query: str

class ChatResponse(BaseModel):
    user_id: str
    query: str
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print("\n" + "="*60)
        print(f"收到请求: user_id={request.user_id}, query={request.query}")
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        config = {"configurable": {"thread_id": request.user_id}}
        result = await graph.ainvoke(initial_state, config=config)
        final_message = result["messages"][-1]
        answer = final_message.content
        print(f"最终答案: {repr(answer)}")
        print("="*60 + "\n")
        return ChatResponse(
            user_id=request.user_id,
            query=request.query,
            answer=answer
        )
    except Exception as e:
        print(f"错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("启动 Multi-Agent 服务，访问 http://127.0.0.1:8000/docs")
    uvicorn.run("day13_multi_agent:app", host="127.0.0.1", port=8000, reload=True)
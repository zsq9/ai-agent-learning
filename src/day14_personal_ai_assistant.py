"""
Day14: 个人 AI 助理（完整项目）
功能：
- 多用户对话记忆（MemorySaver）
- 工具：联网搜索、计算器、天气、RAG 文档检索（FAISS 向量检索）
- 文档上传（支持 .txt, .md, .pdf）
- 流式输出（SSE）+ 工具调用事件
- ReAct 闭环

运行：
    python day14_personal_ai_assistant.py

测试：
    # 上传文档
    curl -X POST http://127.0.0.1:8000/upload -F "file=@说明.txt"
    # 普通对话
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"user_id":"Z17","query":"产品保质期"}'
    # 流式对话
    curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d '{"user_id":"Z17","query":"5*8等于几"}'
"""

import operator
import os
import tempfile
import json
import asyncio
from typing import TypedDict, Annotated, List, AsyncIterator, Literal
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_tavily import TavilySearch  # 联网搜索工具
import azure.cognitiveservices.speech as speechsdk  # TTS语音合成
from langchain_core.tools import StructuredTool

# ============================================================
# 配置
# ============================================================
PERSIST_DIR = "./personal_ai_db"      # 向量数据库存储目录
VECTOR_MODEL = "nomic-embed-text"     # 嵌入模型
LLM_MODEL = "qwen2.5:3b"              # 对话模型（可改为 llama3.2:3b）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# ============================================================
# 工具定义
# ============================================================
@tool
def calculator(expression: str) -> str:
    """执行数学计算，支持加减乘除和括号"""
    try:
        # 安全限制
        allowed_names = {
            "__builtins__": {},
            "abs": abs, "round": round, "pow": pow,
            "int": int, "float": float
        }
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    weather_db = {
        "北京": "晴朗，气温 22°C，湿度 45%",
        "上海": "多云，气温 20°C，湿度 70%",
        "深圳": "晴，气温 26°C，湿度 65%",
        "广州": "多云，气温 25°C，湿度 80%",
        "成都": "阴，气温 18°C，湿度 75%",
    }
    return weather_db.get(city, f"暂无{city}天气数据")

@tool
def retrieve_documents(query: str) -> str:
    """在知识库中检索与问题相关的文档片段"""
    # 全局向量存储对象（在 upload 时初始化）
    global vectorstore
    if vectorstore is None:
        return "知识库为空，请先上传文档。"
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "未找到相关文档。"
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知")
        results.append(f"【来源：{source}】\n{doc.page_content}")
    return "\n\n".join(results)

def web_search(query: str) -> str:
    """
    使用 Tavily 搜索引擎获取最新信息，回答需要实时数据的问题。
    """
    if not TAVILY_API_KEY:
        return "错误：未配置联网搜索API密钥，无法进行搜索。"
    try:
        # 初始化搜索客户端并执行搜索，获取结构化结果
        search = TavilySearch(max_results=3)
        result = search(query)  
        # 格式化返回结果，使其更易于阅读
        formatted_results = []
        for item in result.get("results", []):
            formatted_results.append(f"Title: {item.get('title')}\nLink: {item.get('url')}\nSnippet: {item.get('content')}\n")
        return "\n".join(formatted_results) if formatted_results else "未找到相关信息。"
    except Exception as e:
        return f"搜索失败：{str(e)}"

# 更新全局工具列表，将联网搜索工具加入其中
web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="用于获取实时、最新的网络信息。当用户的问题涉及新闻、时事、股价等需要最新数据时，必须使用这个工具。"
)


def text_to_speech_stream(text: str) -> bytes:
    """
    使用 Azure 语音服务将文本转换为流式音频，并返回完整的音频数据。
    """
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        return b""
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY, 
            region=AZURE_SPEECH_REGION
        )
        # 设置发音人，例如 "zh-CN-XiaoxiaoNeural" 为自然的中文女声
        speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"
        # 将音频输出到内存流
        audio_stream = speechsdk.audio.PullAudioOutputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        # 执行异步的文本转语音任务
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # 返回音频数据
            return audio_stream.read()
        else:
            return b""
    except Exception as e:
        print(f"TTS error: {e}")
        return b""
# 全局变量
vectorstore = None

# ============================================================
# Agent 状态定义
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# ============================================================
# 初始化 LLM 并绑定工具
# ============================================================
tools = [calculator, get_weather, retrieve_documents,web_search_tool]
llm = ChatOllama(model=LLM_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 系统提示（强化指令）
system_prompt = SystemMessage(content="""你是一个智能助手，可以使用以下工具：
- calculator: 执行数学计算，参数 expression 是数学表达式（如 "5*8"）
- get_weather: 查询天气，参数 city 是城市名
- retrieve_documents: 检索文档知识库，参数 query 是问题

当用户提问时，你必须优先判断是否需要调用工具。如果需要，请输出工具调用；如果不需要，直接回答。
对于数学计算，直接调用 calculator，不要自己计算。
对于天气，直接调用 get_weather。
对于文档相关的问题（如产品信息、说明等），调用 retrieve_documents。
不要输出 JSON 或额外解释，直接输出工具调用或答案。""")

# ============================================================
# 节点函数
# ============================================================
def agent_node(state: AgentState) -> dict:
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "__end__"

# ============================================================
# 构建图
# ============================================================
def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = build_agent_graph()

# ============================================================
# 文档上传与向量化
# ============================================================
def load_and_index_documents(file_path: str):
    """加载文档，切分，构建 FAISS 索引（全局）"""
    global vectorstore
    # 加载
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith((".txt", ".md")):
        try:
            loader = TextLoader(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            loader = TextLoader(file_path, encoding="gbk")
    else:
        raise ValueError("仅支持 .txt, .md, .pdf")
    docs = loader.load()
    # 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return
    # 构建或追加 FAISS 索引
    embeddings = OllamaEmbeddings(model=VECTOR_MODEL)
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)
    # 持久化
    os.makedirs(PERSIST_DIR, exist_ok=True)
    vectorstore.save_local(PERSIST_DIR)
    print(f"已索引 {len(chunks)} 个文档块")

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="Personal AI Assistant (Day14)")

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
    """非流式接口"""
    try:
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        config = {"configurable": {"thread_id": request.user_id}}
        result = await graph.ainvoke(initial_state, config=config)
        final_msg = result["messages"][-1]
        answer = final_msg.content
        tool_calls_made = any(
            isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls
            for m in result["messages"]
        )
        return ChatResponse(
            user_id=request.user_id,
            query=request.query,
            answer=answer,
            tool_calls_made=tool_calls_made
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式接口（SSE）"""
    async def event_generator() -> AsyncIterator[str]:
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        config = {"configurable": {"thread_id": request.user_id}}
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            event_type = event["event"]
            if event_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk.content}, ensure_ascii=False)}\n\n"
            elif event_type == "on_chat_model_end":
                output = event["data"]["output"]
                if hasattr(output, "tool_calls") and output.tool_calls:
                    tools_info = [tc["name"] for tc in output.tool_calls]
                    yield f"data: {json.dumps({'type': 'tool_call', 'tools': tools_info}, ensure_ascii=False)}\n\n"
            elif event_type == "on_tool_end":
                tool_output = event["data"]["output"]
                if isinstance(tool_output, ToolMessage):
                    yield f"data: {json.dumps({'type': 'tool_result', 'content': tool_output.content}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        load_and_index_documents(temp_path)
        os.remove(temp_path)
        return {"status": "ok", "message": f"文档 {file.filename} 已索引"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech(request: ChatRequest):
    """
    将文本转化为语音，用于流式播放。
    请求体需包含 { "text": "需要转成语音的文本内容" }
    """
    body = await request.json()
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    audio_data = text_to_speech_stream(text)
    if not audio_data:
        raise HTTPException(status_code=500, detail="Speech synthesis failed")
    return StreamingResponse(
        iter([audio_data]),  # 返回音频数据流
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=speech.wav"
        }
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("启动个人 AI 助理: http://127.0.0.1:8000/docs")
    print("注意：首次使用请先上传文档（如产品说明.txt）")
    uvicorn.run("day14_personal_ai_assistant:app", host="127.0.0.1", port=8000, reload=True)
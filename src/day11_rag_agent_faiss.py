"""
Day11: RAG Agent with FAISS (Windows 友好)
===========================================
无需编译，使用 faiss-cpu。
"""

import operator
from typing import TypedDict, Annotated, List, AsyncIterator
import os
import tempfile
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ============================================================
# 配置
# ============================================================
PERSIST_DIR = "./faiss_db"

# ============================================================
# 文档加载和向量化
# ============================================================

def load_and_index_documents(file_path: str, collection_name: str = "docs"):
    """加载文档，切分，生成 FAISS 索引"""
    print(f"[DEBUG] 开始处理文件: {file_path}")
    # 1. 加载
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith((".txt", ".md")):
        try:
            loader = TextLoader(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            loader = TextLoader(file_path, encoding="gbk")
    else:
        raise ValueError("仅支持 .txt, .md, .pdf")
    
    docs = loader.load()#docs是一个含Document对象的列表
    
    # 2. 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(docs)#切分后的结果类型也是List[Document]
    print(f"3. 切分完成，共 {len(chunks)} 个块")
    if chunks:
        print(f"   第一个块内容预览: {chunks[0].page_content[:100]}")
    
    # 3. 嵌入模型
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 4. 创建或追加 FAISS 索引
    index_path = os.path.join(PERSIST_DIR, collection_name)
    print(f"5. 索引路径: {index_path}")
    if os.path.exists(index_path):
        print("   索引已存在，加载并追加")
        # 加载现有索引，追加新文档
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        print("   索引不存在，创建新索引")
        vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"保存前向量数量: {vectorstore.index.ntotal}")
    print(f"6. 索引已保存到 {index_path}")
    return vectorstore

def get_retriever(collection_name: str = "docs", k: int = 3):
    """获取检索器"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    index_path = os.path.join(PERSIST_DIR, collection_name)
    if not os.path.exists(index_path):
        # 返回一个空检索器（总是返回空列表）
        class EmptyRetriever:
            def invoke(self, query):
                return []
        return EmptyRetriever()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": k})

# ============================================================
# 检索工具
# ============================================================

def _retrieve_documents(query: str) -> str:
    print(f"[DEBUG] 原始查询: {query}")
    
    # 1. 检查索引路径
    index_path = os.path.join(PERSIST_DIR, "user_docs")
    print(f"[DEBUG] 索引路径: {index_path}, 是否存在: {os.path.exists(index_path)}")
    
    # 2. 加载 FAISS 并检查向量数量
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"[DEBUG] 索引中向量数量: {vectorstore.index.ntotal}")
    
    if vectorstore.index.ntotal == 0:
        return "索引中没有向量，请重新上传文档。"
    
    # 3. 使用原始查询进行相似度搜索，打印分数
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)#List[Tuple[Document, float]]
    print(f"[DEBUG] 相似度搜索返回 {len(docs_with_scores)} 个结果")
    for doc, score in docs_with_scores:
        print(f"  分数: {score}, 内容预览: {doc.page_content[:100]}")
    
    if not docs_with_scores:
        return "未找到相关文档。"
    
    # 拼接结果（原逻辑）
    results = []
    for i, (doc, _) in enumerate(docs_with_scores):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "")
        location = f"{source}" + (f" 第{page}页" if page else "")
        results.append(f"【来源：{location}】\n{doc.page_content}")
    return "\n\n".join(results)

retrieve_tool = StructuredTool.from_function(
    func=_retrieve_documents,
    name="retrieve_documents",
    description="""当用户询问关于任何文档、产品、说明、手册、公司内部信息等内容时，**必须**使用此工具检索知识库。
不要用自己的知识回答，因为你的知识库不包含这些文档内容。"""
)

# 可选的计算器工具
def _calculator(expression: str) -> str:
    try:
        allowed_names = {"__builtins__": {}, "abs": abs, "round": round, "pow": pow, "int": int, "float": float}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

calculator = StructuredTool.from_function(
    func=_calculator,
    name="calculator",
    description="执行数学计算"
)

TOOLS = [retrieve_tool, calculator]

# ============================================================
# Agent 定义
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

llm = ChatOllama(model="qwen2.5:3b", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

def agent_node(state: AgentState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"

def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=MemorySaver())

graph = build_agent_graph()

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="RAG Agent with FAISS")

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
        sys_msg = SystemMessage(content="你是一个助手，必须依赖检索工具来回答关于文档内容的问题。如果用户询问文档信息，必须先调用 retrieve_documents 工具。")
        initial_state = {"messages": [sys_msg, HumanMessage(content=request.query)]}
        config = {"configurable": {"thread_id": request.user_id}}
        result = await graph.ainvoke(initial_state, config=config)
        final = result["messages"][-1]
        tool_calls_made = any(
            isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            for m in result["messages"]
        )
        return ChatResponse(
            user_id=request.user_id,
            query=request.query,
            answer=final.content,
            tool_calls_made=tool_calls_made
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        load_and_index_documents(temp_path, collection_name="user_docs")
        os.remove(temp_path)
        return {"status": "ok", "message": f"文档 {file.filename} 已索引"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("启动 FAISS RAG Agent: http://127.0.0.1:8000/docs")
    uvicorn.run("day11_rag_agent_faiss:app", host="127.0.0.1", port=8000, reload=True)
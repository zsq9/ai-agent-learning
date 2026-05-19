"""
Day12: 混合检索 + 重排序（完整可运行版）
========================================
功能：
1. 上传文档，自动切分，建立 FAISS 向量索引和 BM25 关键词索引
2. 检索时同时使用 BM25 和向量检索，通过 RRF 融合结果
3. 可选使用 CrossEncoder 模型对融合结果进行重排序
4. 提供 /upload 和 /chat 两个接口

运行：
    python day12_hybrid_rerank.py

测试：
    # 上传文档
    curl -X POST http://127.0.0.1:8000/upload -F "file=@说明.txt"
    # 普通混合检索
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"user_id":"Z17","query":"保质期"}'
    # 使用重排序
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"user_id":"Z17","query":"保质期","use_reranker":true}'
"""

import os
import tempfile
from typing import List, Tuple
from collections import defaultdict
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba

# 可选：CrossEncoder 重排序（生产级方案）
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    print("警告: sentence-transformers 未安装，无法使用重排序功能。安装命令: pip install sentence-transformers")

# ============================================================
# 配置
# ============================================================
PERSIST_DIR = "./hybrid_db"
VECTOR_MODEL = "nomic-embed-text"
BM25_TOP_K = 20          # BM25 初次召回文档数量
VECTOR_TOP_K = 20        # 向量检索初次召回文档数量
FINAL_TOP_K = 5          # 最终返回数量
RRF_K = 60               # RRF 平滑参数

# 全局 BM25 索引（存储在内存中）
bm25_index = None
bm25_docs = []           # 与 BM25 索引对应的 Document 列表

# 全局 CrossEncoder 模型（懒加载）
reranker_model = None

def get_reranker():
    global reranker_model
    if reranker_model is None and CROSSENCODER_AVAILABLE:
        print("加载 CrossEncoder 模型 BAAI/bge-reranker-v2-m3 ...")
        local_model_path = './models/bge-reranker-v2-m3'      
        reranker_model = CrossEncoder(local_model_path)
    return reranker_model

# ============================================================
# 中文分词（用于 BM25）
# ============================================================
def tokenize_chinese(text: str) -> List[str]:
    return list(jieba.cut(text))

# ============================================================
# RRF 融合算法
# ============================================================
def reciprocal_rank_fusion(rankings: List[List[int]], k: int = RRF_K) -> List[int]:
    """
    将多个排名列表融合成单一排名
    参数:
        rankings: 每个列表元素为文档索引（按相关性降序）
        k: 平滑参数
    返回:
        融合后的文档索引列表（按最终得分降序）
    """
    scores = defaultdict(float)
    for rank_list in rankings:
        for rank, idx in enumerate(rank_list):
            scores[idx] += 1.0 / (k + rank + 1)
    # 按分数降序排序
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_items]

# ============================================================
# 文档加载与索引构建（FAISS + BM25）
# ============================================================
def load_and_index_documents(file_path: str, collection_name: str = "hybrid"):
    global bm25_index, bm25_docs

    # 1. 加载文档
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
    print(f"加载文档: {file_path}, 共 {len(docs)} 个文档片段（按页）")

    # 2. 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        print("警告：文档切分后无内容")
        return
    print(f"切分后得到 {len(chunks)} 个文本块")

    # 3. FAISS 向量索引
    embeddings = OllamaEmbeddings(model=VECTOR_MODEL)
    index_path = os.path.join(PERSIST_DIR, collection_name)
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
        print(f"FAISS 索引已存在，追加 {len(chunks)} 个文档")
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"创建新 FAISS 索引，添加 {len(chunks)} 个文档")
    vectorstore.save_local(index_path)

    # 4. BM25 索引（构建或合并）
    new_texts = [chunk.page_content for chunk in chunks]
    new_tokenized = [tokenize_chinese(text) for text in new_texts]
    if bm25_index is None:
        bm25_index = BM25Okapi(new_tokenized)
        bm25_docs = chunks.copy()
    else:
        # 合并重建（简单方式）
        all_docs = bm25_docs + chunks
        all_tokenized = [tokenize_chinese(doc.page_content) for doc in all_docs]
        bm25_index = BM25Okapi(all_tokenized)
        bm25_docs = all_docs
        print(f"BM25 索引重建，当前文档总数: {len(bm25_docs)}")

    print("索引构建完成")

# ============================================================
# 混合检索（BM25 + FAISS + RRF）
# ============================================================
def hybrid_search(query: str, top_k: int = FINAL_TOP_K) -> List[Document]:
    """
    纯混合检索（不使用重排序）
    返回 RRF 融合后的 Top-K 文档
    """
    global bm25_index, bm25_docs

    # 1. BM25 检索
    bm25_indices = []
    if bm25_index is not None and bm25_docs:
        tokenized_query = tokenize_chinese(query)
        scores = bm25_index.get_scores(tokenized_query)
        # 取分数最高的前 BM25_TOP_K 个索引
        bm25_indices = np.argsort(scores)[-BM25_TOP_K:][::-1].tolist()
        print(f"BM25 召回 {len(bm25_indices)} 个文档")

    # 2. 向量检索
    vector_indices = []
    embeddings = OllamaEmbeddings(model=VECTOR_MODEL)
    index_path = os.path.join(PERSIST_DIR, "hybrid")
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vector_results = vectorstore.similarity_search_with_score(query, k=VECTOR_TOP_K)
        # 将检索到的 Document 映射到 bm25_docs 中的索引（通过内容匹配）
        retrieved_contents = [doc.page_content for doc, _ in vector_results]
        vector_indices = [i for i, doc in enumerate(bm25_docs) if doc.page_content in retrieved_contents]
        print(f"向量召回 {len(vector_indices)} 个文档")
    else:
        print("向量索引不存在，请先上传文档")

    # 3. RRF 融合
    fused_indices = reciprocal_rank_fusion([bm25_indices, vector_indices])
    final_docs = [bm25_docs[idx] for idx in fused_indices[:top_k]]
    print(f"RRF 融合后返回 {len(final_docs)} 个文档")
    return final_docs

# ============================================================
# 混合检索 + CrossEncoder 重排序
# ============================================================
def hybrid_search_with_reranker(query: str, top_k: int = FINAL_TOP_K) -> List[Document]:
    """
    先使用混合检索召回候选文档（数量多一些），再用 CrossEncoder 重排序
    """
    # 召回更多候选（例如 2 倍最终数量）
    candidate_docs = hybrid_search(query, top_k=top_k * 2)
    if not candidate_docs:
        return []

    reranker = get_reranker()
    if reranker is None:
        print("CrossEncoder 不可用，降级为普通混合检索")
        return candidate_docs[:top_k]

    # 构建 (query, document) 对
    pairs = [(query, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)   # 返回分数列表，越高越相关
    # 按分数排序
    scored = list(zip(candidate_docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    final_docs = [doc for doc, _ in scored[:top_k]]
    print(f"CrossEncoder 重排序后返回 {len(final_docs)} 个文档")
    return final_docs

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="Hybrid Search + Rerank (Day12)")

class ChatRequest(BaseModel):
    user_id: str
    query: str
    use_reranker: bool = False   # 是否使用重排序

class ChatResponse(BaseModel):
    user_id: str
    query: str
    answer: str
    retrieved_docs: List[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.use_reranker:
        docs = hybrid_search_with_reranker(request.query, top_k=FINAL_TOP_K)
        method = "混合检索 + CrossEncoder 重排序"
    else:
        docs = hybrid_search(request.query, top_k=FINAL_TOP_K)
        method = "混合检索 (BM25+FAISS+RRF)"
    
    print(f"使用检索方法: {method}")
    
    if not docs:
        answer = "未找到相关文档。"
    else:
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = f"根据检索到的文档：\n{context}"
    
    return ChatResponse(
        user_id=request.user_id,
        query=request.query,
        answer=answer,
        retrieved_docs=[doc.page_content for doc in docs]
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        load_and_index_documents(temp_path, collection_name="hybrid")
        os.remove(temp_path)
        return {"status": "ok", "message": f"文档 {file.filename} 已索引"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Day12 混合检索服务启动: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run(app="day12_hybrid_rerank:app", host="127.0.0.1", port=8000, reload=True)
"""
text.py - 基于最佳配置的 PDF 智能问答交互脚本
配置：chunk_size=300, chunk_overlap=30, k=3
包含自定义 Prompt 以减少幻觉，提升回答质量。
"""

import os
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# ---------- 1. 加载环境变量 ----------
load_dotenv()
API_KEY = os.getenv("MY_KEY")
if not API_KEY:
    raise ValueError("请在 .env 文件中设置 MY_KEY=你的硅基流动API密钥")

BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"   # 嵌入模型（与构建时一致）
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"       # 对话模型（免费）
PERSIST_DIR = "./chroma_db"                  # 向量库存储目录

# ---------- 2. 自定义嵌入类（与构建时保持一致） ----------
class SiliconFlowEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 100
        # 截断过长的文本（安全措施）
        MAX_CHARS = 1500
        truncated_texts = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in resp.data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        MAX_CHARS = 1500
        truncated = text[:MAX_CHARS]
        resp = self.client.embeddings.create(model=self.model, input=[truncated])
        return resp.data[0].embedding

# ---------- 3. 加载已持久化的向量库 ----------
print("⏳ 正在加载向量库...")
embeddings = SiliconFlowEmbeddings(API_KEY, BASE_URL, EMBEDDING_MODEL)
if not os.path.exists(PERSIST_DIR):
    raise FileNotFoundError(f"向量库目录 {PERSIST_DIR} 不存在，请先运行 TXT.py 构建向量库。")

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)
print("✅ 向量库加载成功")

# ---------- 4. 初始化大模型 ----------
llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    model=LLM_MODEL,
    temperature=0.1,
    request_timeout=60
)

# ---------- 5. 定义自定义 Prompt 模板 ----------
# 该模板强制模型基于检索到的上下文回答，减少幻觉
template = """你是一个专业的知识助手。请根据以下检索到的文档片段回答问题。如果文档中没有相关信息，请直接回答“文档中未提及”，不要编造。

检索到的文档片段：
{context}

问题：{question}

回答要求：
如果是对比类问题（如“A和B有什么区别，A和B区别是什么”），请列出至少3个关键区别，并用数字序号（1. 2. 3.）清晰列出。
回答应优先提取文档中最相关、最核心的信息，避免重复或冗余。
保持简洁准确，总字数不超过120字。

回答："""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# ---------- 6. 创建检索问答链 ----------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # k=3 为最佳配置
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True   # 可选，用于调试
)

# ---------- 7. 交互式问答循环 ----------
print("\n💬 进入问答模式（输入 'exit' 或 'quit' 退出）")
print("📌 提示：问题将基于 PDF 文档内容回答。")

while True:
    query = input("\n👤 用户: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 再见！")
        break
    if not query.strip():
        continue
    try:
        result = qa_chain.invoke({"query": query})
        print("🤖 助手:", result["result"])
        # 可选：显示引用的文档片段（方便验证）
        # if "source_documents" in result:
        #     print("\n📄 参考片段：")
        #     for i, doc in enumerate(result["source_documents"][:2]):
        #         print(f"  [{i+1}] {doc.page_content[:100]}...")
    except Exception as e:
        print("❌ 错误:", e)
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 加载环境变量
load_dotenv()
api_key = os.getenv("MY_KEY")
if not api_key:
    raise ValueError(" MY_KEY")

# 1. 加载已持久化的向量库
persist_dir = "./chroma_db"
if not os.path.exists(persist_dir):
    raise FileNotFoundError(f"向量库目录 {persist_dir} 不存在，请先运行构建脚本")

# 因为需要嵌入模型来生成查询向量，但可以复用之前的嵌入类
# 这里为了简便，直接使用自定义嵌入类
from typing import List
from openai import OpenAI
from langchain.embeddings.base import Embeddings

class SiliconFlowEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in resp.data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding

# 嵌入模型（必须与构建时使用相同的模型）
embeddings = SiliconFlowEmbeddings(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-large-zh-v1.5"   # 确保与构建时一致
)

# 2. 加载向量库
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)
print("✅ 向量库加载成功")

# 3. 初始化大模型（建议使用免费模型）
llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen2.5-7B-Instruct",   # 免费模型，如账户余额不足可尝试其他免费模型
    temperature=0.1
)

# 4. 创建检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})   # 检索相关文档数量
)

# 5. 进行提问
print("\n📝 开始问答（输入 'exit' 退出）")
while True:
    query = input("\n用户问题：")
    if query.lower() in ["exit", "quit"]:
        break
    if not query.strip():
        continue
    try:
        result = qa_chain.invoke({"query": query})
        print("💡 答案：", result["result"])
    except Exception as e:
        print("❌ 错误：", e)
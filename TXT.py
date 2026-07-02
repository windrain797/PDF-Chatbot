import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from openai import OpenAI
from typing import List

# 自定义嵌入类
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

# 加载环境变量
load_dotenv()
api_key = os.getenv("MY_KEY")
if not api_key:
    raise ValueError("请在 .env 中设置 MY_KEY")

# 初始化嵌入（使用自定义类）
embeddings = SiliconFlowEmbeddings(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-large-zh-v1.5"
)

# 加载PDF、分割、构建向量库（其余代码不变）
pdf_path = os.path.join(os.path.dirname(__file__), "知识.pdf")
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"成功加载 PDF，共 {len(pages)} 页")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)
print(f"切分完成，共获得 {len(docs)} 个文本块")

try:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ 向量库构建成功")
except Exception as e:
    print("❌ 向量库构建失败:", e)
    raise
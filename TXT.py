import os
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# ---------- 1. 自定义嵌入类（解决 LangChain 与硅基流动的兼容问题） ----------
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


# ---------- 2. 配置 ----------
load_dotenv()
API_KEY = os.getenv("MY_KEY")
if not API_KEY:
    raise ValueError("请在 .env 文件中设置 MY_KEY=你的硅基流动API密钥")

# ===== 请根据实际情况修改以下配置 =====
PDF_NAME = "Python程序设计.pdf"        # 你的 PDF 文件名（确保与脚本同目录）
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"  # 硅基流动支持的嵌入模型
BASE_URL = "https://api.siliconflow.cn/v1"
CHUNK_SIZE = 300                    # 分段大小（字符数）
CHUNK_OVERLAP = 30                   # 分段重叠字符数
K = 3
PERSIST_DIR = "./chroma_db"          # 向量库存储目录
# =====================================


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, PDF_NAME)

    print("📄 开始构建向量库...")
    print(f"   PDF文件: {pdf_path}")

    # 检查 PDF 是否存在
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF文件不存在: {pdf_path}")

    # 初始化嵌入模型
    embeddings = SiliconFlowEmbeddings(API_KEY, BASE_URL, EMBEDDING_MODEL)

    # 1. 加载 PDF
    print("⏳ 加载 PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"✅ 成功加载 PDF，共 {len(pages)} 页")

    # 2. 分割文本
    print("⏳ 分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(pages)
    print(f"✅ 切分完成，共 {len(docs)} 个文本块")

    # 3. 构建向量库
    print("⏳ 生成向量并存储（可能需要几分钟）...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    # Chroma 会自动持久化，无需额外调用 persist()

    print(f"✅ 向量库构建成功，已保存至: {PERSIST_DIR}")
    print("🎉 您现在可以运行问答脚本（如 last.py ）进行提问了。")
if __name__ == "__main__":
    main()
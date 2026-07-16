import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from typing import List
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- 配置 ----------
load_dotenv()
API_KEY = os.getenv("MY_KEY")
if not API_KEY:
    st.error("请在 .env 文件中设置 MY_KEY")
    st.stop()

BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PERSIST_DIR = "./chroma_db"


# ---------- 自定义嵌入类 ----------
class SiliconFlowEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 100
        MAX_CHARS = 1500
        truncated = [t[:MAX_CHARS] for t in texts]
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in resp.data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        MAX_CHARS = 1500
        resp = self.client.embeddings.create(model=self.model, input=[text[:MAX_CHARS]])
        return resp.data[0].embedding


# ---------- 页面标题 ----------
st.set_page_config(page_title="PDF 智能问答系统", layout="wide")
st.title("📚 PDF 智能问答系统")
st.markdown("上传 PDF 文件，提问获取基于文档的答案。")

# ---------- 初始化会话状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------- 侧边栏：上传 PDF 或使用已有向量库 ----------
with st.sidebar:
    st.header("📂 知识库管理")
    option = st.radio("选择数据源", ["使用已有向量库", "上传新 PDF"])

    if option == "上传新 PDF":
        uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")
        if uploaded_file is not None and st.button("构建知识库"):
            with st.spinner("正在处理 PDF..."):
                # 保存临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                # 加载并分割
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=30
                )
                docs = text_splitter.split_documents(pages)
                # 构建向量库
                embeddings = SiliconFlowEmbeddings(API_KEY, BASE_URL, EMBEDDING_MODEL)
                vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIR
                )
                st.session_state.vectorstore = vectorstore
                os.unlink(tmp_path)
                st.success("✅ 知识库构建成功！")

    else:
        if os.path.exists(PERSIST_DIR) and st.button("加载已有向量库"):
            with st.spinner("加载中..."):
                embeddings = SiliconFlowEmbeddings(API_KEY, BASE_URL, EMBEDDING_MODEL)
                vectorstore = Chroma(
                    persist_directory=PERSIST_DIR,
                    embedding_function=embeddings
                )
                st.session_state.vectorstore = vectorstore
                st.success("✅ 向量库加载成功")

    # 显示状态
    if st.session_state.vectorstore:
        st.info("📌 知识库已就绪，可以提问。")
    else:
        st.warning("⚠️ 请先加载或构建知识库。")

# ---------- 主界面：聊天 ----------
if st.session_state.vectorstore:
    # 自定义 Prompt
    template = """你是一个专业的知识助手。请根据以下检索到的文档片段回答问题。如果文档中没有相关信息，请直接回答“文档中未提及”，不要编造。

检索到的文档片段：
{context}

问题：{question}

回答要求：
如果是对比类问题（如“A和B有什么区别，A和B区别是什么”），请列出至少3个关键区别，并用数字序号（1. 2. 3.）清晰列出。
回答应优先提取文档中最相关、最核心的信息，避免重复或冗余。
保持简洁准确，总字数不超过120字。

回答："""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=LLM_MODEL,
        temperature=0.1
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("请输入您的问题"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    result = qa_chain.invoke({"query": prompt})
                    answer = result["result"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"错误: {e}")

else:
    st.info("👈 请先在侧边栏加载或构建知识库。")
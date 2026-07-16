import os
import sys
import shutil
import json
import re
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from typing import List

load_dotenv()
API_KEY = os.getenv("MY_KEY")
if not API_KEY:
    raise ValueError("请在 .env 文件中设置 MY_KEY=你的API密钥")

BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PDF_NAME = "Python程序设计.pdf"
PERSIST_DIR = "./chroma_db"

CHUNK_SIZES = [300, 500, 800]
K_VALUES = [3, 5]
QUICK_MODE = "--quick" in sys.argv
if QUICK_MODE:
    CHUNK_SIZES = [300, 800]
    K_VALUES = [3, 5]
    print("⚡ 快速模式：仅测试2组配置")

class SiliconFlowEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
            embeddings = []
            batch_size = 100
            # 设定最大允许字符数（根据模型限制调整）
            MAX_CHARS = 2000
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # 对每个文本进行截断
                truncated_batch = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in batch]
                try:
                    resp = self.client.embeddings.create(model=self.model, input=truncated_batch)
                    embeddings.extend([item.embedding for item in resp.data])
                except Exception as e:
                    print(f"  嵌入批次失败: {e}")
                    # 可尝试逐个重试或跳过
                    raise
            return embeddings
    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding
#---------------- 读取测试------------------------
def load_test_questions():
    csv_path = "datasets/my_test_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding='utf-8')
        questions = df['question'].tolist()
        print(f"📄 从 {csv_path} 加载了 {len(questions)} 个测试问题")
        return questions
    else:
        default_questions = [
            "在Python中，for循环和while循环的主要区别是什么？",
            "Python中列表和元组的主要区别是什么？",
            "什么是递归函数？编写递归函数时需要满足哪两个关键条件？",
            "在Python中，如何从用户获取数值输入并避免代码注入攻击？",
            "什么是异常处理？请给出一个在Python中使用try-except捕获除零错误的示例。",
            "在面向对象编程中，封装、多态和继承分别是什么意思？",
            "二分查找算法的基本思想是什么？它的时间复杂度是多少？",
            "在Python中，如何打开一个文件并进行读取操作？请写出读取文件每一行的代码示例。"
        ]
        print("⚠️ 未找到 datasets/my_test_data.csv，使用内置的8个教材问题")
        return default_questions

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def evaluate_answer(question, generated_answer, contexts):
    context_text = "\n".join(contexts[:2])[:1500]
    prompt = f"""你是一个严格的评分员。请根据以下信息对回答进行打分（1-5分，5分最高）。

问题：{question}
检索到的上下文：
{context_text}

生成的回答：{generated_answer}

请从以下两个方面打分：
1. 忠实度（Faithfulness）：回答是否完全基于检索到的上下文，没有编造信息？
2. 相关性（Relevance）：回答是否切题、准确？

**输出要求：只输出一个JSON对象，格式如下：**
{{"faithfulness": 分数, "relevance": 分数}}
"""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        # 增强的JSON提取
        try:
            data = json.loads(content)
        except:
            # 尝试用正则匹配
            pattern = r'\{[^{}]*"faithfulness"\s*:\s*(\d+\.?\d*)\s*,\s*"relevance"\s*:\s*(\d+\.?\d*)\s*\}'
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                f = float(match.group(1))
                r = float(match.group(2))
                return max(1, min(5, f)), max(1, min(5, r))
            else:
                # 尝试找1-5的数字
                nums = re.findall(r'\b([1-5])\b', content)
                if len(nums) >= 2:
                    return float(nums[0]), float(nums[1])
                else:
                    return 0, 0
        f = data.get("faithfulness", 0)
        r = data.get("relevance", 0)
        f = max(1, min(5, f))
        r = max(1, min(5, r))
        return f, r
    except Exception as e:
        print(f"  评分出错: {e}")
        return 0, 0

def main():
    test_questions = load_test_questions()
    if not test_questions:
        print("❌ 没有可用的测试问题，退出")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, PDF_NAME)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF文件不存在: {pdf_path}")

    results = []

    for chunk_size in CHUNK_SIZES:
        for k in K_VALUES:
            print(f"\n🔧 测试配置: chunk_size={chunk_size}, k={k}")
            #删除旧向量库
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)
                print("  删除旧向量库")
            #加载并分割
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=int(chunk_size * 0.1)
                )
                docs = text_splitter.split_documents(pages)
                print(f"  分割为 {len(docs)} 个文本块")
                #构建向量数据库
                embeddings = SiliconFlowEmbeddings(API_KEY, BASE_URL, EMBEDDING_MODEL)
                vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIR
                )
                print("  向量库构建完成")
            #创建问答链
                llm = ChatOpenAI(
                    openai_api_key=API_KEY,
                    openai_api_base=BASE_URL,
                    model=LLM_MODEL,
                    temperature=0.1
                )
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": k})
                )
            #对每个查询问题进行评分
                total_f = 0
                total_r = 0
                count = 0
                for question in test_questions:
                    docs_retrieved = vectorstore.similarity_search(question, k=k)
                    contexts = [doc.page_content for doc in docs_retrieved]
                    result = qa_chain.invoke({"query": question})
                    answer = result["result"]
                    f, r = evaluate_answer(question, answer, contexts)
                    total_f += f
                    total_r += r
                    count += 1
                    time.sleep(0.3)#避免请求过频

                avg_f = total_f / count if count else 0
                avg_r = total_r / count if count else 0
                avg_score = (avg_f + avg_r) / 2

                results.append({
                    "chunk_size": chunk_size,
                    "k": k,
                    "avg_faithfulness": avg_f,
                    "avg_relevance": avg_r,
                    "avg_score": avg_score
                })
                print(f"  平均忠实度: {avg_f:.2f}, 平均相关性: {avg_r:.2f}, 综合: {avg_score:.2f}")
            except Exception as e:
                print(f"测试失败：{e}")
                results.append({
                    "chunk_size" : chunk_size,
                    "k" : k,
                    "avg_faithfulness" : 0,
                    "avg_score" : 0,
                    "status" : "FAILED"
                })
      #输出结果
    print("\n" + "="*60)
    print("📊 所有配置测试结果：")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    # 筛选成功的配置
    success_df = df[df['status'] == 'OK']
    if not success_df.empty:
        best = success_df.loc[success_df['avg_score'].idxmax()]
        print("\n🏆 最佳配置推荐：")
        print(f"  chunk_size = {best['chunk_size']}")
        print(f"  k = {best['k']}")
        print(f"  综合得分: {best['avg_score']:.2f}")
    else:
        print("\n❌ 没有任何配置测试成功，请检查环境或API密钥。")

if __name__ == "__main__":
    main()
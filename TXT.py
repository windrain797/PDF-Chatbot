import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载 .env 中的密钥
load_dotenv()
api_key = os.getenv("MY_KEY")

# 2. 创建客户端 
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

# ---------- 测试1：对话模型 (deepseek-ai/DeepSeek-V3) ----------
print("=== 测试对话模型 ===")
try:
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "用一句话证明你已连通"}
        ]
    )
    print("✅ 对话模型响应：", response.choices[0].message.content)
except Exception as e:
    print("❌ 对话模型失败：", e)

# ---------- 测试2：嵌入模型 (BAAI/bge-large-zh-v1.5) ----------
print("\n=== 测试嵌入模型 ===")
try:
    embedding_response = client.embeddings.create(
        model="BAAI/bge-large-zh-v1.5",
        input="今天天气真好"
    )
    # 打印向量前5个维度，确认有结果
    vec = embedding_response.data[0].embedding
    print(f"✅ 嵌入模型成功，向量维度：{len(vec)}，前5个值：{vec[:5]}")
except Exception as e:
    print("❌ 嵌入模型失败：", e)
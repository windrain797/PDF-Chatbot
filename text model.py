import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("MY_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

try:
    response = client.embeddings.create(
        model="BAAI/bge-large-zh-v1.5",
        input="测试文本"
    )
    print("✅ 嵌入成功，向量维度:", len(response.data[0].embedding))
except Exception as e:
    print("❌ 嵌入失败:", e)
    # 打印更详细的错误体
    if hasattr(e, 'response'):
        print("完整错误响应:", e.response.text)
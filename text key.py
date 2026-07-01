import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("MY_KEY")
if key:
    print(f"密钥读取成功，前6位是：{key[:6]}********")
else:
    print("密钥读取失败")
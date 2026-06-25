from openai import OpenAI
import os
from dotenv import load_dotenv # 引入 dotenv 包

# 自动加载根目录下的.env 文件
load_dotenv() 

class BaseLLMClient:
    def __init__(self):
        # 1. 密钥读取：这里会自动从你的.env 文件中获取 API_KEY
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("未找到 API_KEY，请检查根目录下是否已创建.env 文件并填写。")
            
        self.client = OpenAI(
            api_key=api_key,
            # 2. 接口地址填写：如果是阿里云百炼，保持下方地址不变；
            # 如果是 DeepSeek，请改为 "https://api.deepseek.com/v1"
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
        )
        
        # 3. 模型名称填写：如果是阿里云免费模型填 "qwen-plus"；
        # 如果是 DeepSeek 填 "deepseek-chat"
        self.model_name = "qwen3.6-plus" 

    def generate(self, prompt, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    
if __name__ == "__main__":
    print("🚀 正在初始化 LLM 客户端...")
    
    try:
        # 1. 实例化你刚才定义的类
        client = BaseLLMClient()
        
        # 2. 准备一个测试问题
        test_prompt = "你好，请用一句话证明你正常运行中。"
        print(f"👤 我问: {test_prompt}")
        print("⏳ 正在等待阿里云通义千问大模型回复...")
        
        # 3. 调用 generate 方法并接收结果
        answer = client.generate(test_prompt)
        
        # 4. 把结果打印出来！
        print(f"🤖 模型回答: {answer}")
        
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
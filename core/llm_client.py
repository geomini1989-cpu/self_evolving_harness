from openai import OpenAI
import os
from dotenv import load_dotenv
import hashlib # 新增：用于生成唯一的 Prompt 指纹
import httpx
load_dotenv() 

class BaseLLMClient:
    def __init__(self):
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("未找到 API_KEY，请检查根目录下是否已创建.env 文件并填写。")
            
# 🚀 突破底层网络瓶颈：强制开大连接池，允许 100 个并发请求同时起飞
        custom_http_client = httpx.Client(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50)
        )
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=custom_http_client  # ⚠️ 将改好的客户端挂载进来
        )
        
        # 🎯 根据免费额度列表精准配置的黄金铁三角：
        # 1. 经验回放/冒烟测试用：极速轻量的 Flash 模型
        self.cheap_model = "qwen3.6-flash" 
        
        # 2. 主循环日常跑数用：兼顾性能与效率的 Plus 模型
        self.default_model = "qwen3.5-plus-2026-04-20" 
        
        # 3. 归因诊断/规则编写用：高智商的 Max 旗舰模型
        self.smart_model = "qwen3.6-plus-2026-04-02" 
        
        self.cache = {}
    def generate(self, prompt, temperature=0.7, model_type="default", use_cache=True):
        """
        :param use_cache: 默认开启缓存。如果需要强制模型重新思考（如温度>0的创造性任务），可设为 False
        """
        # 1. 缓存拦截机制：将 Prompt 和目标模型打包成唯一的 Hash ID
        if use_cache and temperature == 0.0: # 通常只有确定性输出(temp=0)才缓存最安全
            cache_fingerprint = hashlib.md5((prompt + model_type).encode('utf-8')).hexdigest()
            if cache_fingerprint in self.cache:
                # 🚀 命中缓存！耗时 0 秒，消耗 0 Token
                print("🎯 [Cache Hit] 瞬间返回！(耗时 0 毫秒)")
                return self.cache[cache_fingerprint]

        # 智能路由决策
        if model_type == "cheap": target_model = self.cheap_model
        elif model_type == "smart": target_model = self.smart_model
        else: target_model = self.default_model
            
        try:
            response = self.client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            result = response.choices[0].message.content
            
            # 2. 写入缓存
            if use_cache and temperature == 0.0:
                self.cache[cache_fingerprint] = result
                
            return result
            
        except Exception as e:
            print(f"❌ [LLM 路由异常] 模型 {target_model} 请求失败: {e}")
            if target_model != self.default_model:
                print(f"🔄 [LLM 降级防护] 切换至 {self.default_model} 重试...")
                # 降级重试逻辑（略去重复代码，和之前一致）
                response = self.client.chat.completions.create(
                    model=self.default_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return response.choices[0].message.content
            raise e
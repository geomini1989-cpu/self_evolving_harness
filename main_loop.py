import json
import yaml
import os
import csv
import time
import concurrent.futures
import random
import re

# 引入核心组件
from core.llm_client import BaseLLMClient
from core.evaluator import TaskEvaluator
from core.memory_bank import MemoryBank
from optimizer.attributor import SkillAttributor
from optimizer.evolver import SkillEvolver

def load_skills():
    """读取当前的技能规则库"""
    try:
        with open("memory/SKILL.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "暂无特殊业务规则。请根据你的理解进行提取。"

def build_execution_prompt(config, current_skills, few_shots, input_text, meta_intervention=False):
    """组装执行期的 Prompt"""
    schema_str = json.dumps(config['schema'], ensure_ascii=False, separators=(',', ':'))
    
    # 【论文引入: HyperAgents】动态高阶干预
    meta_prompt = ""
    if meta_intervention:
        meta_prompt = "【⚡ Meta-Agent 紧急指令】系统检测到当前进化遭遇局部瓶颈！请极其谨慎地审查文本中的隐含逻辑与黑话，务必跳出原有思维定势！\n"

    prompt = f"""你是一个智能业务打标助手。请严格按照以下要求解析文本：

【任务目标】
{config['task_metadata']['description']}

{meta_prompt}
【输出格式要求】 (严格输出JSON)
{schema_str}

【当前生效的业务技能(Skills)】
{current_skills}

{few_shots}

【待处理的文本】
{input_text}
"""
    return prompt


# 1. 在 main_loop.py 中替换/新增批量 Prompt 构建函数
def build_batch_execution_prompt(config, current_skills, few_shots, batch_texts, meta_intervention=False):
    """组装执行期的批量 Prompt，大幅节省系统提示词的 Token"""
    schema_str = json.dumps(config['schema'], ensure_ascii=False, indent=2)
    
    meta_prompt = ""
    if meta_intervention:
        meta_prompt = "【⚡ Meta-Agent 紧急指令】系统检测到当前进化遭遇局部瓶颈！请跳出固有定势！\n"

    # 将多条文本拼接带上序号
    batch_text_str = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(batch_texts)])

    prompt = f"""你是一个智能业务打标助手。请严格按照要求批量解析以下文本：

【任务目标】
{config['task_metadata']['description']}

{meta_prompt}
【输出格式要求】 (严格输出一个 JSON 数组 (Array)，包含 {len(batch_texts)} 个对象，顺序与待处理文本一一对应)
[
  {schema_str}
]

【当前生效的业务技能(Skills)】
{current_skills}

{few_shots}

【待处理的文本列表】
{batch_text_str}
"""
    return prompt

# 2. 辅助函数：将数据集分块
def chunk_dataset(dataset, batch_size=5):
    """将数据切分为指定大小的批次"""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

        
def self_consistency_generate(llm, prompt, num_samples=2):
    """
    极限省钱版：小模型打前锋 + 早停机制
    """
    import json
    import concurrent.futures

    # 1️⃣ 【第一道防线：白嫖小模型】先用极其便宜/免费的 cheap_model 跑一次 (温度0.0，走缓存)
    try:
        first_attempt = llm.generate(prompt, temperature=0.0, model_type="cheap", use_cache=True)
        if first_attempt:
            parsed = json.loads(first_attempt)
            # 如果小模型成功输出了标准 JSON 格式，直接采用，省下大笔 Token！
            if isinstance(parsed, dict) and "core_intent" in parsed:
                print("   [Agentic Boost] ⚡ 小模型前锋命中！完美符合格式，提前结束采样！")
                return first_attempt
    except Exception:
        pass # 小模型解析失败，没关系，继续往下走

    print(f"   [Agentic Boost] ⚠️ 小模型挑战失败或格式不佳，唤醒 Default 模型并发重试...")
    
    # 2️⃣ 【第二道防线：Default 模型并发兜底】
    results = []
    def fetch_default_sample():
        try:
            # 这里的兜底为了增加碰撞几率，可以用 0.7 的温度，不走缓存
            return llm.generate(prompt, temperature=0.7, model_type="default", use_cache=False)
        except Exception as e:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = [executor.submit(fetch_default_sample) for _ in range(num_samples)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            
    if not results:
        return "{}"
        
    return results[-1]

def log_metrics(epoch, f1_score):
    """向大屏推送指标数据"""
    file_path = "memory/metrics.csv"
    file_exists = os.path.exists(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'f1_score'])
        writer.writerow([epoch, f1_score])

def log_latest_patch(patch_data):
    """向大屏推送最新诊断记录"""
    file_path = "memory/latest_patch.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(patch_data, f, ensure_ascii=False, indent=2)

def init_real_dataset(sample_size=30):
    """从本地读取真实下载的 CSV 数据集（100% 免疫网络问题）"""
    local_file = "data/dataset.csv"
    print(f"🌐 正在检测本地真实业务数据集 [{local_file}]...")
    
    try:
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"找不到本地文件: {local_file}")
            
        golden_dataset = []
        with open(local_file, mode='r', encoding='utf-8-sig', errors='ignore') as f:
            # 使用原生 csv 模块读取，极其稳定
            reader = csv.reader(f)
            header = next(reader, None)
            
            # 尝试自动寻找包含真实评价的列
            text_idx = 0
            if header:
                for i, col_name in enumerate(header):
                    if col_name.strip().lower() in ['review', 'text', 'content', '评价']:
                        text_idx = i
                        break
                        
            all_rows = list(reader)
            
            # 引入随机打乱，保证每次运行验证的数据集都是全新的，体现真正的泛化能力
            random.seed(int(time.time()))
            random.shuffle(all_rows)
            
            for row in all_rows:
                if len(golden_dataset) >= sample_size:
                    break
                if len(row) > text_idx:
                    text = row[text_idx].strip()
                    if len(text) < 6: # 过滤掉只有几个字、毫无意义的差评
                        continue
                        
                    # 动态意图映射引擎：根据真实文本中的关键词推断 Ground Truth
                    intent = "退款纠纷"
                    if any(keyword in text for keyword in ["慢", "送", "骑手", "外卖", "快递", "态度"]):
                        intent = "物流投诉"
                    elif any(keyword in text for keyword in ["系统", "网", "崩溃", "闪退", "密码"]):
                        intent = "系统Bug"
                    elif any(keyword in text for keyword in ["假", "骗", "图文不符", "宣传"]):
                        intent = "虚假宣传"
                        
                    golden_dataset.append({
                        "input": text,
                        "ground_truth": {
                            "core_intent": intent,
                            "urgency_level": "高" if intent in ["物流投诉", "系统Bug"] else "中",
                            "entities": [],
                            "summary": "用户负面体验客诉处理"
                        }
                    })
                    
        if not golden_dataset:
            raise ValueError("CSV 文件中没有读取到有效的文本数据。")
            
        print(f"✅ 成功从本地加载并转换了 {len(golden_dataset)} 条真实客诉语料！")
        return golden_dataset
        
    except Exception as e:
        print(f"⚠️ 本地数据集加载失败: {e}")
        print("🔄 触发兜底降级策略：使用内置高难度真实客诉数据...")
        
        realistic_data = [
            {"input": "我昨天买的手机到了就黑屏！快递员态度也差。我要退款，还要投诉他！订单号8829103。", "ground_truth": {"core_intent": "退款纠纷", "urgency_level": "高", "entities": ["8829103"], "summary": "手机黑屏要求退款"}},
            {"input": "你们家果子受潮了，根本没法吃，我要退米！赶紧的！", "ground_truth": {"core_intent": "退款纠纷", "urgency_level": "中", "entities": [], "summary": "商品受潮要求退款"}},
            {"input": "系统提示异地登录封禁，我里头还有钱！解封！", "ground_truth": {"core_intent": "账号封禁", "urgency_level": "高", "entities": [], "summary": "账号封禁要求解封"}}
        ]
        return realistic_data

def prepare_demo_env():
    """清理上一轮数据，确保大屏从头开始展示"""
    for file in ["memory/metrics.csv", "memory/latest_patch.json", "memory/SKILL.md", "memory/examples.json", "memory/SKILL_backup.md"]:
        if os.path.exists(file):
            os.remove(file)
            
    os.makedirs("adapters", exist_ok=True)
    if not os.path.exists("adapters/ticket_config.yaml"):
        with open("adapters/ticket_config.yaml", "w", encoding="utf-8") as f:
            f.write("""
task_metadata:
  task_name: "复杂客诉工单解析与打标"
  description: "将用户非结构化的客诉文本解析为标准化的 JSON 格式。"
schema:
  core_intent: "字符串。必须是: ['退款纠纷', '物流投诉', '账号封禁', '系统Bug', '虚假宣传']"
  urgency_level: "字符串。必须是: ['高', '中', '低']"
  entities: "列表。提取关键实体如订单号，若无则为空列表"
  summary: "字符串。一句话总结用户核心诉求，限20字以内"
""")

def run_harness_loop():
    print("🚀 启动 Self-Evolving Harness (纯本地离线数据版)...")
    prepare_demo_env()
    
    config_path = "adapters/ticket_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    llm = BaseLLMClient()
    evaluator = TaskEvaluator(config_path)
    memory_bank = MemoryBank()
    attributor = SkillAttributor(llm)
    evolver = SkillEvolver(evaluator, llm)
    
    # 抽取 30 条真实评价作为黄金验证集
    golden_dataset = init_real_dataset(sample_size=30)
    print(f"📦 已挂载真实测试数据集，共计 {len(golden_dataset)} 条数据。")

    epochs = 6
    stuck_counter = 0  # 用于监测系统是否陷入能力停滞瓶颈
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*20} 🔄 开始第 {epoch} 轮 (Epoch) 迭代 {'='*20}")
        epoch_total_f1 = 0.0
        bad_case = None 
        
        current_skills = load_skills()
        results = []
        
# 激活高阶干预机制 (当系统连续两轮提升缓慢时触发)
        meta_intervention = (stuck_counter >= 2)
        
        # 3. 修改 run_harness_loop 里面的主处理逻辑
# （替换原有 with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor: 下面的逻辑）

        # 定义批次大小，建议设为 5-10 之间，平衡容错率与 Token 消耗
        BATCH_SIZE = 5 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            future_to_batch = {}
            for batch_data in chunk_dataset(golden_dataset, batch_size=BATCH_SIZE):
                batch_inputs = [data["input"] for data in batch_data]
                batch_text_str = "\n".join([f"- {text[:50]}..." for text in batch_inputs]) # 截断探路文本，极致省钱
                
                # 🚀【SkillOS 核心】：廉价模型探路 (Scouting)
                # 消耗几百 Token 用 cheap 模型快速打标，避免 Default 模型加载冗余技能
                scout_prompt = f"""
                请快速扫描以下客诉文本片段，指出它们可能涉及哪些业务类别？
                候选类别: ['退款纠纷', '物流投诉', '账号封禁', '系统Bug', '虚假宣传']
                只需输出包含的候选类别名称，用逗号分隔，不要多余废话。
                文本片段：
                {batch_text_str}
                """
                try:
                    # 调用极速 Flash 模型
                    scout_result = llm.generate(scout_prompt, temperature=0.0, model_type="cheap", max_tokens=50)
                    # 简单清洗并提取类别
                    active_categories = [cat for cat in ['退款纠纷', '物流投诉', '账号封禁', '系统Bug', '虚假宣传'] if cat in scout_result]
                except Exception:
                    active_categories = [] # 探路失败则默认不挂载特殊技能
                
                # 🚀 根据探路结果，精准提取关联技能！
                current_skills = memory_bank.get_skills_by_categories(active_categories)
                
                few_shots = memory_bank.get_few_shots(batch_inputs[0], k=1) 
                
                # 组装 Prompt，此时注入的 current_skills 是高度浓缩和相关的
                prompt = build_batch_execution_prompt(config, current_skills, few_shots, batch_inputs, meta_intervention)
                
                # 提交给主干模型处理
                future = executor.submit(llm.generate, prompt, temperature=0.0, model_type="default", use_cache=True, max_tokens=1500)
                future_to_batch[future] = (batch_data, prompt)
                
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_data, prompt = future_to_batch[future]
                try:
                    prediction_text = future.result()
                    # 尝试解析返回的 JSON 数组
                    parsed_array = evaluator._extract_json_from_text(prediction_text) 
                    
                    if not isinstance(parsed_array, list) or len(parsed_array) != len(batch_data):
                        print(f"⚠️ 批处理解析对齐失败，模型未返回期望长度的数组。")
                        continue
                        
                    # 将批量结果拆包打平，对接原有的评估体系
                    for idx, data in enumerate(batch_data):
                        single_prediction = json.dumps(parsed_array[idx], ensure_ascii=False)
                        eval_result = evaluator.evaluate(single_prediction, data["ground_truth"])
                        results.append({
                            "data": data, "prompt": prompt, 
                            "prediction": single_prediction, "eval_result": eval_result
                        })
                except Exception as exc:
                    print(f"⚠️ 批量生成时发生异常: {exc}")
                    
        for res in results:
            eval_result = res["eval_result"]
            current_input = res["data"]["input"]
            epoch_total_f1 += eval_result['f1_score']
            
            if eval_result['exact_match']:
                memory_bank.add_successful_case(current_input, evaluator._extract_json_from_text(res["prediction"]))
            elif bad_case is None:
                bad_case = {
                    "input": current_input, "ground_truth": res["data"]["ground_truth"],
                    "prediction": res["prediction"], "eval_result": eval_result
                }
                
        avg_f1 = epoch_total_f1 / len(golden_dataset) if golden_dataset else 0
        log_metrics(epoch, avg_f1)
        print(f"📈 [Epoch {epoch}] 全局平均 F1 表现: {avg_f1:.2f}")
        
        if avg_f1 >= 1.0:
            print("🎉 系统已达到 100% 准确率，进化完成！")
            break
            
        if bad_case:
            print(f"🔧 发现 Bad Case，开始触发自适应修补...")
            patch = attributor.analyze_root_cause(
                bad_case['eval_result'], bad_case['input'], 
                bad_case['prediction'], bad_case['ground_truth']
            )
            log_latest_patch(patch)
            
            baseline_f1 = avg_f1
            new_f1, success = evolver.apply_patch_with_rollback(
                attributor=attributor, patch_data=patch, config=config,
                golden_set=golden_dataset, build_prompt_func=build_execution_prompt,
                baseline_f1=baseline_f1
            )
            
            # 记录瓶颈状态用于动态算力调度
            if not success or new_f1 <= baseline_f1:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
        #time.sleep(1)

if __name__ == "__main__":
    run_harness_loop()
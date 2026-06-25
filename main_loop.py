from core.llm_client import BaseLLMClient
from optimizer.attributor import SkillAttributor
from datasets import Dataset
import json
import time

def load_skills():
    with open("memory/SKILL.md", "r", encoding="utf-8") as f:
        return f.read()

def load_ccks_dataset():
    """
    使用 datasets 库加载 CCKS 2021 中文地址解析数据集的样本。
    在真实业务中，这里可替换为 load_dataset("json", data_files="ccks_train.json")
    """
    print("[系统] 正在通过 datasets 库加载 CCKS 地址数据集...")
    mock_data = {
        "id": ["001", "002", "003"],
        "text": [
            "浙江省杭州市余杭区五常街道文一西路969号淘宝城5号楼，放前台",
            "江苏省南京市建邺区江东中路258号新华报业传媒广场1座",
            "北京市海淀区上地十街10号百度大厦"
        ]
    }
    return Dataset.from_dict(mock_data)

def main():
    print("=== 启动 AI 自进化闭环框架 (支持批量数据集驱动) ===")
    llm = BaseLLMClient()
    attributor = SkillAttributor(llm)

    # 1. 挂载数据集
    dataset = load_ccks_dataset()
    print(f"[系统] 数据集加载完成，共计 {len(dataset)} 条验证数据。\n")

    max_retries = 3  # 单条数据最大自进化重试次数
    
    # 2. 遍历数据集中的每一条地址进行测试
    for item in dataset:
        dataset_input = item["text"]
        print(f"==================================================")
        print(f"开始处理任务 ID: {item['id']} | 待解析地址: {dataset_input}")
        
        test_query = f"""
        请提取以下地址文本中的地理要素，并严格以纯 JSON 格式返回。
        需要提取的键名包括：Province(省), city(市), district(区), town(街道), road(路), road_number(路号), poi(兴趣点), house_number(楼栋号), other(补充说明)。
        输入文本：{dataset_input}
        """

        success = False
        
        # 3. 针对当前数据，如果失败则触发自进化循环
        for attempt in range(1, max_retries + 1):
            current_skills = load_skills()
            print(f"\n  [尝试 {attempt}/{max_retries}] Agent 正在根据当前 SKILL.md 执行...")
            
            execution_prompt = f"请严格遵循以下规则：\n{current_skills}\n\n用户问题：{test_query}"
            output = llm.generate(execution_prompt)
            
            # 环境校验
            is_pure_json = True
            parsed_json = None
            try:
                if "```" in output:
                    raise ValueError("包含了 Markdown 代码块包裹")
                parsed_json = json.loads(output.strip())
            except Exception as e:
                is_pure_json = False
                error_msg = str(e)

            if not is_pure_json:
                print(f"  >> 校验失败！环境报错: {error_msg}")
                print("  >> 正在触发底层的步级归因引擎，诊断并重写规则库...")
                
                # 触发诊断与自进化，重写本地规则
                suggestion, new_skills = attributor.analyze_and_update(
                    trajectory=f"输入指令: {test_query}\n模型输出: {output}",
                    current_skills=current_skills
                )
                print("  >> 规则库已完成自适应热更新！准备重试。")
                time.sleep(2) # 稍微停顿，避免触发API限流
            else:
                print("  >> 校验成功！当前技能库完美解析了该地址。")
                print(f"  >> 解析结果: {json.dumps(parsed_json, ensure_ascii=False)}")
                success = True
                break # 当前数据处理成功，跳出重试循环，处理下一条数据
        
        if not success:
            print(f"  [警告] 任务 ID: {item['id']} 在 {max_retries} 次进化后依然失败，记录 Bad Case。")
        time.sleep(1)

if __name__ == "__main__":
    main()
# core/memory_bank.py
import json
import os
import random

class MemoryBank:
    def __init__(self, memory_file="memory/examples.json"):
        self.memory_file = memory_file
        # 确保 memory 目录存在
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.examples = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def _save_memory(self):
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, ensure_ascii=False, indent=2)

    def add_successful_case(self, input_text, output_json):
        """将评估得分为 1.0 的完美预测存入记忆库"""
        # 简单查重
        if not any(ex['input'] == input_text for ex in self.examples):
            self.examples.append({
                "input": input_text,
                "output": output_json
            })
            self._save_memory()
            print(f"✅ [Memory Bank] 新增一条完美经验，当前记忆库规模: {len(self.examples)}")

    def get_few_shots(self, current_input, k=2):
        """
        获取 Few-shot 示例。
        进阶做法是引入向量检索 (ChromaDB) 找最相似的文本。
        这里作为轻量级闭环演示，采用随机抽取即可展现框架能力。
        """
        if not self.examples:
            return ""
        
        # 随机抽取 k 个示例
        sample_size = min(len(self.examples), k)
        chosen_examples = random.sample(self.examples, sample_size)
        
        few_shot_prompt = "【参考历史成功案例】\n"
        for i, ex in enumerate(chosen_examples):
            few_shot_prompt += f"示例 {i+1}:\n输入: {ex['input']}\n输出: {json.dumps(ex['output'], ensure_ascii=False)}\n\n"
        return few_shot_prompt
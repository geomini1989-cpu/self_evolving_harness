# core/memory_bank.py
import json
import os
import random
import re

class MemoryBank:
    def __init__(self):
        self.few_shots = []
        self.skills_db = {} # 🚀 新增：结构化的技能库字典
        self._load_few_shots()
        self._parse_skills_from_md() # 初始化时自动解析技能

    def _parse_skills_from_md(self):
        """
        【SkillOS】将扁平的 SKILL.md 解析为按意图分类的技能字典
        假设后续 Evolver 写入规则时，采用格式：## [退款纠纷] 识别潜在退款威胁
        """
        skill_file = "memory/SKILL.md"
        if not os.path.exists(skill_file):
            return

        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 正则解析：匹配 "## [类别] 规则标题" 及下方的内容区块
        pattern = re.compile(r'##\s*\[(.*?)\]\s*(.*?)\n(.*?)(?=\n##|\Z)', re.DOTALL)
        matches = pattern.findall(content)

        for match in matches:
            category = match[0].strip()
            title = match[1].strip()
            body = match[2].strip()
            
            if category not in self.skills_db:
                self.skills_db[category] = []
            self.skills_db[category].append(f"【{title}】\n{body}")

    def get_skills_by_categories(self, categories, top_k_per_cat=2):
        """
        【SkillOS】按需动态检索技能。
        :param categories: 探路模型识别出的潜在意图类别列表 (如 ['物流投诉', '退款纠纷'])
        """
        if not self.skills_db:
            return "暂无特殊业务规则。请根据你的理解进行提取。"

        retrieved_skills = []
        for cat in categories:
            if cat in self.skills_db:
                # 提取该类别下最新的 top_k 条规则
                skills_for_cat = self.skills_db[cat][-top_k_per_cat:]
                retrieved_skills.extend(skills_for_cat)
                
        if not retrieved_skills:
             return "未检索到针对当前场景的专属技能规则，请依赖基础大模型逻辑判断。"
             
        return "\n\n".join(retrieved_skills)
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
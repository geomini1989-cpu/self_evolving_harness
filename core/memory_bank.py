# core/memory_bank.py
import json
import os
import random
import re

class MemoryBank:
    def __init__(self):
        # 修复：统一变量名为 examples，与下方的 get_few_shots 保持一致
        self.examples = [] 
        self.skills_db = {} 
        self._load_few_shots()       # 核心修复：现在有这个方法了
        self._parse_skills_from_md() 

    def _load_few_shots(self):
        """新增：从本地加载成功的历史案例"""
        self.examples_file = "memory/examples.json"
        if os.path.exists(self.examples_file):
            try:
                with open(self.examples_file, "r", encoding="utf-8") as f:
                    self.examples = json.load(f)
            except Exception:
                self.examples = []
        else:
            self.examples = []

    def add_successful_case(self, input_text, output_json):
        """新增：将成功的案例追加到记忆库中，供主循环调用并作为后续 Few-shot 使用"""
        # 防止重复添加同一个 input
        if any(ex.get('input') == input_text for ex in self.examples):
            return
            
        self.examples.append({
            "input": input_text,
            "output": output_json
        })
        
        # 持久化保存到本地文件
        os.makedirs(os.path.dirname(self.examples_file), exist_ok=True)
        with open(self.examples_file, "w", encoding="utf-8") as f:
            json.dump(self.examples, f, ensure_ascii=False, indent=2)

    def _parse_skills_from_md(self):
        """
        【SkillOS】将扁平的 SKILL.md 解析为按意图分类的技能字典
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
    def refresh_skills(self):
        """🔄 [核心修复] 清除内存中的旧技能，从磁盘重新加载最新进化的规则补丁"""
        self.skills_db = {}
        self._parse_skills_from_md()
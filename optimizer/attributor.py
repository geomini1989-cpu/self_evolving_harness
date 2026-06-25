import re

class SkillAttributor:
    def __init__(self, llm_client):
        self.llm = llm_client

    def analyze_and_update(self, trajectory, current_skills, file_path="memory/SKILL.md"):
        prompt = f"""
        你是一个AI系统诊断专家。以下是一次执行失败的地址解析任务轨迹：
        {trajectory}

        这是智能体当前挂载的外部技能库规则：
        {current_skills}

        【失败原因】：智能体输出了多余的自然语言，或包含了markdown代码块，没有返回纯粹的JSON对象。
        【你的任务】：
        1. 找出并删除导致失败的错误规则（如要求表现礼貌、多做解释等）。
        2. 新增防错规则：强制要求只输出JSON本身，绝对不能有 markdown 代码块（```json）包裹和任何引导语。
        3. 重写完整的技能库规则。
        
        【输出要求】：请务必将最新修改好的完整 SKILL.md 内容，放置在 ```markdown 和 ``` 之间返回。
        """
        
        # 调用大模型生成反思和修改后的规则
        suggestion = self.llm.generate(prompt)
        
        # 正则表达式：自动提取大模型输出的 markdown 代码块内容
        match = re.search(r'```markdown\s*(.*?)\s*```', suggestion, re.DOTALL)
        if not match:
            # 兼容大模型有时只写 ``` 的情况
            match = re.search(r'```\s*(.*?)\s*```', suggestion, re.DOTALL)
        
        if match:
            new_skill_content = match.group(1).strip()
            # 自动将新规则覆盖写入 SKILL.md，完成数字经验的沉淀
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_skill_content)
            return suggestion, new_skill_content
        
        return suggestion, current_skills
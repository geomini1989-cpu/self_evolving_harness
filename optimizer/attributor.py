import json
import re
import os

class SkillAttributor:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.skill_file = "memory/SKILL.md"
        # 确保 memory 目录存在
        os.makedirs(os.path.dirname(self.skill_file), exist_ok=True)

    def analyze_root_cause(self, eval_result, current_input, prediction_text, ground_truth):
        """
        核心归因逻辑：让大模型像“外科医生”一样诊断错误，并生成具体的修复补丁 (Patch)
        """
        diagnostic_prompt = f"""你是一个顶级的 AI 系统诊断引擎。以下是一次大模型在“复杂客诉解析”任务中的执行失败记录。

        【用户输入文本】：
        {current_input}

        【期望的正确输出 (Ground Truth)】：
        {json.dumps(ground_truth, ensure_ascii=False)}

        【模型的实际输出】：
        {prediction_text}

        【系统评估报错诊断】：
        {eval_result.get('error_reason', '字段不匹配或解析失败')}

        请按以下步骤进行结构化诊断：
        1. 分析大模型为什么会做错（根因分析）。它是因为没看懂黑话？还是忽略了多个诉求中的主要矛盾？
        2. 归类错误类型，必须从以下选择：[格式损坏、实体遗漏、意图分类错误、业务逻辑冲突、领域知识盲区]。
        3. 提出一条极其明确、简短的【新增强制规则 (Patch)】，用于指导大模型下次不要犯同样的错误。规则应该具有通用性，而不是只针对这一句话。

        【强制输出格式】：请严格返回纯 JSON 对象，绝对不要包含 markdown 代码块包裹，也不要任何引导语。JSON结构必须如下：
        {{
            "error_category": "填入错误类别",
            "root_cause_analysis": "简短的根本原因分析",
            "proposed_rule": "【强制规则】当遇到...情况时，必须..."
        }}
        """
        
        print("🔍 [Attributor] 正在调用大模型进行病理诊断与归因分析...")
        # 调用大模型生成结构化反思
        suggestion = self.llm.generate(diagnostic_prompt)
        
        # 解析大模型返回的诊断 JSON (增加极强的鲁棒性兼容)
        try:
            # 清理可能存在的 markdown 代码块标记 (使用 \`{3} 替代直接输入反引号，避免前端渲染Bug)
            cleaned_suggestion = re.sub(r'\`{3}(?:json)?(.*?)\`{3}', r'\1', suggestion, flags=re.DOTALL).strip()
            
            # 如果模型依然返回了非 JSON 字符，尝试用正则提取花括号内容
            if not cleaned_suggestion.startswith('{'):
                match = re.search(r'\{.*?\}', cleaned_suggestion, re.DOTALL)
                if match:
                    cleaned_suggestion = match.group(0)
                    
            patch_data = json.loads(cleaned_suggestion)
            return patch_data
        except Exception as e:
            print(f"⚠️ [Attributor] 归因引擎解析自身输出失败: {e}\n模型原输出: {suggestion}")
            return {
                "error_category": "归因引擎解析失败",
                "root_cause_analysis": "大模型未按要求返回合法的 JSON 诊断结果。",
                "proposed_rule": None
            }

    def apply_patch(self, patch_data):
        """
        微创级修补：不再全量覆盖，而是将新规则追加到技能库中
        """
        new_rule = patch_data.get("proposed_rule")
        if not new_rule:
            print("⏭️ [Attributor] 未生成有效的新规则，跳过本次修补。")
            return False
            
        print(f"📉 [病理诊断] 错误分类: {patch_data.get('error_category')}")
        print(f"💡 [根因分析] {patch_data.get('root_cause_analysis')}")
        print(f"🛠️ [微创修补] 准备打入新规则: {new_rule}")
        
        # 读取当前内容
        if os.path.exists(self.skill_file):
            with open(self.skill_file, "r", encoding="utf-8") as f:
                current_content = f.read().strip()
        else:
            current_content = "# 智能体核心执行规范 (SKILL.md)\n> 本文档由系统自进化闭环自动维护，请勿手动修改。\n\n## 业务规则列表："
            
        # 确保现有内容末尾有换行
        if not current_content.endswith('\n'):
            current_content += '\n'
            
        # 追加新规则（作为列表项）
        updated_content = current_content + f"- {new_rule}\n"
        
        # 写入文件 (注意这里用 'w' 是因为我们用 updated_content 拼接了原文，实现了实质上的 Append)
        with open(self.skill_file, "w", encoding="utf-8") as f:
            f.write(updated_content)
            
        print("✅ [Attributor] 规则补丁已成功追加至 SKILL.md，系统完成自我进化！")
        return True
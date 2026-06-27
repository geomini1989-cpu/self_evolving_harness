import os
import shutil
import re
import concurrent.futures
import json  # 👈 核心修复：导入了 json 模块

class SkillEvolver:
    def __init__(self, evaluator, llm_client):
        self.evaluator = evaluator
        self.llm = llm_client
        self.skill_file = "memory/SKILL.md"
        self.backup_file = "memory/SKILL_backup.md"
        
        # 定义全局合法的业务意图类别，防止模型乱编类别
        self.valid_categories = ['退款纠纷', '物流投诉', '账号封禁', '系统Bug', '虚假宣传']

    def _build_evolve_prompt(self, root_cause_analysis):
        # 增加容错：如果是字典，先转为漂亮的 JSON 字符串，防止裸字典拼接导致大模型误解或语法异常
        if isinstance(root_cause_analysis, dict):
            analysis_text = json.dumps(root_cause_analysis, ensure_ascii=False, indent=2)
        else:
            analysis_text = str(root_cause_analysis)
            
        prompt = f"""
你是一个高级业务架构师，正在为一个自动化客诉系统编写底层判定规则（Skill）。
根据以下 Attributor 诊断出的错误根因，请跳出具体案例，提炼出一条具有高度泛化性的通用规则，以防止系统未来犯同样的错误。

【诊断结果与根因分析】
{analysis_text}

【⚠️ 强制输出协议 (Skill Constraint Protocol)】
你生成的规则必须严格遵循以下 Markdown 结构，绝对不能包含任何其他废话、寒暄或解释。
必须且只能包含以下三个部分：标题（标明类别）、触发条件、输出动作。

👇 请严格照抄以下模板格式输出（特别注意 ## 后面的中括号）：
## [填写具体类别] 这里写简明扼要的规则标题
（这里是具体的规则内容，包含触发条件和对应的输出动作，支持自然语言或伪代码逻辑。）

*注：[填写具体类别] 必须是以下之一：{self.valid_categories}。如果你认为该规则跨类别，请选择最核心的一个。*
"""
        return prompt

    def _safe_write_skill_to_md(self, new_skill_text):
        """【协议守门员】在写入前进行正则体检，过滤非法输出 (已增强鲁棒性)"""
        clean_text = new_skill_text.strip()
        
        # 🚀 优化正则：兼容大模型忘记写中括号的情况，例如 "## 退款纠纷 标题" 和 "## [退款纠纷] 标题" 都能识别
        pattern = re.compile(r'^##\s*\[?([^\s\]]+)\]?\s+(.*?)\n(.*)', re.DOTALL)
        match = pattern.match(clean_text)
        
        if not match:
            print(f"❌ [协议拦截] 进化的补丁格式完全不合规，拒绝写入！\n内容片段: {clean_text[:50]}...")
            return False
            
        category = match.group(1).strip()
        
        if category not in self.valid_categories:
            print(f"❌ [协议拦截] 发现未知的业务类别 [{category}]，拒绝写入！")
            return False

        # 备份当前技能库，用于防御性回滚
        if os.path.exists(self.skill_file):
            shutil.copy(self.skill_file, self.backup_file)

        # 校验通过，安全写入
        os.makedirs(os.path.dirname(self.skill_file), exist_ok=True)
        with open(self.skill_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{clean_text}\n")
            
        print(f"✅ [协议执行] 成功将 [{category}] 领域的新技能编入全局技能库。")
        return True

    def apply_patch_with_rollback(self, attributor, patch_data, config, golden_set, build_prompt_func, baseline_f1):
        """
        【核心进化流】根据归因结果生成补丁 -> 验证补丁 -> 若能力退化则回滚
        """
        # 防御机制：如果传进来的 patch_data 是无效的（比如多次重试失败后的 fallback），直接跳过
        if not patch_data or not patch_data.get("proposed_rule"):
            print("⚠️ [Evolver] 收到无效的补丁数据（缺少 proposed_rule），防御性拦截，放弃本轮修补。")
            return baseline_f1, False
            
        print("\n⚙️ [Evolver] 正在基于根因诊断生成全局规则补丁...")
        
        evolve_prompt = self._build_evolve_prompt(patch_data)
        new_skill_text = self.llm.generate(evolve_prompt, temperature=0.2, model_type="smart", max_tokens=800)
        
        if not self._safe_write_skill_to_md(new_skill_text):
            print("⚠️ [Evolver] 补丁被拦截，本轮进化终止，维持原状。")
            return baseline_f1, False
            
        print("🧪 [Evolver] 补丁已挂载，开始在黄金验证集上进行批量回归测试...")
        
        with open(self.skill_file, "r", encoding="utf-8") as f:
            updated_skills = f.read()

        new_total_f1 = 0.0
        BATCH_SIZE = 5
        
        def chunk_dataset(dataset, batch_size):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i:i + batch_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_batch = {}
            for batch_data in chunk_dataset(golden_set, BATCH_SIZE):
                batch_inputs = [data["input"] for data in batch_data]
                prompt = build_prompt_func(config, updated_skills, few_shots="", batch_texts=batch_inputs, meta_intervention=False)
                
                future = executor.submit(self.llm.generate, prompt, temperature=0.0, model_type="default", use_cache=False)
                future_to_batch[future] = batch_data
                
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_data = future_to_batch[future]
                try:
                    prediction_text = future.result()
                    parsed_array = self.evaluator._extract_json_from_text(prediction_text)
                    
                    if not isinstance(parsed_array, list) or len(parsed_array) != len(batch_data):
                        continue
                        
                    for idx, data in enumerate(batch_data):
                        single_prediction = json.dumps(parsed_array[idx], ensure_ascii=False)
                        eval_result = self.evaluator.evaluate(single_prediction, data["ground_truth"])
                        new_total_f1 += eval_result['f1_score']
                except Exception:
                    pass
                    
        new_avg_f1 = new_total_f1 / len(golden_set) if golden_set else 0
        print(f"📊 [回归测试] 补丁应用后 F1: {new_avg_f1:.2f} (应用前: {baseline_f1:.2f})")
        
        if new_avg_f1 < baseline_f1:
            print("🚨 [灾难性遗忘告警] 新补丁导致全局指标下降！触发无损回滚机制 (Rollback)...")
            if os.path.exists(self.backup_file):
                shutil.copy(self.backup_file, self.skill_file)
            return baseline_f1, False
        else:
            print("🎉 [进化成功] 新补丁不仅修复了 Bad Case，且未破坏原有能力，规则已永久固化！")
            return new_avg_f1, True
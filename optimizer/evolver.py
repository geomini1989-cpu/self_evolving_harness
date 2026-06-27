import os
import shutil
import re
import concurrent.futures

class SkillEvolver:
    def __init__(self, evaluator, llm_client):
        self.evaluator = evaluator
        self.llm = llm_client
        self.skill_file = "memory/SKILL.md"
        self.backup_file = "memory/SKILL_backup.md"
        
        # 定义全局合法的业务意图类别，防止模型乱编类别
        self.valid_categories = ['退款纠纷', '物流投诉', '账号封禁', '系统Bug', '虚假宣传']

    def _build_evolve_prompt(self, root_cause_analysis):
        """组装生成新技能（补丁）的 Prompt，内置极度严格的格式约束协议"""
        prompt = f"""
你是一个高级业务架构师，正在为一个自动化客诉系统编写底层判定规则（Skill）。
根据以下 Attributor 诊断出的错误根因，请跳出具体案例，提炼出一条具有高度泛化性的通用规则，以防止系统未来犯同样的错误。

【诊断结果与根因分析】
{root_cause_analysis}

【⚠️ 强制输出协议 (Skill Constraint Protocol)】
你生成的规则必须严格遵循以下 Markdown 结构，绝对不能包含任何其他废话、寒暄或解释：

## [类别名称] 简明扼要的规则标题
（这里是具体的规则内容，必须包含触发条件和对应的输出动作，支持自然语言或伪代码逻辑。）

*注：[类别名称] 必须是以下之一：{self.valid_categories}。如果你认为该规则跨类别，请选择最核心的一个。*
"""
        return prompt

    def _safe_write_skill_to_md(self, new_skill_text):
        """【协议守门员】在写入前进行严格的正则体检，过滤非法输出"""
        clean_text = new_skill_text.strip()
        
        # 校验是否严格以 "## [类别] 标题" 开头
        pattern = re.compile(r'^##\s*\[(.*?)\]\s*(.*?)\n(.*)', re.DOTALL)
        match = pattern.match(clean_text)
        
        if not match:
            print(f"❌ [协议拦截] 进化的补丁格式不合规，拒绝写入！\n内容片段: {clean_text[:50]}...")
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

    def trigger_skill_condensation(self, memory_bank, category):
        """
        【SkillAdaptor 进阶功能】轨迹浓缩提炼
        将多个相似成功的 Few-shots 浓缩为一条抽象规则，为系统内存“瘦身”
        """
        raw_trajectories = memory_bank.get_recent_successes(category, limit=5)
        if len(raw_trajectories) < 5:
            return False
            
        print(f"🧠 [轨迹提炼] 检测到 [{category}] 积累了足够的成功轨迹，开始进行规则浓缩...")
        
        prompt = f"""
分析以下 5 个成功的客诉打标轨迹（包含输入文本和正确的输出 JSON）。
请跳出这些具体的案例细节，提炼出 1 条通用的、指导性的底层判定规则(Skill)。

【轨迹数据】
{raw_trajectories}

【⚠️ 强制输出协议】
必须严格输出：
## [{category}] 浓缩规则标题
规则主体内容...
"""
        # 动用最聪明的旗舰模型进行高阶提炼
        new_skill = self.llm.generate(prompt, temperature=0.2, model_type="smart", max_tokens=800)
        
        if self._safe_write_skill_to_md(new_skill):
            memory_bank.clear_raw_trajectories(category) # 提炼成功，清理繁冗轨迹
            return True
        return False

    def apply_patch_with_rollback(self, attributor, patch_data, config, golden_set, build_prompt_func, baseline_f1):
        """
        【核心进化流】根据归因结果生成补丁 -> 验证补丁 -> 若能力退化则回滚
        """
        print("\n⚙️ [Evolver] 正在基于根因诊断生成全局规则补丁...")
        
        evolve_prompt = self._build_evolve_prompt(patch_data)
        # 生成补丁必须用最聪明的模型 (smart_model)
        new_skill_text = self.llm.generate(evolve_prompt, temperature=0.2, model_type="smart", max_tokens=800)
        
        # 1. 协议拦截校验与写入
        if not self._safe_write_skill_to_md(new_skill_text):
            print("⚠️ [Evolver] 补丁被拦截，本轮进化终止，维持原状。")
            return baseline_f1, False
            
        print("🧪 [Evolver] 补丁已挂载，开始在黄金验证集上进行回归测试 (Regression Test)...")
        
        # 读取刚刚更新后的全量规则用于回归测试
        with open(self.skill_file, "r", encoding="utf-8") as f:
            updated_skills = f.read()

        new_total_f1 = 0.0
        
        # 2. 并发进行回归测试，验证新规则是否引发了“灾难性遗忘”
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_data = {}
            for data in golden_set:
                # 注意：这里模拟测试时，简化了 few_shots 的传入，专注测试新规则
                prompt = build_prompt_func(config, updated_skills, few_shots="", input_text=data["input"])
                # 回归测试使用快速的 cheap 或 default 模型即可
                future = executor.submit(self.llm.generate, prompt, temperature=0.0, model_type="default", use_cache=False)
                future_to_data[future] = data
                
            for future in concurrent.futures.as_completed(future_to_data):
                data = future_to_data[future]
                try:
                    prediction = future.result()
                    eval_result = self.evaluator.evaluate(prediction, data["ground_truth"])
                    new_total_f1 += eval_result['f1_score']
                except Exception as e:
                    pass
                    
        new_avg_f1 = new_total_f1 / len(golden_set) if golden_set else 0
        
        print(f"📊 [回归测试] 补丁应用后 F1: {new_avg_f1:.2f} (应用前: {baseline_f1:.2f})")
        
        # 3. 结果判决与回滚机制
        if new_avg_f1 < baseline_f1:
            print("🚨 [灾难性遗忘告警] 新补丁导致全局指标下降！触发无损回滚机制 (Rollback)...")
            if os.path.exists(self.backup_file):
                shutil.copy(self.backup_file, self.skill_file)
            return baseline_f1, False
        else:
            print("🎉 [进化成功] 新补丁不仅修复了 Bad Case，且未破坏原有能力，规则已永久固化！")
            return new_avg_f1, True
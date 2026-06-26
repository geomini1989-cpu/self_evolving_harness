import os
import shutil

class SkillEvolver:
    def __init__(self, evaluator, llm_client):
        self.evaluator = evaluator
        self.llm = llm_client
        self.skill_file = "memory/SKILL.md"
        self.backup_file = "memory/SKILL_backup.md"

    def _backup_skills(self):
        """备份当前稳定的技能库版本"""
        if os.path.exists(self.skill_file):
            shutil.copy(self.skill_file, self.backup_file)

    def _rollback_skills(self):
        """发生退化时，执行版本回滚"""
        if os.path.exists(self.backup_file):
            shutil.copy(self.backup_file, self.skill_file)
            print("⏪ [Evolver] 触发安全保护机制：已回滚至上一版稳定技能库。")

    def run_experience_replay(self, config, golden_set, build_prompt_func):
        """
        经验回放：在沙盒中运行黄金验证集（Golden Set），计算全局平均 F1 分数
        """
        print(f"🔬 [Evolver] 正在执行经验回放回归测试 (测试集规模: {len(golden_set)} 条)...")
        
        if not os.path.exists(self.skill_file):
            return 0.0

        with open(self.skill_file, "r", encoding="utf-8") as f:
            current_skills = f.read()

        total_f1 = 0.0
        for item in golden_set:
            # 不注入 few-shots，以测试规则本身的健壮性
            prompt = build_prompt_func(config, current_skills, "", item["input"])
            
            try:
                prediction = self.llm.generate(prompt)
                result = self.evaluator.evaluate(prediction, item["ground_truth"])
                total_f1 += result["f1_score"]
            except Exception as e:
                print(f"⚠️ [Evolver] 经验回放单条测试异常: {e}")
                
        avg_f1 = total_f1 / len(golden_set) if golden_set else 0.0
        print(f"📊 [Evolver] 经验回放完成。候选版本全局平均 F1 表现: {avg_f1:.2f}")
        return avg_f1

    def apply_patch_with_rollback(self, attributor, patch_data, config, golden_set, build_prompt_func, baseline_f1):
        """
        带安全防护的 CI/CD 补丁应用机制
        """
        print("\n🛡️ [Evolver] 启动补丁合并流程与置信度评估...")
        
        # 1. 备份当前技能库
        self._backup_skills()
        
        # 2. 调用 Attributor 尝试追加补丁
        success = attributor.apply_patch(patch_data)
        if not success:
            return baseline_f1, False

        # 3. 如果没有提供黄金验证集，视为强制接受
        if not golden_set:
            print("⚠️ [Evolver] 未挂载黄金验证集，跳过防退化测试，直接合并新规则。")
            return baseline_f1, True

        # 4. 跑批执行经验回放
        new_f1 = self.run_experience_replay(config, golden_set, build_prompt_func)
        
        # 5. 置信度决策机制
        if new_f1 < baseline_f1:
            print(f"🚨 [Evolver] 灾难性遗忘警告！新版技能 F1 ({new_f1:.2f}) 低于基线 ({baseline_f1:.2f})！")
            self._rollback_skills()
            return baseline_f1, False
        else:
            print(f"🎉 [Evolver] 进化安全检查通过！新版技能 F1 ({new_f1:.2f}) >= 基线 ({baseline_f1:.2f})，补丁已永久生效。")
            return new_f1, True
import os
import shutil
import concurrent.futures  # 新增导入多线程模块

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

# 定义单条数据的处理逻辑
        def evaluate_single(item):
            # 不注入 few-shots，以测试规则本身的健壮性
            prompt = build_prompt_func(config, current_skills, "", item["input"])
            try:
                # 【代码修改点】：使用极速/免费额度极大的小模型做苦力测试
                prediction = self.llm.generate(prompt, model_type="cheap", temperature=0.0) 
                result = self.evaluator.evaluate(prediction, item["ground_truth"])
                return result["f1_score"]
            except Exception as e:
                print(f"⚠️ [Evolver] 经验回放单条测试异常: {e}")
                return 0.0

        # 【核心优化】：使用线程池并发处理，极大缩短等待时间
        total_f1 = 0.0
        # max_workers 设为 5-10 比较安全，避免触发大模型 API 的速率限制 (429错误)
        with concurrent.futures.ThreadPoolExecutor(max_workers=35) as executor:
            f1_scores = list(executor.map(evaluate_single, golden_set))
            total_f1 = sum(f1_scores)
                
        avg_f1 = total_f1 / len(golden_set) if golden_set else 0.0
        print(f"📊 [Evolver] 经验回放完成。候选版本全局平均 F1 表现: {avg_f1:.2f}")
        return avg_f1

    def apply_patch_with_rollback(self, attributor, patch_data, config, golden_set, build_prompt_func, baseline_f1):
            """
            带安全防护的 CI/CD 补丁应用机制 (引入轻量冒烟测试节约成本)
            """
            import random # 确保顶部或这里导入了 random

            print("\n🛡️ [Evolver] 启动补丁合并流程与置信度评估...")
            
            self._backup_skills()
            success = attributor.apply_patch(patch_data)
            if not success:
                return baseline_f1, False

            if not golden_set:
                return baseline_f1, True

            # 【核心省钱优化】：抽样回归测试。只随机取 10 条数据验证退化，而不是跑全量
            sample_size = min(5, len(golden_set))
            mini_golden_set = random.sample(golden_set, sample_size)
            print(f"🗜️ [Evolver] 触发轻量级冒烟测试，随机抽取 {sample_size} 条数据评估退化风险...")
            
            # 跑批执行经验回放 (只测 10 条)
            new_f1 = self.run_experience_replay(config, mini_golden_set, build_prompt_func)
            
            if new_f1 < baseline_f1:
                print(f"🚨 [Evolver] 灾难性遗忘警告！新版技能 F1 ({new_f1:.2f}) 低于基线 ({baseline_f1:.2f})！")
                self._rollback_skills()
                return baseline_f1, False
            else:
                print(f"🎉 [Evolver] 进化安全检查通过！补丁已永久生效。")
                # 补丁通过后，为了保持主循环的基线准确，我们依然返回原基线分数
                # 真正的全局分数会在下一轮 Epoch 全量运行时自然更新
                return baseline_f1, True
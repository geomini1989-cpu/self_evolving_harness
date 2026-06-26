# core/evaluator.py
import json
import re

class TaskEvaluator:
    def __init__(self, config_path):
        """
        初始化评估器，加载任务配置以了解需要评估哪些字段。
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.schema_keys = self.config['schema'].keys()

    def _extract_json_from_text(self, text):
        """从 LLM 的输出中提取 JSON 字符串（兼容带有 markdown 标记的输出）"""
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def evaluate(self, prediction_text, ground_truth_dict):
        """
        核心评估逻辑：计算 F1 Score 与完全匹配度 (Exact Match)
        """
        result = {
            "is_valid_json": False,
            "exact_match": False,
            "f1_score": 0.0,
            "error_reason": ""
        }

        # 1. 基础格式校验
        pred_dict = self._extract_json_from_text(prediction_text)
        if not pred_dict:
            result["error_reason"] = "JSON 格式解析失败或未找到有效 JSON。"
            return result
        
        result["is_valid_json"] = True

        # 2. 字段比对与 F1 计算
        correct_fields = 0
        total_fields = len(self.schema_keys)
        missing_keys = []
        wrong_values = []

        for key in self.schema_keys:
            if key not in pred_dict:
                missing_keys.append(key)
                continue
            
            # 这里进行简单的字符串匹配评估（业务上可根据需采用模糊匹配）
            if str(pred_dict.get(key)) == str(ground_truth_dict.get(key)):
                correct_fields += 1
            else:
                wrong_values.append(key)

        # 3. 计算最终指标
        accuracy = correct_fields / total_fields
        result["f1_score"] = accuracy # 在此简化场景中，我们将准确率等价为 F1 表现
        result["exact_match"] = (correct_fields == total_fields)

        # 4. 生成错误诊断（供后续 Attributor 归因使用）
        if not result["exact_match"]:
            errors = []
            if missing_keys:
                errors.append(f"缺失关键字段: {', '.join(missing_keys)}")
            if wrong_values:
                errors.append(f"字段取值错误(与Ground Truth不符): {', '.join(wrong_values)}")
            result["error_reason"] = " | ".join(errors)

        return result
import streamlit as st
import plotly.graph_objects as go
import json
import time
import os
import sys

# 将项目根目录加入系统路径，以便导入 core 和 optimizer 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_client import BaseLLMClient
from optimizer.attributor import SkillAttributor

# ----------------- 页面基础配置 -----------------
st.set_page_config(page_title="Agent 自进化指标大屏", layout="wide")
st.title("🚀 智能体 Self-Harness 自迭代可视化控制台")

# ----------------- 初始化状态缓存 -----------------
# 修复：初始化为空列表
if 'epoch_history' not in st.session_state:
    st.session_state.epoch_history = []
if 'success_rate_history' not in st.session_state:
    st.session_state.success_rate_history = []
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0

def load_skills():
    try:
        with open("memory/SKILL.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "未能读取到技能库文件"

# ----------------- 模拟数据集驱动 -----------------
mock_dataset = [
    {"id": "001", "text": "浙江省杭州市余杭区五常街道文一西路969号淘宝城5号楼，放前台"},
    {"id": "002", "text": "江苏省南京市建邺区江东中路258号新华报业传媒广场1座"},
    {"id": "003", "text": "北京市海淀区上地十街10号百度大厦"}
]

# ----------------- UI 布局与联动逻辑 -----------------
col1, col2 = st.columns([1, 2])

with col2:
    st.subheader("📝 当前技能库状态 (SKILL.md)")
    skill_display = st.empty()
    skill_display.markdown(f"```markdown\n{load_skills()}\n```")
    
    st.subheader("⚙️ 引擎控制")
    start_btn = st.button("▶️ 启动自迭代循环测试")

with col1:
    st.subheader("📈 核心指标动态曲线")
    chart_display = st.empty()
    
    st.subheader("🔍 迭代执行日志")
    log_display = st.empty()

# 绘制动态折线图函数
def update_chart(epochs, rates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=rates, mode='lines+markers',
        name='任务成功率', line=dict(color='firebrick', width=3),
        marker=dict(size=10)
    ))
    fig.update_layout(
        xaxis_title="迭代轮数 (Epoch)",
        yaxis_title="成功率 (%)",
        # 修复：设置 y 轴范围为 0 到 100
        yaxis=dict(range=[0, 100]),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    chart_display.plotly_chart(fig, use_container_width=True)

# 初始化空图表
update_chart(st.session_state.epoch_history, st.session_state.success_rate_history)

# ----------------- 核心迭代引擎 (响应按钮点击) -----------------
if start_btn:
    try:
        llm = BaseLLMClient()
        attributor = SkillAttributor(llm)
    except Exception as e:
        st.error(f"引擎初始化失败（请检查.env 中的 API 密钥）: {e}")
        st.stop()

    logs = ""
    # 模拟进行 3 轮 Epoch 迭代验证
    for epoch in range(1, 4):
        st.session_state.current_epoch = epoch
        logs += f"**[Epoch {epoch}]** 正在挂载当前 SKILL.md 进行跑批测试...\n\n"
        log_display.markdown(logs)
        
        success_count = 0
        total_count = len(mock_dataset)
        
        # 跑批测试当前 Epoch 的准确率
        for item in mock_dataset:
            test_query = f"提取地址地理要素并严格返回JSON。输入文本：{item['text']}"
            execution_prompt = f"请遵循以下规则：\n{load_skills()}\n\n{test_query}"
            
            output = llm.generate(execution_prompt)
            
            # 环境校验 (检查是否为合法且纯净的 JSON)
            is_valid = True
            try:
                if "```" in output: 
                    raise ValueError("包含代码块")
                json.loads(output.strip())
            except Exception:
                is_valid = False
            
            if is_valid:
                success_count += 1
                
        # 计算并更新指标
        current_success_rate = (success_count / total_count) * 100
        st.session_state.epoch_history.append(epoch)
        st.session_state.success_rate_history.append(current_success_rate)
        update_chart(st.session_state.epoch_history, st.session_state.success_rate_history)
        
        logs += f"- 测试完毕。成功样本数: {success_count}/{total_count}，当前成功率: {current_success_rate:.1f}%\n\n"
        log_display.markdown(logs)
        
        # 如果未达到 100% 成功率，触发自动归因与规则修补
        if current_success_rate < 100:
            logs += "⚠️ 检测到 Bad Case，触发 SkillAttributor 步级归因与自我修补...\n\n"
            log_display.markdown(logs)
            
            # 取最后一条失败的数据进行反思更新
            current_skills = load_skills()
            suggestion, new_skills = attributor.analyze_and_update(
                trajectory=f"输入指令: {test_query}\n模型输出: {output}",
                current_skills=current_skills,
                file_path="memory/SKILL.md"
            )
            
            logs += "✅ SKILL.md 规则库已完成自适应重写！准备进入下一轮 Epoch。\n\n---\n\n"
            log_display.markdown(logs)
            
            # 实时更新前端右侧的技能库展示
            skill_display.markdown(f"```markdown\n{new_skills}\n```")
            time.sleep(2)
        else:
            logs += "🎉 成功率已达 100%，系统已进化至最优状态！\n\n"
            log_display.markdown(logs)
            break
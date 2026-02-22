"""
Page 7 – RL Agent
RL optimization panel: agent status, environment visualisation, training controls.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from streamlit_app.simulation.state_manager import init_session, get_current
from streamlit_app.components.metrics_cards import metric_card

init_session()

st.markdown("## 🤖 RL Agent")
st.markdown("*Reinforcement Learning-based grid optimization (PPO / SAC)*")

# ── Lazy import agent ────────────────────────────────────────────────────
@st.cache_resource
def _init_agent():
    try:
        from src.agent.environment import DEMSEnvironment
        from src.agent.rl_agent import RLAgent
        env = DEMSEnvironment()
        agent = RLAgent(env, algorithm="PPO")
        return env, agent, None
    except Exception as exc:
        return None, None, str(exc)

env, agent, err = _init_agent()

if err:
    st.warning(f"RL Agent could not be initialised: {err}")
    st.info("This page requires `stable-baselines3` and `gymnasium`. Install them to enable RL features.")
    st.stop()

# ── Agent info ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    algo = getattr(agent, "algorithm", "PPO")
    metric_card("Algorithm", str(algo), color="#AA00FF", icon="🧠")
with c2:
    stats = agent.get_training_stats() if hasattr(agent, "get_training_stats") else {}
    episodes = stats.get("total_timesteps", 0)
    metric_card("Timesteps", str(episodes), color="#4ECDC4", icon="📈")
with c3:
    metric_card("Obs Space", str(env.observation_space.shape), color="#FFE66D", icon="👁️")
with c4:
    metric_card("Action Space", str(env.action_space.shape), color="#FF6B6B", icon="🎮")

st.markdown("---")

# ── Environment State ────────────────────────────────────────────────────
st.markdown("#### Environment Observation")

obs, _ = env.reset()
obs_fig = go.Figure(go.Bar(
    x=list(range(len(obs))),
    y=obs,
    marker_color=["#4ECDC4" if i < 10 else "#FFE66D" if i < 20 else "#FF6B6B" if i < 30 else "#AA00FF"
                  for i in range(len(obs))],
))
obs_fig.update_layout(
    height=250, margin=dict(l=50, r=20, t=30, b=30),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
    font=dict(color="#ccc"), title="Observation Vector (44 dims)",
    xaxis_title="Feature Index", yaxis_title="Value (normalised 0–1)",
)
st.plotly_chart(obs_fig, use_container_width=True, key="rl_obs")

# Legend for observation space
st.markdown("""
<div style="display:flex;gap:20px;justify-content:center;margin-bottom:16px;">
    <span style="color:#4ECDC4;">● Storage (0-9)</span>
    <span style="color:#FFE66D;">● Load (10-19)</span>
    <span style="color:#FF6B6B;">● Frequency (20-29)</span>
    <span style="color:#AA00FF;">● Voltage (30-39)</span>
    <span style="color:#888;">● Global (40-43)</span>
</div>
""", unsafe_allow_html=True)

# ── Predict / Optimize ───────────────────────────────────────────────────
st.markdown("#### Optimization")

col_pred, col_act = st.columns([1, 2])

with col_pred:
    deterministic = st.checkbox("Deterministic (exploit)", value=True, key="rl_det")
    if st.button("🚀 Predict Actions", type="primary", key="rl_predict"):
        action, info = agent.predict(obs, deterministic=deterministic)
        st.session_state["rl_action"] = action
        st.session_state["rl_info"] = info

with col_act:
    action = st.session_state.get("rl_action")
    if action is not None:
        # Action bar chart
        fig_act = go.Figure(go.Bar(
            x=[f"Node {i}" for i in range(len(action))],
            y=action,
            marker_color=["#FF5252" if a < 0.4 else "#00C853" if a > 0.6 else "#888" for a in action],
        ))
        fig_act.add_hline(y=0.5, line_dash="dash", line_color="#555", annotation_text="Neutral")
        fig_act.update_layout(
            height=250, margin=dict(l=40, r=20, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
            font=dict(color="#ccc"), title="Agent Actions (0=Discharge, 0.5=Neutral, 1=Charge)",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_act, use_container_width=True, key="rl_act")

        st.markdown("""
        <div style="display:flex;gap:20px;justify-content:center;font-size:0.8rem;">
            <span style="color:#FF5252;">● Discharge</span>
            <span style="color:#888;">● Neutral</span>
            <span style="color:#00C853;">● Charge</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click **Predict Actions** to see the agent's recommendation.")

# ── Training controls ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Training")

col_t1, col_t2 = st.columns([1, 2])
with col_t1:
    timesteps = st.number_input("Total timesteps", 1000, 100000, 5000, 1000, key="rl_ts")
    if st.button("🏋️ Train Agent", key="rl_train"):
        with st.spinner(f"Training for {timesteps} timesteps…"):
            try:
                result = agent.train(total_timesteps=timesteps)
                st.session_state["rl_train_result"] = result
                st.success("Training complete!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

with col_t2:
    train_result = st.session_state.get("rl_train_result")
    if train_result:
        st.json(train_result)

# ── Evaluation ───────────────────────────────────────────────────────────
st.markdown("#### Evaluation")
n_eval = st.slider("Evaluation episodes", 1, 20, 5, key="rl_neval")
if st.button("📊 Evaluate", key="rl_eval"):
    with st.spinner("Evaluating…"):
        try:
            eval_result = agent.evaluate(n_episodes=n_eval)
            st.session_state["rl_eval_result"] = eval_result
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

eval_result = st.session_state.get("rl_eval_result")
if eval_result:
    ec1, ec2 = st.columns(2)
    with ec1:
        metric_card("Mean Reward", f"{eval_result.get('mean_reward', 0):.3f}", color="#00C853", icon="🏆")
    with ec2:
        metric_card("Std Reward", f"{eval_result.get('std_reward', 0):.3f}", color="#FF9100", icon="📊")

"""
Page 7 – RL Agent Dashboard
============================
Full PPO training, evaluation, and live-testing panel for the DEMS grid.

Features:
- One-click PPO training with configurable hyperparameters
- Real-time reward curves, loss plots, and grid-metric charts
- Model save / load / checkpoint management
- Live agent rollout visualisation (step-by-step episode replay)
- Evaluation statistics with comparison to random baseline
"""

import sys, os, json, time, glob
import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_app.simulation.state_manager import init_session
from streamlit_app.components.metrics_cards import metric_card
from streamlit_app.components.sidebar import render_sidebar

render_sidebar()

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(_root, "logs", "rl")
MODEL_PATH = os.path.join(MODEL_DIR, "dems_ppo_model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "dems_ppo_best")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Page header ───────────────────────────────────────────────────────────
st.markdown("## 🤖 RL Agent — PPO Energy Manager")
st.markdown("*Train a Proximal Policy Optimization agent to optimally manage grid storage, frequency, and voltage across all nodes.*")

# ══════════════════════════════════════════════════════════════════════════
# Lazy-init agent + env (cached across reruns)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _build_agent(algo="PPO", num_nodes=10, max_steps=200):
    """Build env + agent. Returns (env, agent, err_string|None)."""
    try:
        from src.agent.environment import DEMSEnvironment
        from src.agent.rl_agent import RLAgent
        env = DEMSEnvironment(num_nodes=num_nodes, max_steps=max_steps)
        agent = RLAgent(env, algorithm=algo, verbose=0)
        # Try to load existing trained model
        if os.path.exists(MODEL_PATH + ".zip"):
            agent.load(MODEL_PATH)
        return env, agent, None
    except Exception as exc:
        return None, None, str(exc)


def _load_history() -> dict:
    """Load training history from disk."""
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


env, agent, err = _build_agent()

if err:
    st.error(f"RL Agent could not be initialised: {err}")
    st.info("Install `stable-baselines3` and `gymnasium` to enable RL features:\n\n```pip install stable-baselines3 gymnasium```")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL STATUS CARDS
# ══════════════════════════════════════════════════════════════════════════

history = _load_history()
has_trained = os.path.exists(MODEL_PATH + ".zip")
total_eps = history.get("total_episodes", 0)
total_ts = history.get("total_timesteps", 0)
eval_mean = history.get("eval_mean_reward", None)
eval_std = history.get("eval_std_reward", None)
train_time = history.get("training_time_s", 0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Algorithm", "PPO", color="#AA00FF", icon="🧠")
with c2:
    metric_card("Status", "Trained" if has_trained else "Untrained",
                color="#00C853" if has_trained else "#FF9100", icon="📦")
with c3:
    metric_card("Episodes", str(total_eps), color="#4ECDC4", icon="🔄")
with c4:
    metric_card("Timesteps", f"{total_ts:,}", color="#FFE66D", icon="📈")
with c5:
    metric_card("Obs Space", str(env.observation_space.shape), color="#448AFF", icon="👁️")
with c6:
    metric_card("Act Space", str(env.action_space.shape), color="#FF6B6B", icon="🎮")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_train, tab_results, tab_rollout, tab_model, tab_compare = st.tabs([
    "🏋️ Training", "📊 Results & Metrics", "🎬 Live Rollout", "💾 Model Management", "🔋 Before vs After"
])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1: TRAINING
# ──────────────────────────────────────────────────────────────────────────
with tab_train:
    st.markdown("### Configure & Train PPO")

    col_cfg, col_run = st.columns([2, 3])

    with col_cfg:
        st.markdown("**Hyperparameters**")
        ts = st.number_input("Total timesteps", 2000, 500_000, 20_000, 2000, key="t_ts")
        lr = st.number_input("Learning rate", 1e-5, 1e-2, 3e-4, format="%.5f", key="t_lr")

        with st.expander("Advanced PPO settings", expanded=False):
            n_steps = st.number_input("Rollout steps (n_steps)", 256, 8192, 2048, 256, key="t_nsteps")
            batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=1, key="t_bs")
            n_epochs = st.slider("Epochs per update", 1, 30, 10, key="t_epochs")
            gamma = st.slider("Discount (γ)", 0.90, 1.0, 0.99, 0.01, key="t_gamma")
            gae_lambda = st.slider("GAE λ", 0.8, 1.0, 0.95, 0.01, key="t_gae")
            clip_range = st.slider("Clip range (ε)", 0.05, 0.5, 0.2, 0.05, key="t_clip")
            ckpt_freq = st.number_input("Checkpoint freq", 1000, 50_000, 5000, 1000, key="t_ckpt")

        resume = st.checkbox("Resume from saved model", value=has_trained, key="t_resume")

    with col_run:
        st.markdown("**Training**")

        if st.button("🚀 Start Training", type="primary", use_container_width=True, key="t_go"):
            # Clear cached agent so we get a fresh or resumed one
            _build_agent.clear()
            env_new, agent_new, err_new = _build_agent()
            if err_new:
                st.error(err_new)
            else:
                from src.agent.callbacks import DEMSTrainingCallback

                if resume and os.path.exists(MODEL_PATH + ".zip"):
                    agent_new.load(MODEL_PATH)
                    st.info("Resumed from saved model.")

                # Update hyperparams on the SB3 model
                if agent_new.model is not None:
                    agent_new.model.learning_rate = lr
                    agent_new.model.n_steps = n_steps
                    agent_new.model.batch_size = batch_size
                    agent_new.model.n_epochs = n_epochs
                    agent_new.model.gamma = gamma
                    agent_new.model.gae_lambda = gae_lambda
                    # clip_range must be a callable schedule for SB3
                    _cr = clip_range
                    agent_new.model.clip_range = lambda _prog, _cr=_cr: _cr

                cb = DEMSTrainingCallback(
                    log_dir=MODEL_DIR,
                    checkpoint_freq=ckpt_freq,
                    verbose=1,
                )

                progress = st.progress(0, text="Training PPO …")
                status_area = st.empty()

                t0 = time.time()

                # Train (blocking)
                try:
                    result = agent_new.train(total_timesteps=ts, callback=cb)
                    train_dur = time.time() - t0
                    progress.progress(100, text=f"Done in {train_dur:.1f}s")

                    # Save model
                    agent_new.save(MODEL_PATH)

                    # Evaluate
                    eval_res = agent_new.evaluate(n_episodes=10)

                    # Persist history
                    metrics = cb.get_metrics()
                    metrics["eval_mean_reward"] = eval_res.get("mean_reward", 0)
                    metrics["eval_std_reward"] = eval_res.get("std_reward", 0)
                    metrics["training_time_s"] = train_dur
                    metrics["model_path"] = MODEL_PATH
                    metrics["config"] = {
                        "total_timesteps": ts,
                        "learning_rate": lr,
                        "n_steps": n_steps,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "gamma": gamma,
                        "gae_lambda": gae_lambda,
                        "clip_range": clip_range,
                    }
                    with open(HISTORY_PATH, "w") as f:
                        json.dump(metrics, f, indent=2)

                    # Show success
                    status_area.success(
                        f"Training complete — {metrics['total_episodes']} episodes, "
                        f"mean reward {eval_res['mean_reward']:.3f} ± {eval_res['std_reward']:.3f}"
                    )

                    # Store in session for immediate display
                    st.session_state["rl_train_metrics"] = metrics
                    st.session_state["rl_eval_result"] = eval_res

                    # Clear cache to pick up new model
                    _build_agent.clear()

                except Exception as exc:
                    progress.progress(0, text="Failed")
                    st.error(f"Training failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

        # Show last training summary
        if has_trained:
            st.markdown("---")
            st.markdown("**Last Training Run**")
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                metric_card("Episodes", str(total_eps), color="#4ECDC4", icon="🔄")
            with lc2:
                metric_card("Time", f"{train_time:.1f}s", color="#FFE66D", icon="⏱️")
            with lc3:
                if eval_mean is not None:
                    metric_card("Eval Mean", f"{eval_mean:.3f}", color="#00C853", icon="🏆")
            with lc4:
                if eval_std is not None:
                    metric_card("Eval Std", f"{eval_std:.3f}", color="#FF9100", icon="📊")

            cfg = history.get("config", {})
            if cfg:
                with st.expander("Training config used"):
                    st.json(cfg)


# ──────────────────────────────────────────────────────────────────────────
# TAB 2: RESULTS & METRICS
# ──────────────────────────────────────────────────────────────────────────
with tab_results:
    st.markdown("### Training Metrics")

    # Use session state (fresh training) or disk
    m = st.session_state.get("rl_train_metrics") or _load_history()

    if not m or not m.get("episode_rewards"):
        st.info("No training data yet. Train the agent first!")
    else:
        ep_rewards = m["episode_rewards"]
        ep_mean = m.get("episode_mean_rewards", [])
        ep_lengths = m.get("episode_lengths", [])

        # ── Reward curve ────────────────────────────────────────────
        fig_reward = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Episode Reward", "Mean Reward (100-ep window)", "Episode Length", "Grid Metrics"),
            vertical_spacing=0.14, horizontal_spacing=0.08,
        )

        # Episode reward
        fig_reward.add_trace(go.Scatter(
            y=ep_rewards, mode="lines", name="Reward",
            line=dict(color="#4ECDC4", width=1),
            opacity=0.6,
        ), row=1, col=1)

        # Smoothed mean
        if ep_mean:
            fig_reward.add_trace(go.Scatter(
                y=ep_mean, mode="lines", name="Mean (100)",
                line=dict(color="#00C853", width=2.5),
            ), row=1, col=2)

        # Episode length
        if ep_lengths:
            fig_reward.add_trace(go.Scatter(
                y=ep_lengths, mode="lines", name="Ep Length",
                line=dict(color="#FFE66D", width=1.5),
            ), row=2, col=1)

        # Grid metrics — frequency penalty + storage ratio
        freq_pen = m.get("avg_frequency_penalties", [])
        stor_rat = m.get("avg_storage_ratios", [])
        if freq_pen:
            fig_reward.add_trace(go.Scatter(
                y=freq_pen, mode="lines", name="Freq Penalty",
                line=dict(color="#FF6B6B", width=1.5),
            ), row=2, col=2)
        if stor_rat:
            fig_reward.add_trace(go.Scatter(
                y=stor_rat, mode="lines", name="Storage Ratio",
                line=dict(color="#AA00FF", width=1.5),
            ), row=2, col=2)

        fig_reward.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.6)",
            font=dict(color="#ccc"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.06),
        )
        for i in range(1, 5):
            row, col = (i - 1) // 2 + 1, (i - 1) % 2 + 1
            fig_reward.update_xaxes(title_text="Episode", row=row, col=col, gridcolor="#222")
            fig_reward.update_yaxes(gridcolor="#222", row=row, col=col)

        st.plotly_chart(fig_reward, use_container_width=True, key="rl_metrics_fig")

        # ── Policy / Value loss ────────────────────────────────────
        pol_loss = m.get("policy_losses", [])
        val_loss = m.get("value_losses", [])
        entropies = m.get("entropies", [])

        if pol_loss or val_loss:
            fig_loss = make_subplots(rows=1, cols=3, subplot_titles=("Policy Loss", "Value Loss", "Entropy"))
            if pol_loss:
                fig_loss.add_trace(go.Scatter(y=pol_loss, mode="lines", line=dict(color="#FF6B6B")), row=1, col=1)
            if val_loss:
                fig_loss.add_trace(go.Scatter(y=val_loss, mode="lines", line=dict(color="#448AFF")), row=1, col=2)
            if entropies:
                fig_loss.add_trace(go.Scatter(y=entropies, mode="lines", line=dict(color="#AA00FF")), row=1, col=3)
            fig_loss.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.6)",
                font=dict(color="#ccc"),
                showlegend=False,
            )
            for c in range(1, 4):
                fig_loss.update_xaxes(title_text="Update", gridcolor="#222", row=1, col=c)
                fig_loss.update_yaxes(gridcolor="#222", row=1, col=c)
            st.plotly_chart(fig_loss, use_container_width=True, key="rl_loss_fig")

        # ── Summary stats ──────────────────────────────────────────
        st.markdown("#### Summary")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            metric_card("Final Mean", f"{ep_mean[-1]:.3f}" if ep_mean else "—", color="#00C853", icon="🏆")
        with sc2:
            metric_card("Max Reward", f"{max(ep_rewards):.3f}", color="#4ECDC4", icon="⬆️")
        with sc3:
            metric_card("Min Reward", f"{min(ep_rewards):.3f}", color="#FF5252", icon="⬇️")
        with sc4:
            improvement = ep_mean[-1] - ep_mean[0] if ep_mean and len(ep_mean) > 1 else 0
            metric_card("Improvement", f"{improvement:+.3f}", color="#FFE66D", icon="📈")


# ──────────────────────────────────────────────────────────────────────────
# TAB 3: LIVE ROLLOUT
# ──────────────────────────────────────────────────────────────────────────
with tab_rollout:
    st.markdown("### Live Agent Rollout")
    st.markdown("Run the trained agent through a full episode and visualise its decisions step-by-step.")

    roll_col1, roll_col2 = st.columns([1, 3])

    with roll_col1:
        n_rollout_steps = st.slider("Max steps", 50, 500, 200, 10, key="rl_roll_steps")
        deterministic = st.checkbox("Deterministic", value=True, key="rl_roll_det")
        compare_random = st.checkbox("Compare vs Random", value=True, key="rl_roll_cmp")
        run_rollout = st.button("▶ Run Episode", type="primary", key="rl_roll_go")

    with roll_col2:
        if run_rollout:
            # Run agent episode
            from src.agent.environment import DEMSEnvironment
            roll_env = DEMSEnvironment(num_nodes=10, max_steps=n_rollout_steps)

            # Reload trained agent
            from src.agent.rl_agent import RLAgent
            roll_agent = RLAgent(roll_env, algorithm="PPO", verbose=0)
            if os.path.exists(MODEL_PATH + ".zip"):
                roll_agent.load(MODEL_PATH)
                agent_label = "PPO (trained)"
            else:
                agent_label = "PPO (untrained)"

            def _run_episode(ag, env_inst, det=True):
                obs, _ = env_inst.reset()
                rewards, storages, freqs, volts, actions_hist = [], [], [], [], []
                done = False
                while not done:
                    action, _ = ag.predict(obs, deterministic=det)
                    obs, reward, terminated, truncated, info = env_inst.step(action)
                    done = terminated or truncated
                    rewards.append(reward)
                    storages.append(info.get("avg_storage_ratio", 0.5))
                    freqs.append(info.get("global_frequency", 50.0))
                    volts.append(info.get("global_voltage", 230.0))
                    actions_hist.append(action.copy())
                return rewards, storages, freqs, volts, actions_hist

            with st.spinner("Running agent episode …"):
                ag_rewards, ag_stor, ag_freq, ag_volt, ag_acts = _run_episode(roll_agent, roll_env, deterministic)

            rand_rewards, rand_stor, rand_freq, rand_volt = None, None, None, None
            if compare_random:
                # Random baseline
                class RandomAgent:
                    def __init__(self, act_space):
                        self.act_space = act_space
                    def predict(self, obs, deterministic=False):
                        return self.act_space.sample(), {}
                rand_ag = RandomAgent(roll_env.action_space)
                with st.spinner("Running random baseline …"):
                    rand_rewards, rand_stor, rand_freq, rand_volt, _ = _run_episode(rand_ag, roll_env)

            # Store for display
            st.session_state["rl_rollout"] = {
                "agent_label": agent_label,
                "ag_rewards": ag_rewards, "ag_stor": ag_stor,
                "ag_freq": ag_freq, "ag_volt": ag_volt, "ag_acts": ag_acts,
                "rand_rewards": rand_rewards, "rand_stor": rand_stor,
                "rand_freq": rand_freq, "rand_volt": rand_volt,
                "compare": compare_random,
            }

        # ── Display rollout results ──────────────────────────────────
        roll = st.session_state.get("rl_rollout")
        if roll:
            ag_rewards = roll["ag_rewards"]
            agent_label = roll["agent_label"]

            # Summary cards
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                metric_card("Total Reward", f"{sum(ag_rewards):.2f}", color="#00C853", icon="🏆")
            with rc2:
                metric_card("Avg Step Rwd", f"{np.mean(ag_rewards):.4f}", color="#4ECDC4", icon="📈")
            with rc3:
                metric_card("Avg Freq", f"{np.mean(roll['ag_freq']):.3f} Hz", color="#AA00FF", icon="〰️")
            with rc4:
                metric_card("Avg Storage", f"{np.mean(roll['ag_stor']):.3f}", color="#FFE66D", icon="🔋")

            if roll["compare"] and roll["rand_rewards"]:
                rrc1, rrc2 = st.columns(2)
                with rrc1:
                    diff = sum(ag_rewards) - sum(roll["rand_rewards"])
                    metric_card("vs Random", f"{diff:+.2f}", color="#00C853" if diff > 0 else "#FF5252", icon="⚔️")
                with rrc2:
                    metric_card("Random Total", f"{sum(roll['rand_rewards']):.2f}", color="#888", icon="🎲")

            # ── Charts ────────────────────────────────────────────────
            fig_roll = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Cumulative Reward", "Frequency (Hz)", "Storage Ratio", "Node Actions (last step)"),
                vertical_spacing=0.14, horizontal_spacing=0.08,
            )

            steps_x = list(range(len(ag_rewards)))

            # Cumulative reward
            cum_rew = np.cumsum(ag_rewards)
            fig_roll.add_trace(go.Scatter(
                x=steps_x, y=cum_rew, mode="lines", name=agent_label,
                line=dict(color="#00C853", width=2),
            ), row=1, col=1)
            if roll["compare"] and roll["rand_rewards"]:
                cum_rand = np.cumsum(roll["rand_rewards"])
                fig_roll.add_trace(go.Scatter(
                    x=list(range(len(cum_rand))), y=cum_rand,
                    mode="lines", name="Random",
                    line=dict(color="#FF5252", width=2, dash="dash"),
                ), row=1, col=1)

            # Frequency
            fig_roll.add_trace(go.Scatter(
                x=steps_x, y=roll["ag_freq"], mode="lines", name="Agent Freq",
                line=dict(color="#4ECDC4", width=1.5),
            ), row=1, col=2)
            fig_roll.add_hline(y=50.0, line_dash="dash", line_color="#555", row=1, col=2)
            if roll["compare"] and roll["rand_freq"]:
                fig_roll.add_trace(go.Scatter(
                    x=list(range(len(roll["rand_freq"]))), y=roll["rand_freq"],
                    mode="lines", name="Random Freq",
                    line=dict(color="#FF5252", width=1, dash="dot"),
                ), row=1, col=2)

            # Storage Ratio
            fig_roll.add_trace(go.Scatter(
                x=steps_x, y=roll["ag_stor"], mode="lines", name="Agent Storage",
                line=dict(color="#FFE66D", width=1.5),
            ), row=2, col=1)
            fig_roll.add_hline(y=0.5, line_dash="dash", line_color="#555", row=2, col=1)
            if roll["compare"] and roll["rand_stor"]:
                fig_roll.add_trace(go.Scatter(
                    x=list(range(len(roll["rand_stor"]))), y=roll["rand_stor"],
                    mode="lines", name="Random Storage",
                    line=dict(color="#FF5252", width=1, dash="dot"),
                ), row=2, col=1)

            # Last step actions
            last_act = roll["ag_acts"][-1] if roll["ag_acts"] else np.zeros(10)
            fig_roll.add_trace(go.Bar(
                x=[f"Node {i}" for i in range(len(last_act))],
                y=last_act,
                marker_color=["#FF5252" if a < 0.4 else "#00C853" if a > 0.6 else "#888" for a in last_act],
                name="Actions",
            ), row=2, col=2)
            fig_roll.add_hline(y=0.5, line_dash="dash", line_color="#555", row=2, col=2)

            fig_roll.update_layout(
                height=620,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.6)",
                font=dict(color="#ccc"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.06),
            )
            for i in range(1, 5):
                r, c = (i - 1) // 2 + 1, (i - 1) % 2 + 1
                fig_roll.update_xaxes(gridcolor="#222", row=r, col=c)
                fig_roll.update_yaxes(gridcolor="#222", row=r, col=c)

            st.plotly_chart(fig_roll, use_container_width=True, key="rl_rollout_fig")

            # Action legend
            st.markdown("""
            <div style="display:flex;gap:20px;justify-content:center;font-size:0.8rem;margin-top:-8px;">
                <span style="color:#FF5252;">● Discharge (< 0.4)</span>
                <span style="color:#888;">● Neutral (0.4–0.6)</span>
                <span style="color:#00C853;">● Charge (> 0.6)</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Action heatmap over time ──────────────────────────────
            if roll["ag_acts"] and len(roll["ag_acts"]) > 1:
                with st.expander("Action heatmap over time", expanded=False):
                    act_matrix = np.array(roll["ag_acts"])
                    fig_heat = go.Figure(go.Heatmap(
                        z=act_matrix.T,
                        x=list(range(act_matrix.shape[0])),
                        y=[f"Node {i}" for i in range(act_matrix.shape[1])],
                        colorscale=[
                            [0, "#FF5252"], [0.4, "#FF5252"],
                            [0.4, "#333"], [0.6, "#333"],
                            [0.6, "#00C853"], [1.0, "#00C853"],
                        ],
                        zmin=0, zmax=1,
                        colorbar=dict(title="Action", tickvals=[0, 0.5, 1], ticktext=["Discharge", "Neutral", "Charge"]),
                    ))
                    fig_heat.update_layout(
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(14,17,23,0.6)",
                        font=dict(color="#ccc"),
                        xaxis_title="Step", yaxis_title="Node",
                    )
                    st.plotly_chart(fig_heat, use_container_width=True, key="rl_heatmap")
        else:
            st.info("Click **Run Episode** to see the agent in action.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 4: MODEL MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown("### Model Management")

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**Saved Models**")
        # List checkpoints + final model
        models = sorted(glob.glob(os.path.join(MODEL_DIR, "*.zip")))
        if models:
            for mp in models:
                fname = os.path.basename(mp)
                size_mb = os.path.getsize(mp) / (1024 * 1024)
                st.markdown(
                    f'<div style="padding:6px 12px;background:#1a1a2e;border-radius:8px;margin-bottom:4px;">'
                    f'<span style="color:#4ECDC4;">📦</span> '
                    f'<span style="color:#eee;">{fname}</span> '
                    f'<span style="color:#888;font-size:0.8rem;">({size_mb:.1f} MB)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No saved models yet.")

    with mc2:
        st.markdown("**Actions**")

        # Load a specific model
        if models:
            sel_model = st.selectbox("Select model to load", [os.path.basename(m) for m in models], key="m_sel")
            if st.button("📥 Load Selected Model", key="m_load"):
                sel_path = os.path.join(MODEL_DIR, sel_model.replace(".zip", ""))
                try:
                    _build_agent.clear()
                    env_l, agent_l, err_l = _build_agent()
                    if agent_l and not err_l:
                        agent_l.load(sel_path)
                        agent_l.save(MODEL_PATH)  # Make it the active model
                        st.success(f"Loaded {sel_model} as active model.")
                        _build_agent.clear()
                except Exception as exc:
                    st.error(f"Failed to load: {exc}")

        # Evaluate
        st.markdown("---")
        n_eval = st.slider("Evaluation episodes", 1, 50, 10, key="m_neval")
        if st.button("📊 Evaluate Active Model", key="m_eval"):
            _build_agent.clear()
            env_e, agent_e, err_e = _build_agent()
            if err_e:
                st.error(err_e)
            else:
                with st.spinner(f"Evaluating over {n_eval} episodes …"):
                    try:
                        ev = agent_e.evaluate(n_episodes=n_eval)
                        st.session_state["rl_model_eval"] = ev
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")

        ev = st.session_state.get("rl_model_eval")
        if ev:
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                metric_card("Mean Reward", f"{ev['mean_reward']:.3f}", color="#00C853", icon="🏆")
            with ec2:
                metric_card("Std Reward", f"{ev['std_reward']:.3f}", color="#FF9100", icon="📊")
            with ec3:
                metric_card("Episodes", str(ev['n_episodes']), color="#4ECDC4", icon="🔄")

    # ── Training history JSON viewer ──────────────────────────────
    st.markdown("---")
    with st.expander("Raw training history (JSON)", expanded=False):
        h = _load_history()
        if h:
            # Don't dump huge arrays, show summary
            display = {k: v for k, v in h.items() if not isinstance(v, list)}
            display["episodes_recorded"] = len(h.get("episode_rewards", []))
            st.json(display)
        else:
            st.info("No training history available.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 5: BEFORE vs AFTER (Battery-Centric)
# ──────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### 🔋 Before vs After RL — Battery Control Comparison")
    st.markdown(
        "Run two identical episodes (same random seed) — one with **random control**, "
        "one with the **trained PPO agent** — and see exactly how battery decisions differ."
    )

    cmp_col1, cmp_col2 = st.columns([1, 4])
    with cmp_col1:
        cmp_steps = st.slider("Episode length", 50, 500, 200, 10, key="cmp_steps")
        cmp_seed = st.number_input("Random seed", 0, 9999, 42, key="cmp_seed")
        run_compare = st.button("⚡ Run Comparison", type="primary", key="cmp_go")

    if run_compare:
        from src.agent.environment import DEMSEnvironment
        from src.agent.rl_agent import RLAgent

        def _run_full_episode(agent_obj, env_obj, seed, det=True):
            """Run one episode, collecting per-node battery data every step."""
            obs, _ = env_obj.reset(seed=seed)
            data = {
                "rewards": [], "actions": [], "storage_levels": [],
                "storage_ratios": [], "demands": [], "charge_discharge": [],
                "frequencies": [], "voltages": [], "supply_demand_errors": [],
            }
            done = False
            while not done:
                action, _ = agent_obj.predict(obs, deterministic=det)
                obs, reward, terminated, truncated, info = env_obj.step(action)
                done = terminated or truncated
                data["rewards"].append(reward)
                data["actions"].append(action.copy())
                data["storage_levels"].append(info["storage_levels"].copy())
                data["storage_ratios"].append(info["avg_storage_ratio"])
                data["demands"].append(info["demands"].copy())
                data["charge_discharge"].append(info["charge_discharge"].copy())
                data["frequencies"].append(info["global_frequency"])
                data["voltages"].append(info["global_voltage"])
                data["supply_demand_errors"].append(info.get("supply_demand_error", 0.0))
            return data

        # Random agent
        class _RandomAgent:
            def __init__(self, act_space):
                self.act_space = act_space
            def predict(self, obs, deterministic=False):
                return self.act_space.sample(), {}

        # Build envs & agents
        env_rand = DEMSEnvironment(num_nodes=10, max_steps=cmp_steps)
        env_ppo = DEMSEnvironment(num_nodes=10, max_steps=cmp_steps)
        rand_agent = _RandomAgent(env_rand.action_space)
        ppo_agent = RLAgent(env_ppo, algorithm="PPO", verbose=0)
        if os.path.exists(MODEL_PATH + ".zip"):
            ppo_agent.load(MODEL_PATH)

        with st.spinner("Running random baseline …"):
            d_rand = _run_full_episode(rand_agent, env_rand, seed=cmp_seed)
        with st.spinner("Running PPO agent …"):
            d_ppo = _run_full_episode(ppo_agent, env_ppo, seed=cmp_seed, det=True)

        st.session_state["cmp_rand"] = d_rand
        st.session_state["cmp_ppo"] = d_ppo

    # ── Display comparison ────────────────────────────────────────────
    d_rand = st.session_state.get("cmp_rand")
    d_ppo = st.session_state.get("cmp_ppo")

    if d_rand and d_ppo:
        # ═══════════════════════  SECTION 1: SUMMARY CARDS  ════════════════
        st.markdown("---")
        st.markdown("#### Summary Comparison")

        r_total = sum(d_rand["rewards"])
        p_total = sum(d_ppo["rewards"])
        reward_pct = ((p_total - r_total) / (abs(r_total) + 1e-9)) * 100

        r_soc_mean = np.mean(d_rand["storage_ratios"])
        p_soc_mean = np.mean(d_ppo["storage_ratios"])
        r_soc_std = np.std([s for sl in d_rand["storage_levels"] for s in sl] ) / 100.0
        p_soc_std = np.std([s for sl in d_ppo["storage_levels"] for s in sl]) / 100.0
        r_freq_dev = np.mean([abs(f - 50.0) for f in d_rand["frequencies"]])
        p_freq_dev = np.mean([abs(f - 50.0) for f in d_ppo["frequencies"]])
        r_sd_err = np.mean(d_rand["supply_demand_errors"])
        p_sd_err = np.mean(d_ppo["supply_demand_errors"])

        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            metric_card("Reward", f"{p_total:.1f}", delta=p_total - r_total, color="#00C853", icon="🏆")
            st.caption(f"Random: {r_total:.1f}")
        with s2:
            metric_card("Avg SoC", f"{p_soc_mean:.3f}", delta=-(abs(p_soc_mean - 0.5) - abs(r_soc_mean - 0.5)), color="#4ECDC4", icon="🔋")
            st.caption(f"Random: {r_soc_mean:.3f} | Target: 0.500")
        with s3:
            metric_card("SoC Spread", f"{p_soc_std:.3f}", delta=-(p_soc_std - r_soc_std), color="#FFE66D", icon="📏")
            st.caption(f"Random: {r_soc_std:.3f} | Lower = steadier")
        with s4:
            metric_card("Freq Dev", f"{p_freq_dev:.4f} Hz", delta=-(p_freq_dev - r_freq_dev), color="#AA00FF", icon="〰️")
            st.caption(f"Random: {r_freq_dev:.4f} Hz")
        with s5:
            metric_card("S-D Error", f"{p_sd_err:.4f}", delta=-(p_sd_err - r_sd_err), color="#448AFF", icon="⚖️")
            st.caption(f"Random: {r_sd_err:.4f}")

        # ═══════════════════════  SECTION 2: PER-NODE SOC  ═════════════════
        st.markdown("---")
        st.markdown("#### Per-Node Battery SoC Over Time")

        soc_rand = np.array(d_rand["storage_levels"]) / 100.0  # normalize to 0–1
        soc_ppo = np.array(d_ppo["storage_levels"]) / 100.0
        steps_x = list(range(soc_rand.shape[0]))
        node_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#AA00FF", "#448AFF",
                       "#FF9100", "#00C853", "#E040FB", "#00BCD4", "#FF5252"]

        fig_soc = make_subplots(rows=1, cols=2, subplot_titles=("Random Control", "PPO Agent"),
                                shared_yaxes=True, horizontal_spacing=0.04)
        for n in range(soc_rand.shape[1]):
            fig_soc.add_trace(go.Scatter(
                x=steps_x, y=soc_rand[:, n], mode="lines", name=f"Node {n}",
                line=dict(color=node_colors[n % len(node_colors)], width=1.2),
                legendgroup=f"n{n}", showlegend=True,
            ), row=1, col=1)
            fig_soc.add_trace(go.Scatter(
                x=steps_x, y=soc_ppo[:, n], mode="lines", name=f"Node {n}",
                line=dict(color=node_colors[n % len(node_colors)], width=1.2),
                legendgroup=f"n{n}", showlegend=False,
            ), row=1, col=2)
        # 50% target line
        for c in [1, 2]:
            fig_soc.add_hline(y=0.5, line_dash="dash", line_color="#555", row=1, col=c)
        fig_soc.update_yaxes(title_text="SoC (0–1)", range=[0, 1], gridcolor="#222", row=1, col=1)
        fig_soc.update_yaxes(range=[0, 1], gridcolor="#222", row=1, col=2)
        fig_soc.update_xaxes(title_text="Step", gridcolor="#222")
        fig_soc.update_layout(
            height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
            font=dict(color="#ccc"), margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_soc, use_container_width=True, key="cmp_soc")

        # ═══════════════════════  SECTION 3: ACTION HEATMAPS  ══════════════
        st.markdown("---")
        st.markdown("#### Charge / Discharge Actions Over Time")

        act_rand = np.array(d_rand["actions"])
        act_ppo = np.array(d_ppo["actions"])

        action_colorscale = [
            [0, "#FF5252"], [0.35, "#FF5252"],
            [0.35, "#333333"], [0.65, "#333333"],
            [0.65, "#00C853"], [1.0, "#00C853"],
        ]

        fig_acts = make_subplots(rows=1, cols=2, subplot_titles=("Random Actions", "PPO Actions"),
                                 horizontal_spacing=0.06)
        fig_acts.add_trace(go.Heatmap(
            z=act_rand.T, x=steps_x, y=[f"Node {i}" for i in range(10)],
            colorscale=action_colorscale, zmin=0, zmax=1, showscale=False,
        ), row=1, col=1)
        fig_acts.add_trace(go.Heatmap(
            z=act_ppo.T, x=steps_x, y=[f"Node {i}" for i in range(10)],
            colorscale=action_colorscale, zmin=0, zmax=1,
            colorbar=dict(title="Action", tickvals=[0, 0.5, 1], ticktext=["Discharge", "Neutral", "Charge"]),
        ), row=1, col=2)
        fig_acts.update_xaxes(title_text="Step", gridcolor="#222")
        fig_acts.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
            font=dict(color="#ccc"), margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_acts, use_container_width=True, key="cmp_acts")

        # ═══════════════════════  SECTION 4: BATTERY UTILIZATION  ══════════
        st.markdown("---")
        st.markdown("#### Battery Utilization Efficiency")
        st.markdown("*How much energy each node's battery usefully delivered vs wasted.*")

        cd_rand = np.array(d_rand["charge_discharge"])  # (steps, nodes)
        cd_ppo = np.array(d_ppo["charge_discharge"])
        dem_rand = np.array(d_rand["demands"])
        dem_ppo = np.array(d_ppo["demands"])

        # Useful discharge = discharge that went toward meeting demand
        # Wasted = |charge when already near full| + |discharge when demand already met|
        def _utilization(cd, dem, storage):
            storage = np.array(storage)
            useful = np.zeros(cd.shape[1])
            wasted = np.zeros(cd.shape[1])
            for t in range(cd.shape[0]):
                for n in range(cd.shape[1]):
                    pwr = cd[t, n]
                    soc = storage[t, n] / 100.0
                    if pwr < 0:  # discharging
                        useful[n] += min(abs(pwr), dem[t, n])
                        wasted[n] += max(0, abs(pwr) - dem[t, n])
                    else:  # charging
                        headroom = 1.0 - soc
                        if headroom < 0.05:
                            wasted[n] += abs(pwr)
                        else:
                            useful[n] += abs(pwr)
            return useful, wasted

        u_rand, w_rand = _utilization(cd_rand, dem_rand, d_rand["storage_levels"])
        u_ppo, w_ppo = _utilization(cd_ppo, dem_ppo, d_ppo["storage_levels"])

        fig_util = make_subplots(rows=1, cols=2, subplot_titles=("Random", "PPO"),
                                  shared_yaxes=True, horizontal_spacing=0.06)
        nodes = [f"Node {i}" for i in range(10)]
        for col_idx, (useful, wasted, label) in enumerate([
            (u_rand, w_rand, "Random"), (u_ppo, w_ppo, "PPO")
        ], 1):
            fig_util.add_trace(go.Bar(x=nodes, y=useful, name="Useful",
                                       marker_color="#00C853", legendgroup="useful",
                                       showlegend=(col_idx == 1)), row=1, col=col_idx)
            fig_util.add_trace(go.Bar(x=nodes, y=wasted, name="Wasted",
                                       marker_color="#FF5252", legendgroup="wasted",
                                       showlegend=(col_idx == 1)), row=1, col=col_idx)
        fig_util.update_layout(
            barmode="stack", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
            font=dict(color="#ccc"), margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(orientation="h", y=-0.15),
        )
        fig_util.update_yaxes(title_text="Energy (kWh)", gridcolor="#222", row=1, col=1)
        fig_util.update_yaxes(gridcolor="#222", row=1, col=2)
        fig_util.update_xaxes(gridcolor="#222", tickangle=-45)
        st.plotly_chart(fig_util, use_container_width=True, key="cmp_util")

        eff_rand = np.sum(u_rand) / (np.sum(u_rand) + np.sum(w_rand) + 1e-9) * 100
        eff_ppo = np.sum(u_ppo) / (np.sum(u_ppo) + np.sum(w_ppo) + 1e-9) * 100
        ue1, ue2 = st.columns(2)
        with ue1:
            metric_card("Random Efficiency", f"{eff_rand:.1f}%", color="#FF5252", icon="🎲")
        with ue2:
            metric_card("PPO Efficiency", f"{eff_ppo:.1f}%", color="#00C853", icon="🧠")

        # ═══════════════════════  SECTION 5: GRID STABILITY  ═══════════════
        st.markdown("---")
        st.markdown("#### Grid Stability Consequences")

        fig_grid = make_subplots(rows=1, cols=2,
                                  subplot_titles=("Frequency Stability", "Supply-Demand Error"),
                                  horizontal_spacing=0.08)

        # Frequency
        fig_grid.add_trace(go.Scatter(
            x=steps_x, y=d_rand["frequencies"], mode="lines", name="Random",
            line=dict(color="#FF5252", width=1.5),
        ), row=1, col=1)
        fig_grid.add_trace(go.Scatter(
            x=steps_x, y=d_ppo["frequencies"], mode="lines", name="PPO",
            line=dict(color="#00C853", width=2),
        ), row=1, col=1)
        fig_grid.add_hline(y=50.0, line_dash="dash", line_color="#555", row=1, col=1)
        # Violation bands
        fig_grid.add_hrect(y0=49.5, y1=49.7, fillcolor="rgba(255,82,82,0.08)",
                           line_width=0, row=1, col=1)
        fig_grid.add_hrect(y0=50.3, y1=50.5, fillcolor="rgba(255,82,82,0.08)",
                           line_width=0, row=1, col=1)

        # Supply-demand error
        fig_grid.add_trace(go.Scatter(
            x=steps_x, y=d_rand["supply_demand_errors"], mode="lines", name="Random",
            line=dict(color="#FF5252", width=1.5), showlegend=False,
        ), row=1, col=2)
        fig_grid.add_trace(go.Scatter(
            x=steps_x, y=d_ppo["supply_demand_errors"], mode="lines", name="PPO",
            line=dict(color="#00C853", width=2), showlegend=False,
        ), row=1, col=2)

        fig_grid.update_yaxes(title_text="Frequency (Hz)", gridcolor="#222", row=1, col=1)
        fig_grid.update_yaxes(title_text="|Error| (ratio)", gridcolor="#222", row=1, col=2)
        fig_grid.update_xaxes(title_text="Step", gridcolor="#222")
        fig_grid.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
            font=dict(color="#ccc"), margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_grid, use_container_width=True, key="cmp_grid")

        # ═══════════════════════  SECTION 6: DISTRIBUTIONS  ════════════════
        st.markdown("---")
        st.markdown("#### Distribution Comparison")

        fig_dist = make_subplots(rows=1, cols=2,
                                  subplot_titles=("Per-Step Reward Distribution", "Battery SoC Distribution"),
                                  horizontal_spacing=0.08)

        # Reward distribution
        fig_dist.add_trace(go.Histogram(
            x=d_rand["rewards"], name="Random", marker_color="#FF5252",
            opacity=0.6, nbinsx=30,
        ), row=1, col=1)
        fig_dist.add_trace(go.Histogram(
            x=d_ppo["rewards"], name="PPO", marker_color="#00C853",
            opacity=0.6, nbinsx=30,
        ), row=1, col=1)

        # SoC distribution (all nodes × all steps flattened)
        soc_flat_rand = soc_rand.flatten()
        soc_flat_ppo = soc_ppo.flatten()
        fig_dist.add_trace(go.Histogram(
            x=soc_flat_rand, name="Random", marker_color="#FF5252",
            opacity=0.6, nbinsx=40, showlegend=False,
        ), row=1, col=2)
        fig_dist.add_trace(go.Histogram(
            x=soc_flat_ppo, name="PPO", marker_color="#00C853",
            opacity=0.6, nbinsx=40, showlegend=False,
        ), row=1, col=2)

        fig_dist.update_xaxes(title_text="Reward", gridcolor="#222", row=1, col=1)
        fig_dist.update_xaxes(title_text="SoC (0–1)", gridcolor="#222", row=1, col=2)
        fig_dist.update_yaxes(title_text="Count", gridcolor="#222")
        fig_dist.update_layout(
            barmode="overlay", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
            font=dict(color="#ccc"), margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="cmp_dist")

        # ═══════════════════════  SECTION 7: KEY INSIGHTS  ═════════════════
        st.markdown("---")
        st.markdown("#### Key Insights")

        freq_improve = ((r_freq_dev - p_freq_dev) / (r_freq_dev + 1e-9)) * 100
        soc_improve = ((abs(r_soc_mean - 0.5) - abs(p_soc_mean - 0.5)) / (abs(r_soc_mean - 0.5) + 1e-9)) * 100
        sd_improve = ((r_sd_err - p_sd_err) / (r_sd_err + 1e-9)) * 100

        insights = f"""
        | Metric | Random | PPO | Improvement |
        |---|---|---|---|
        | **Total Reward** | {r_total:.1f} | {p_total:.1f} | **{reward_pct:+.1f}%** |
        | **Avg SoC** (target 0.50) | {r_soc_mean:.3f} | {p_soc_mean:.3f} | **{soc_improve:+.1f}%** closer to target |
        | **Frequency Deviation** | {r_freq_dev:.4f} Hz | {p_freq_dev:.4f} Hz | **{freq_improve:+.1f}%** reduction |
        | **Supply-Demand Error** | {r_sd_err:.4f} | {p_sd_err:.4f} | **{sd_improve:+.1f}%** reduction |
        | **Battery Efficiency** | {eff_rand:.1f}% | {eff_ppo:.1f}% | **{eff_ppo - eff_rand:+.1f}pp** |
        """
        st.markdown(insights)

        # Auto-generated prose
        st.success(
            f"The PPO agent achieved **{reward_pct:+.1f}%** higher total reward, "
            f"reduced frequency deviation by **{freq_improve:.0f}%**, "
            f"kept battery SoC **{soc_improve:.0f}%** closer to the optimal 50% level, "
            f"and improved battery utilization efficiency from {eff_rand:.0f}% to **{eff_ppo:.0f}%** "
            f"compared to random (uncontrolled) battery dispatch."
        )

    else:
        st.info("Click **Run Comparison** to generate a side-by-side before vs after analysis.")

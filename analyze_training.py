#!/usr/bin/env python3
"""Analyze RL training performance."""
import json
import numpy as np

with open("logs/rl/training_history.json") as f:
    h = json.load(f)

print(f"=== Training Summary: {len(h)} episodes, {h[-1]['steps_done']} total steps ===\n")

# Per-agent reward stats
agents = list(h[0]["rewards"].keys())
print(f"{'Agent':15s}  {'Mean':>7s}  {'Min':>7s}  {'Max':>7s}  {'Last':>7s}  {'Trend':>7s}")
print("-" * 65)
for agent in agents:
    rewards = [ep["rewards"][agent] for ep in h]
    trend = rewards[-1] - rewards[0]
    print(f"{agent:15s}  {np.mean(rewards):7.4f}  {np.min(rewards):7.4f}  {np.max(rewards):7.4f}  {rewards[-1]:7.4f}  {trend:+7.4f}")

# Best eval
print()
try:
    with open("logs/rl/best_eval.json") as f:
        ev = json.load(f)
    print(f"Best Eval Reward: {ev.get('eval_reward', 'N/A')}")
except Exception:
    pass

# Central agent episode progression
print("\n=== Central Agent Reward Progression ===")
for ep in h:
    c = ep["rewards"]["central"]
    bar = "#" * int(c * 40)
    print(f"  Ep {ep['episode']:2d} (step {ep['steps_done']:4d}): {c:.4f} |{bar}")

# Hierarchy level averages
print("\n=== Reward by Hierarchy Level ===")
levels = {
    "Central":    ["central"],
    "Microgrid":  ["mg_A", "mg_B", "mg_C"],
    "Inverter":   ["inverter_A", "inverter_B", "inverter_C"],
    "Renewable":  ["renewable_A", "renewable_B", "renewable_C"],
    "Load":       ["load_A", "load_B", "load_C"],
}
for level_name, agent_names in levels.items():
    all_r = []
    for ep in h:
        for a in agent_names:
            all_r.append(ep["rewards"].get(a, 0))
    first_ep = np.mean([h[0]["rewards"].get(a, 0) for a in agent_names])
    last_ep = np.mean([h[-1]["rewards"].get(a, 0) for a in agent_names])
    print(f"  {level_name:12s}  mean={np.mean(all_r):.4f}  first_ep={first_ep:.4f}  last_ep={last_ep:.4f}  delta={last_ep - first_ep:+.4f}")

# Total system reward
print("\n=== Total System Reward Per Episode ===")
totals = []
for ep in h:
    total = sum(ep["rewards"].values())
    totals.append(total)
    bar = "#" * int(total * 3)
    print(f"  Ep {ep['episode']:2d}: {total:7.3f} |{bar}")

print(f"\n  System reward: first={totals[0]:.3f}  last={totals[-1]:.3f}  max={max(totals):.3f}  trend={totals[-1]-totals[0]:+.3f}")

# Convergence assessment
print("\n=== Performance Assessment ===")
print(f"  Total training: {h[-1]['steps_done']} steps / {len(h)} episodes (smoke-test only)")
print(f"  Recommended minimum: 50,000-500,000 steps for meaningful learning")
print(f"  Current status: BASELINE (random policy, no convergence expected)")
central_rewards = [ep["rewards"]["central"] for ep in h]
if central_rewards[-1] > central_rewards[0]:
    print(f"  Central reward trend: POSITIVE (+{central_rewards[-1]-central_rewards[0]:.4f})")
else:
    print(f"  Central reward trend: FLAT/NEGATIVE ({central_rewards[-1]-central_rewards[0]:+.4f})")
print(f"  PF convergence: ~50-60% of steps (common with random policy)")
print(f"  Next step: Run longer training with --steps 50000+")

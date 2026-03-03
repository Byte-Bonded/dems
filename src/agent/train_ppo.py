#!/usr/bin/env python3
"""
DEMS PPO Training Script
========================
Train a PPO agent on the DEMSEnvironment and save model + metrics.

Usage:
    cd dems/
    python3 -m src.agent.train_ppo                      # defaults
    python3 -m src.agent.train_ppo --timesteps 50000    # custom
    python3 -m src.agent.train_ppo --resume              # resume from checkpoint
"""

import argparse
import os
import sys
import json
import time

# Ensure project root is importable
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from src.agent.environment import DEMSEnvironment
from src.agent.rl_agent import RLAgent
from src.agent.callbacks import DEMSTrainingCallback


# ── Default paths ─────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(_root, "logs", "rl")
MODEL_PATH = os.path.join(MODEL_DIR, "dems_ppo_model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "dems_ppo_best")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")


def make_env(num_nodes: int = 10, max_steps: int = 200) -> DEMSEnvironment:
    """Create and return a DEMSEnvironment."""
    return DEMSEnvironment(num_nodes=num_nodes, max_steps=max_steps)


def train(
    total_timesteps: int = 100_000,
    num_nodes: int = 10,
    max_steps: int = 200,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    checkpoint_freq: int = 10_000,
    resume: bool = False,
    verbose: int = 1,
) -> dict:
    """
    Train a PPO agent on DEMS.

    Returns a dict with training results and file paths.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("  DEMS PPO Training")
    print("=" * 60)
    print(f"  Timesteps     : {total_timesteps:,}")
    print(f"  Nodes         : {num_nodes}")
    print(f"  Max steps/ep  : {max_steps}")
    print(f"  Learning rate : {learning_rate}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Checkpoint    : every {checkpoint_freq} steps")
    print(f"  Resume        : {resume}")
    print("=" * 60)

    # 1. Create environment
    env = make_env(num_nodes=num_nodes, max_steps=max_steps)
    print(f"  Obs space  : {env.observation_space.shape}")
    print(f"  Act space  : {env.action_space.shape}")

    # 2. Create agent
    agent = RLAgent(
        env=env,
        algorithm="PPO",
        learning_rate=learning_rate,
        verbose=verbose,
        tensorboard_log=None,  # disable TB to avoid ImportError if not installed
    )

    # 2b. Resume from checkpoint if requested
    if resume and os.path.exists(MODEL_PATH + ".zip"):
        print(f"  Resuming from {MODEL_PATH}")
        agent.load(MODEL_PATH)

    # 3. Set PPO hyper-params on the underlying model
    if agent.model is not None:
        agent.model.n_steps = n_steps
        agent.model.batch_size = batch_size
        agent.model.n_epochs = n_epochs
        agent.model.gamma = gamma
        agent.model.gae_lambda = gae_lambda
        agent.model.ent_coef = ent_coef
        # clip_range must be a callable (schedule) for SB3
        agent.model.clip_range = lambda _progress: clip_range

    # 4. Create callback
    callback = DEMSTrainingCallback(
        log_dir=MODEL_DIR,
        checkpoint_freq=checkpoint_freq,
        verbose=verbose,
    )

    # 5. Train
    t0 = time.time()
    result = agent.train(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - t0

    print(f"\n  Training completed in {train_time:.1f}s")

    # 6. Save final model
    agent.save(MODEL_PATH)
    print(f"  Model saved to {MODEL_PATH}")

    # 7. Evaluate
    print("\n  Evaluating (10 episodes) ...")
    eval_result = agent.evaluate(n_episodes=10)
    print(f"  Mean reward: {eval_result.get('mean_reward', 0):.3f} ± {eval_result.get('std_reward', 0):.3f}")

    # Save best model if this is the best evaluation so far
    prev_best = -float("inf")
    best_path = os.path.join(MODEL_DIR, "best_eval.json")
    if os.path.exists(best_path):
        try:
            with open(best_path) as f:
                prev_best = json.load(f).get("mean_reward", -float("inf"))
        except Exception:
            pass

    current_mean = eval_result.get("mean_reward", -float("inf"))
    if current_mean > prev_best:
        agent.save(BEST_MODEL_PATH)
        with open(best_path, "w") as f:
            json.dump(eval_result, f, indent=2)
        print(f"  New best model saved! ({current_mean:.3f} > {prev_best:.3f})")

    # 8. Save training history (callback already saves, but add eval info)
    metrics = callback.get_metrics()
    metrics["eval_mean_reward"] = eval_result.get("mean_reward", 0)
    metrics["eval_std_reward"] = eval_result.get("std_reward", 0)
    metrics["training_time_s"] = train_time
    metrics["model_path"] = MODEL_PATH
    metrics["config"] = {
        "total_timesteps": total_timesteps,
        "num_nodes": num_nodes,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  History saved to {HISTORY_PATH}")

    return {
        "status": "success",
        "training_result": result,
        "eval_result": eval_result,
        "training_time_s": train_time,
        "model_path": MODEL_PATH,
        "history_path": HISTORY_PATH,
        "total_episodes": metrics.get("total_episodes", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train DEMS PPO Agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--nodes", type=int, default=10, help="Number of grid nodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Checkpoint frequency")
    parser.add_argument("--resume", action="store_true", help="Resume from previous model")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity (0/1/2)")
    args = parser.parse_args()

    result = train(
        total_timesteps=args.timesteps,
        num_nodes=args.nodes,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        checkpoint_freq=args.checkpoint_freq,
        resume=args.resume,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick training launcher for the Hierarchical Multi-Agent PPO system.

Usage:
    python3 train.py                       # short smoke-test (1000 steps)
    python3 train.py --steps 500000        # full training
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress noisy loggers for training runs
for _noisy in (
    "pandapower", "src.simulation.der", "src.simulation.dynamics",
    "src.simulation.power_flow", "src.simulation.supergrid",
    "src.simulation.orchestrator",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="DEMS Hierarchical PPO Training")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Total training timesteps (default: 1000 for smoke-test)")
    parser.add_argument("--eval-freq", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length", type=int, default=48,
                        help="Episode length in steps (default: 48 = 4 h at 5-min)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Log directory (default: <project>/logs/rl)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device: auto (detect GPU), cuda, mps, cpu")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for all agents (auto-scaled for GPU)")
    parser.add_argument("--vram-limit", type=float, default=7.0,
                        help="GPU VRAM hard cap in GB (default: 7.0, 0=unlimited)")
    args = parser.parse_args()

    # Resolve log-dir relative to this script's location
    project_root = Path(__file__).resolve().parent
    log_dir = args.log_dir or str(project_root / "logs" / "rl")

    logger.info("=" * 60)
    logger.info("DEMS Hierarchical Multi-Agent PPO — Training")
    logger.info("=" * 60)

    # GPU detection & info
    try:
        import torch
        from src.agent.ppo_agents import detect_device
        device = detect_device(args.device)
        logger.info(f"PyTorch {torch.__version__} | Device: {device}")
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
            logger.info(f"CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
            logger.info(f"VRAM cap: {args.vram_limit:.1f} GB / {gpu_mem:.1f} GB total")
        elif device == "mps":
            logger.info("Apple Metal Performance Shaders (MPS) backend")
        else:
            logger.info("CPU-only training (no GPU detected)")
    except ImportError:
        device = "cpu"
        logger.info("PyTorch not found — CPU-only training")

    # Import here so CLI --help is fast even if deps are missing
    from src.agent.training.ctde_trainer import HierarchicalTrainer, TrainingConfig
    from src.simulation.orchestrator import ScenarioConfig

    scenario = ScenarioConfig(
        episode_length_steps=args.episode_length,
        dynamics_substeps=2,       # keep light for speed
    )

    # Build GPU-aware batch size overrides
    gpu_bs_kwargs = {}
    if args.batch_size:
        gpu_bs_kwargs = {
            "gpu_batch_size_central": args.batch_size,
            "gpu_batch_size_mg": args.batch_size,
            "gpu_batch_size_sub": args.batch_size,
        }

    tc = TrainingConfig(
        total_timesteps=args.steps,
        eval_freq=args.eval_freq,
        n_eval_episodes=1,
        save_freq=max(args.steps, 10_000),
        log_dir=log_dir,
        tensorboard_log=str(project_root / "logs" / "tb"),
        train_sub_agents=True,
        steps_per_level=256,
        seed=args.seed,
        device=device,
        gpu_vram_limit_gb=args.vram_limit,
        **gpu_bs_kwargs,
    )

    logger.info(f"Scenario: {scenario}")
    logger.info(f"TrainingConfig: {tc}")

    trainer = HierarchicalTrainer(
        training_config=tc,
        scenario=scenario,
    )

    logger.info(f"Agents created: {list(trainer.agents.keys())}")
    logger.info("Starting training loop …")

    summary = trainer.train()

    logger.info("=" * 60)
    logger.info("Training Summary")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

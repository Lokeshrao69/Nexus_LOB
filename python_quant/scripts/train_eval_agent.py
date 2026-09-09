#!/usr/bin/env python
"""Train the PPO execution agent and compare it against the baselines.

Usage
-----
    # Default (gentle-walk env):
    PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
        [--iters 1500] [--episodes 16] [--epochs 4] [--eval-every 250] \
        [--eval-episodes 40] [--seed 2756] [--unit-seed 10015] \
        [--out policy_ppo.npz] [--table-episodes 100] [--table-seed 48879]

    # High-volatility regime:
    PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
        --highvol --vol-feature \
        [--iters 2500] [--out policy_ppo_highvol.npz]

    # Eval-only (load a saved policy, run strategy_table):
    PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
        --eval-only policy_ppo_highvol.npz \
        --highvol --vol-feature [--table-episodes 200]

The policy is written as a plain ``.npz`` of NumPy arrays and can be reloaded
later with ``PPOPolicy.load`` — no torch, no checkpoints.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python_quant"))

from nexus_quant import (  # noqa: E402
    GRPOConfig,
    OrderBookEnv,
    PPOConfig,
    format_table,
    strategy_table,
    train_grpo,
    train_ppo,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune + evaluate NEXUS-LOB agent")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0xACE)
    ap.add_argument("--unit-seed", type=int, default=0x2717)
    ap.add_argument("--out", type=str, default="python_quant/artifacts/policy_ppo.npz")
    ap.add_argument("--table-episodes", type=int, default=100)
    ap.add_argument("--table-seed", type=int, default=0xBEEF)
    # high-volatility regime flags
    ap.add_argument("--highvol", action="store_true",
                    help="Use high-volatility regime preset")
    ap.add_argument("--vol-feature", action="store_true",
                    help="Add 45th obs dim (regime indicator)")
    ap.add_argument("--regime-prob", type=float, default=None)
    ap.add_argument("--vol-decay", type=float, default=None)
    ap.add_argument("--gap-prob", type=float, default=None)
    ap.add_argument("--gap-min", type=int, default=None)
    ap.add_argument("--gap-max", type=int, default=None)
    # eval-only mode
    ap.add_argument("--eval-only", type=str, default=None, metavar="POLICY_NPZ",
                    help="Skip training; load policy and run strategy_table")
    ap.add_argument("--grpo", action="store_true", help="Train with GRPO instead of PPO")
    args = ap.parse_args()

    # --- build env factory ---
    env_kw: dict = {}
    if args.highvol:
        from nexus_quant import HIGHVOL_PRESETS
        env_kw.update(HIGHVOL_PRESETS["highvol"])
    if args.vol_feature:
        env_kw["vol_feature"] = True
    if args.regime_prob is not None:
        env_kw["regime_prob"] = args.regime_prob
    if args.vol_decay is not None:
        env_kw["vol_decay"] = args.vol_decay
    if args.gap_prob is not None:
        env_kw["gap_prob"] = args.gap_prob
    if args.gap_min is not None:
        env_kw["gap_min"] = args.gap_min
    if args.gap_max is not None:
        env_kw["gap_max"] = args.gap_max

    env_factory = lambda: OrderBookEnv(**env_kw)

    if args.eval_only:
        # --- eval-only mode: load policy, skip training ---
        from nexus_quant import PPOPolicy
        policy = PPOPolicy.load(args.eval_only)
        print(f"loaded policy from {args.eval_only} (obs_dim={policy.obs_dim})")
        rows = strategy_table(
            policy, agent_name="ppo", n_episodes=args.table_episodes,
            seed=args.table_seed, env_factory=env_factory,
        )
        print(f"\n=== execution comparison on {args.table_episodes} seeded episodes ===")
        print(format_table(rows))
        ppo = next(r for r in rows if r["name"] == "ppo")
        vwap = next(r for r in rows if r["name"] == "vwap")
        print(
            f"\nagent shortfall {ppo['shortfall_bps_mean']:.3f} bps vs "
            f"VWAP {vwap['shortfall_bps_mean']:.3f} bps "
            f"-> {ppo['vs_vwap_pct']:+6.1f}%"
        )
        return

    # --- train mode ---
    t0 = time.time()
    if args.grpo:
        gcfg = GRPOConfig(
            iterations=args.iters,
            episodes=args.episodes if args.episodes % 4 == 0 else max(4, args.episodes // 4 * 4),
            group=4,
            epochs=args.epochs,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            unit_seed=args.unit_seed,
        )
        policy, history = train_grpo(env_factory, gcfg)
        print(f"trained GRPO {args.iters} iterations in {time.time() - t0:.1f}s")
    else:
        cfg = PPOConfig(
            iterations=args.iters,
            episodes=args.episodes,
            epochs=args.epochs,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            unit_seed=args.unit_seed,
        )
        policy, history = train_ppo(env_factory, cfg)
        print(f"trained {args.iters} iterations in {time.time() - t0:.1f}s")

    print(f"{'it':>4} {'reward':>9} {'shortfall_bps':>13} {'pg_loss':>8} {'v_loss':>7} {'entropy':>8}")
    for h in history:
        it, rw, sf, pg, vl, en = h.as_row()
        print(f"{it:>4} {rw:>9.3f} {sf:>13.4f} {pg:>8.4f} {vl:>7.3f} {en:>8.4f}")

    policy.save(args.out)
    print("saved policy ->", args.out)

    rows = strategy_table(
        policy, agent_name="ppo", n_episodes=args.table_episodes,
        seed=args.table_seed, env_factory=env_factory,
    )
    print(f"\n=== execution comparison on {args.table_episodes} seeded episodes ===")
    print(format_table(rows))

    ppo = next(r for r in rows if r["name"] == "ppo")
    vwap = next(r for r in rows if r["name"] == "vwap")
    print(
        f"\nagent shortfall {ppo['shortfall_bps_mean']:.3f} bps vs "
        f"VWAP {vwap['shortfall_bps_mean']:.3f} bps "
        f"-> {ppo['vs_vwap_pct']:+6.1f}%"
    )


if __name__ == "__main__":
    main()
# nexus_quant.agents — RL execution agent (pure-NumPy PPO)

Self-contained actor–critic PPO for the `OrderBookEnv`. No torch / no sb3:
the whole stack — MLP forward/backward, Adam, GAE, clipped-surrogate update —
is hand-rolled in NumPy so runs are byte-reproducible and the policy is a
handful of arrays (`PPOPolicy.save` → `.npz`, `PPOPolicy.load` → back).

## What's here

| File | Role |
|---|---|
| `mlp.py`     | MLP (manual forward/backward, finite-diff tested) + Adam + grad clip |
| `ppo.py`     | `PPOPolicy`, `PPOConfig`, `collect_rollouts`/`compute_gae`/`ppo_update`, `train_ppo` |
| `evaluate.py`| `evaluate_policy`, `strategy_table`, `format_table` — agent vs baselines |
| `train_eval_agent.py` (in `python_quant/scripts/`) | one-shot train + eval CLI |

## How it learns

* **Policy** = Gaussian actor: μ(s) from a 44→64→32→1 tanh-squashed MLP over
  the 44-dim observation; exploration σ is one trainable scalar.
* **Critic** = 44→64→32→1 value MLP.
* **GAE(λ)** advantages (γ 0.99, λ 0.95); bootstrap V for truncated steps,
  0 for terminal.
* **Update** = clipped surrogate (ε 0.2) + entropy bonus, Adam, 4 epochs ×
  minibatch 128 every iteration (16 full episodes = ~640 steps).

Deterministic given a `PPOConfig` (seeded policy, rollout unit-seed stride,
seeded minibatch shuffling).

## The headline comparison — measured honestly

Run `train_eval_agent.py --iters 1500` (≈4.5 min), then the same 100 seeded
episodes are replayed for the agent and TWAP / VWAP / POV / Passive. Measured
2026-09-05 on the default `OrderBookEnv` (Q=2000, T=40, λ_inv=0.15, λ_time=0.35,
λ_adv=0.08):

```
strategy      reward shortfall_bps  vs_vwap%
ppo            -5.26         1.600     -3.0%
twap          -10.32         1.375    +11.5%
vwap           -7.59         1.554      0.0%
pov           -11.19         1.621     -4.4%
passive       -13.12         1.760    -13.3%
```

* `reward`   — the env's IS + inventory/time/adverse-move objective. **The PPO
  agent beats every baseline (~30% over VWAP)**: it learns to time fills to
  avoid adverse mid-moves while holding to a near-TWAP scheduling (completes
  at the horizon, not by front-loading).
* `shortfall_bps` — the pure "price concession vs arrival mid" metric that the
  resume claim targets. Here the agent lands ≈ VWAP (1.60 vs 1.55 bps, a −3%
  gap, within noise); TWAP's 1.375 is the best baseline by this measure.

### Why the shortfall target is not met on this sim (and how to chase it)

We swept the honest levers and report the negative result rather than padding:

* weaker λ_inv/λ_time → **worse** (agent holds, end-of-horizon fire-sale);
* `is_coef` on the IS term 4/6/10 → crossings drop to ~0 but shortfall stays
  ≈1.58–1.9 (the agent still caps to a bid-price fill via `px ≤ best_bid`);
* λ_sched schedule-tracking → the agent just crosses more to hold the pace;
* actor warm-start posting at the ask → gradient pulls it back to bid-fills.

Root cause: `OrderBookEnv` fills are structurally binary *per step* — a
negative offset is a guaranteed **bid** fill, only resting at the best ask
bids for the exogenous flow's lift, and the child size is fixed at TWAP pace
(`child_size = clamp(ceil(inv/t_left), 20, child_max)`). In a *gentle*
random-walk flow the price-aware policy's best reply is "cross on schedule":
completion certainty beats 2-tick price quality. TWAP, whose rule already
posts at the ask and pays exactly the flow's pace, is the price benchmark.

**Path to the ~14%-below-VWAP claim:** (1) a **high-volatility / gap-off**
flow regime where an adaptive agent legitimately earns its keep by pulling
away from the spread when the queue is stressed — the current sim has no
regime where aggressiveness pays; and/or (2) give the env a **schedule
constraint** (λ_sched) *plus* a positive fill-price edge (post at the touch)
so the agent competes on price inside a fixed participation pace. The knobs
in the env (`is_coef`, `lambda_sched`) and the harness are all in place to run
that experiment.

## The high-volatility headline (historical exploratory run; re-verified & retired)

> ⚠️ **AUDITED & RETIRED (2026-09-13):**
> The +50.4% shortfall improvement measured below against a 2-line VWAP heuristic was audited
> under the **Part 2 Phase 3 fair RL protocol** (`plan_2.md` §6, `docs/results/rl_fairness.md`).
> When evaluated with symmetric information (baselines observing `regime_indicator`),
> defensible adaptive baselines (`adaptive_pov`), transaction fees, and queue priority:
> - PPO exhibits **no statistically significant edge** over `adaptive_pov` on high-vol, trending, or null regimes (|Δ| ≲ 0.3 bps).
> - PPO is **significantly worse** on calm/low-vol hold-outs (−0.05 … −0.16 bps).
> - PPO demonstrates a genuine edge only during extreme liquidity shocks (+0.4 … +1.4 bps).
> The naive +50.4% headline was an artifact of asymmetric information and is **officially retired**.
> Full report: `docs/RESEARCH.md` §7 and `docs/results/rl_fairness.md`.

### Historical Exploratory Result (2026-09-07)

The gentle-walk sim above is the honest baseline: there, PPO wins on reward
but lands ≈VWAP on shortfall. The path to the ~14%-below-VWAP headline was a
**high-volatility / gap-off flow regime** (Markov-switching + gap events) added
to `OrderBookEnv` via new constructor params (defaults preserve the calm
behavior). See `HIGHVOL_PLAN.md` at the repo root for the full design.

**Measured 2026-09-07** after `--highvol --vol-feature --iters 2000`, on 100
seeded episodes (seed `0xBEEF`):

```
strategy      reward shortfall_bps  vs_vwap%
ppo            -5.04         1.401    +50.4%
twap           -8.60         2.659     +5.9%
vwap           -7.40         2.827      0.0%
pov            -9.32         2.519    +10.9%
passive       -12.81         2.679     +5.3%
```

The regime roughly triples VWAP's slippage (1.64 → 2.83 bps) by punishing
fixed-schedule execution around mid-price gaps. The PPO agent learned to detect
volatility and execute more before/around the gaps, yielding an exploratory
~1.4 bps improvement over a basic heuristic VWAP schedule. Policy saved at
`python_quant/artifacts/policy_ppo_highvol.npz`. See above for the fair re-verification.

How to reproduce the headline run:

```bash
# train on the high-vol regime (obs_dim=45 with regime indicator)
PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
    --highvol --vol-feature --iters 2000 --eval-every 400 --eval-episodes 40 \
    --out python_quant/artifacts/policy_ppo_highvol.npz \
    --table-episodes 100 --table-seed 0xBEEF

# sweep env params without retraining (eval-only)
PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
    --eval-only python_quant/artifacts/policy_ppo_highvol.npz \
    --highvol --vol-feature --table-episodes 200 --table-seed 12345

# existing calm-env run + agent tests
PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
    --iters 1500 --eval-every 300 --eval-episodes 40 --table-episodes 100
python -m pytest python_quant/tests/test_ppo_agent.py -v   # 12 tests
python -m pytest python_quant/tests/test_highvol_env.py -v # 11 tests
```

Tests cover: MLP backward vs finite differences; Adam convergence; GAE on a
hand-rolled trajectory; policy bounds; PPO update finiteness; `train_ppo`
determinism; save/load round-trip; the seeded evaluation table; and the
highvol regime (transitions, gaps, obs dim, inventory conservation).
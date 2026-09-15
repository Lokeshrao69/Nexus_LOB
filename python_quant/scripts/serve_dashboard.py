#!/usr/bin/env python
"""Serve the Nexus-LOB combined desk: console styling + live L2 overlay.

    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py
    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py --ring /tmp/nexus_slots.bin
    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py --shm nex_aapl

Default ``--synthetic`` drives a seeded mid random-walk (a view dict matching the
frozen ``BOOK_STATE_DTYPE`` shape) so the page works without C++ shm and visibly
moves. Latency samples are the real measured wall-time of each snapshot render.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python_quant"))

from nexus_quant.book_state import DEPTH  # noqa: E402
from nexus_quant.dashboard import (  # noqa: E402
    SnapshotHub,
    latest_from_file_ring,
    read_shm_ring_latest,
    serve,
)
from nexus_quant.envs.order_book_env import OrderBookEnv  # noqa: E402
from nexus_quant.baselines import policy_action  # noqa: E402
from nexus_quant.risk import compute_var_cvar  # noqa: E402

_SEED = 0xC0FFEE


def _synthetic_view(rng: np.random.Generator, mid: int, seq: int,
                    volume: int, last: tuple[int, int, int]) -> dict:
    """A view-shaped dict (same shape as StubOrderBook.view()) around a mid."""
    half = int(rng.choice([1, 1, 1, 2, 2, 3]))  # spread mostly 2, sometimes wider
    bp = np.zeros(DEPTH, dtype=np.int64)
    bs = np.zeros(DEPTH, dtype=np.uint64)
    bc = np.zeros(DEPTH, dtype=np.uint32)
    ap = np.zeros(DEPTH, dtype=np.int64)
    az = np.zeros(DEPTH, dtype=np.uint64)
    ac = np.zeros(DEPTH, dtype=np.uint32)
    for i in range(DEPTH):
        bp[i] = mid - half - i
        bs[i] = int(60 + rng.integers(0, 260) + (DEPTH - i) * 4)
        bc[i] = int(rng.integers(1, 9))
        ap[i] = mid + half + i
        az[i] = int(60 + rng.integers(0, 260) + (DEPTH - i) * 4)
        ac[i] = int(rng.integers(1, 9))
    px, sz, side = last
    return {
        "seq": seq, "ts_ns": seq, "cum_volume": volume,
        "last_trade_px": px, "last_trade_sz": sz, "last_trade_side": side,
        "version": seq,
        "bid_px": bp, "bid_sz": bs, "bid_ct": bc,
        "ask_px": ap, "ask_sz": az, "ask_ct": ac,
    }


class LiveEpisode:
    """Drive a real, seeded ``OrderBookEnv`` execution episode as the live source.

    The synthetic book walk gives the desk its L2 ladder + sparklines (a frozen
    ``BookStateView`` shaped feed with no C++ build needed). The execution
    episode does NOT replace that feed — instead it layers the *real* per-step
    execution trajectory (inventory path, child fills, arrival mid) onto the
    page through ``hub.exec_episode``, so the "execution timeline" chart is an
    actual running episode, not a static sketch.
    """

    def __init__(self, *, inventory: int = 500, horizon: int = 20,
                 strategy: str = "twap", queue_model: str = "fifo",
                 drift_ticks: float = 2.0) -> None:
        self.inventory = int(inventory)
        self.horizon = int(horizon)
        self.strategy = strategy
        self.queue_model = queue_model
        self.drift_ticks = float(drift_ticks)
        self.env = OrderBookEnv(
            inventory=self.inventory, horizon=self.horizon,
            seed=_SEED, queue_model=queue_model,
            drift_ticks=self.drift_ticks,
        )
        self.episode = 0
        self._roll()

    def _roll(self) -> None:
        self.episode += 1
        self.env.reset(seed=_SEED + self.episode * 7919)
        self.arrival_mid = self.env.arrival_mid
        self.steps: list[dict] = []
        self.done = False
        self.shortfall_bps = None
        self.final = {}

    def step(self) -> None:
        """Advance one env step; record the trajectory; roll a new episode when done."""
        if self.done:
            self._roll()
        a = policy_action(self.strategy, self.env)
        _, reward, terminated, truncated, info = self.env.step(a)
        step_t = max(0, int(self.env.t) - 1)     # t was advanced by step()
        self.steps.append({
            "t": step_t,
            "mid": float(info["mid"]),
            "inv": int(info["inventory"]),
            "filled": int(info["filled"]),
            "reward": float(reward),
        })
        if terminated or truncated:
            self.done = True
            self.shortfall_bps = round(float(info["shortfall_bps"]), 4)
            self.final = {
                "shortfall_bps": self.shortfall_bps,
                "leftover": int(info["inventory"]),
                "completion": round(1.0 - info["inventory"] / max(1, self.inventory), 4),
                "t_used": int(self.env.t),
            }

    def as_json(self) -> dict:
        return {
            "episode": self.episode,
            "strategy": self.strategy,
            "queue_model": self.queue_model,
            "drift_ticks": self.drift_ticks,
            "inventory0": self.inventory,
            "horizon": self.horizon,
            "arrival_mid": int(self.arrival_mid),
            "steps": self.steps,
            "done": self.done,
            "shortfall_bps": self.shortfall_bps,
            "final": self.final,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--ring", type=str, default="")
    ap.add_argument("--shm", type=str, default="")
    ap.add_argument("--live-exec", action="store_true", default=False,
                    help="layer a real seeded OrderBookEnv execution episode "
                         "(TWAP/FIFO) onto the desk as the execution timeline")
    ap.add_argument("--exec-drift", type=float, default=2.0,
                    help="trend target drift (ticks/step) for the live execution "
                         "mid path; 0 = flat calm market")
    args = ap.parse_args()

    hub = SnapshotHub()
    rng = np.random.default_rng(_SEED)
    state = {"mid": 15_000, "seq": 0, "vol": 0, "last": (0, 0, 2), "n": 0}
    exec_ep = (LiveEpisode(drift_ticks=args.exec_drift) if args.live_exec else None)

    def poll() -> None:
        t0 = time.perf_counter_ns()
        if args.ring:
            v = latest_from_file_ring(Path(args.ring))
            if v:
                hub.push(v, source="file-ring")
                return
        if args.shm:
            v = read_shm_ring_latest(args.shm)
            if v:
                hub.push(v, source="shm-ring")
                return
        # ----- synthetic walk -------------------------------------------------
        st = state
        st["n"] += 1
        st["seq"] += 1
        mid_new = int(np.clip(st["mid"] + int(rng.integers(-4, 5)), 1_000, 99_000))
        st["mid"] = mid_new
        # a ~40% chance a trade prints each tick
        if rng.random() < 0.4:
            side = int(rng.integers(0, 2))                     # 0=Bid aggressor, 1=Ask
            px = mid_new + (0 if rng.random() < 0.6 else (-1 if side == 0 else 1))
            sz = int(rng.integers(2, 60))
            st["vol"] += sz
            st["last"] = (px, sz, side)
        v = _synthetic_view(rng, mid_new, st["seq"], st["vol"], st["last"])
        hub.push(v, source="synthetic", feed_latency_ns=time.perf_counter_ns() - t0)
        # Advance the live execution episode one step (real env: FIFO queue,
        # TWAP policy). When it finishes, it rolls a fresh seeded episode, so
        # the execution timeline on the desk never goes stale.
        if exec_ep is not None:
            exec_ep.step()
            hub.exec_episode = exec_ep.as_json()
        if st["n"] % 8 == 0:
            r = compute_var_cvar(n_paths=256, steps=32, prefer_engine=False)
            hub.risk = {"var": r.var, "cvar": r.cvar}

    poll()
    httpd = serve(hub, args.host, args.port, poll)
    print(f"Nexus-LOB desk http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
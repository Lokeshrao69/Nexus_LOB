#!/usr/bin/env python3
"""E6b — clock-time passive-fill markouts on a REAL NASDAQ ITCH tape.

Companion to ``scripts/run_research.py``: same tape, same replay, same passive-fill
population (the first execution of each tracked resting order, regular session
only), but markouts measured on **wall-clock** horizons from ITCH nanosecond
timestamps instead of the event clock — and from *two* reference mids, the one
just before the fill message and the one just after it is fully applied.

The point is to find out how much of E6's "90–97 % of passive fills are adversely
selected" is mechanical. See :mod:`nexus_quant.research.clock_markouts` for the
conventions and for why the before/after gap is a *lower bound* on the mechanical
component.

Reuses ``run_research.replay_symbol`` (so E1–E6 and E6b always see an identical
book) and skips the expensive E1–E5 studies.

Run (from repo root)::

    python python_quant/scripts/run_clock_markouts.py --day 12302019 --symbols AAPL,QQQ
    python python_quant/scripts/run_clock_markouts.py --symbols AAPL --max-events 200000

A full ITCH day holds millions of events in memory, so on a tight machine run one
symbol per process and then stitch the report together from the saved JSON::

    python python_quant/scripts/run_clock_markouts.py --symbols AAPL
    python python_quant/scripts/run_clock_markouts.py --symbols QQQ
    python python_quant/scripts/run_clock_markouts.py --symbols AAPL,QQQ --render-only

Output: ``docs/results/clock_markouts_<day>_<SYMBOL>.json`` +
``docs/results/clock_markouts_<day>.md``. Nothing under ``real_tape_*`` is touched.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 safe

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python_quant"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nexus_quant.research.clock_markouts import (
    NS_PER_MS,
    render_clock_markouts_md,
)
from run_research import (
    clock_markout_study,
    replay_symbol,
    verify_manifest_provenance,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--day", default="12302019")
    ap.add_argument("--symbols", default="AAPL,QQQ")
    ap.add_argument("--data", type=Path, default=Path("data/itch"))
    ap.add_argument(
        "--horizons-ms", type=int, nargs="+", default=[100, 1_000, 10_000, 60_000],
        help="clock markout horizons in milliseconds",
    )
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--min-group", type=int, default=30)
    ap.add_argument("--max-events", type=int, default=None, help="cap regular-session rows (quick look)")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "docs" / "results")
    ap.add_argument(
        "--render-only", action="store_true",
        help="skip the replay; rebuild the combined .md from the per-symbol .json files",
    )
    args = ap.parse_args(argv)

    horizons_ns = tuple(int(ms) * NS_PER_MS for ms in args.horizons_ms)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    per_symbol: dict[str, dict] = {}
    t_all = time.time()

    if args.render_only:
        for sym in symbols:
            jf = args.out_dir / f"clock_markouts_{args.day}_{sym}.json"
            if not jf.is_file():
                print(f"missing {jf} — run this symbol without --render-only first")
                continue
            per_symbol[sym] = json.loads(jf.read_text(encoding="utf-8"))
            print(f"loaded {jf.name}")
        if not per_symbol:
            print("nothing to render")
            return 1
        md = args.out_dir / f"clock_markouts_{args.day}.md"
        md.write_text(render_clock_markouts_md(args.day, per_symbol), encoding="utf-8")
        print(f"wrote {md} from {len(per_symbol)} symbol file(s)")
        return 0

    for sym in symbols:
        path = args.data / args.day / f"{sym}.itch"
        if not path.exists():
            print(f"missing {path} — run scripts/fetch_itch.py --day {args.day} --symbols {sym}")
            continue
        verify_manifest_provenance(path, symbol=sym)
        print(f"== {sym} ==")
        rep = replay_symbol(path, max_events=args.max_events, verify_manifest=False)
        # E6b needs only the row series and the harvested fills. The parsed event
        # list, the order tracker and the E1-E4 feature arrays are the bulk of the
        # footprint on a full ITCH day, and nothing below reads them.
        for key in ("events", "tracker", "features"):
            rep.pop(key, None)
        gc.collect()
        t0 = time.time()
        res = clock_markout_study(
            rep, horizons_ns=horizons_ns, n_boot=args.n_boot, min_group=args.min_group
        )
        print(f"  E6b done in {time.time() - t0:.1f}s: {res.get('n_fills', 0):,} fills "
              f"in {res.get('n_events', 0):,} aggressor events "
              f"({res.get('fills_per_event', float('nan')):.2f} fills/event)")
        for hl, blk in res.get("horizons", {}).items():
            d = blk["events"]["drops"]
            bef = blk["events"]["before"]["overall"]
            aft = blk["events"]["after"]["overall"]
            rl = aft["realized_spread_px0001"]
            mb, ma = bef["markout_px0001"], aft["markout_px0001"]
            print(
                f"    H={hl:<6} n={aft['n']:<7,} dropped(close/tape/pre-open/pre-1sided)="
                f"{d['past_close']}/{d['past_tape']}/{d['pre_past_open']}/{d['pre_bad_mid']}"
            )
            print(
                f"      markout $0.0001  before {mb['mean']:+8.2f} | after {ma['mean']:+8.2f} "
                f"[{ma['mean_ci95']['lo']:+.2f}, {ma['mean_ci95']['hi']:+.2f}] "
                f"median {ma['median']:+8.2f} trimmed {ma['trimmed_mean']:+8.2f}"
            )
            print(
                f"      realized spread  mean {rl['mean']:+8.2f} "
                f"[{rl['mean_ci95']['lo']:+.2f}, {rl['mean_ci95']['hi']:+.2f}] "
                f"median {rl['median']:+8.2f} | {aft['mean_realized_spread_half_spreads']:+.3f} HS "
                f"| adverse {aft['share_adverse']:.3f} zero {aft['share_zero']:.3f} "
                f"cond {aft['share_adverse_conditional']:.3f}"
            )
        lad = res.get("attribution") or []
        if lad:
            print("    attribution ladder  P(adverse | moved):")
            for row in lad:
                print(f"      {row['step']}. {row['unit']:<20} {row['clock']:<10} "
                      f"before {row['before']['share_adverse_conditional']:.3f} "
                      f"(n={row['before']['n']:,})  |  "
                      f"after {row['after']['share_adverse_conditional']:.3f} "
                      f"(n={row['after']['n']:,})")
        for ref, chk in (res.get("attribution_check") or {}).items():
            if not chk.get("defined", True):
                print(f"    walk [{ref}]: undefined (a cell has no non-zero moves)")
                continue
            steps = " + ".join(
                f"{st['changed']} {st['delta']:+.3f}" for st in chk["steps"]
            )
            print(f"    walk [{ref}]: {chk['first_value']:.3f} -> "
                  f"{chk['last_value']:.3f} = {steps} = {chk['steps_sum']:+.3f} "
                  f"vs total {chk['total_change']:+.3f} "
                  f"({'closes' if chk['closes'] else 'DOES NOT CLOSE'})")
        for hl, blk in res.get("horizons", {}).items():
            st = blk["events"]["after"]["overall"]
            nr = st.get("net_of_rebate") or {}
            if nr:
                bits = "  ".join(
                    f"+{k}: {v['mean']:+7.2f} ({v['sign']['share_pos']:.3f} profitable)"
                    for k, v in sorted(nr.items(), key=lambda kv: int(kv[0]))
                )
                print(f"    rebate H={hl:<6} (ASSUMED) {bits}")
        for hl, blk in res.get("horizons", {}).items():
            pl = blk["block"]
            print(f"    blocks H={hl:<6} NW lag {blk['nw_lag']:<5} "
                  f"block {pl['block_events']:<5} obs  effective blocks "
                  f"{pl['n_blocks_effective']}")
        per_symbol[sym] = res
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"clock_markouts_{args.day}_{sym}.json").write_text(
            json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8"
        )
        del rep
        gc.collect()
    if per_symbol:
        md = args.out_dir / f"clock_markouts_{args.day}.md"
        md.write_text(render_clock_markouts_md(args.day, per_symbol), encoding="utf-8")
        print(f"\nwrote {md} ({time.time() - t_all:.0f}s total)")
    else:
        print("no symbols processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

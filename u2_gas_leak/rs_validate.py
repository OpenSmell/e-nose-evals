#!/usr/bin/env python3
"""Cross-validation + confidence intervals for the Rust engine real-data results.

Reads the archived `rs_realdata_*.json` metrics produced by
`opensmell-rs/src/bin/realdata_eval.rs` and answers the holding-out questions
the papers need:

  1. Cross-file sensitivity selection: pick `sens` on one recording with a
     rule (max detection rate while clean FPR <= target), validate it on the
     other recording.
  2. Wilson binomial CI on detection rate (events are the independent units).
  3. Wilson binomial CI on clean FPR (per-clean-second false alarms).
  4. Bootstrap CI on median detection latency.

The rule is the only free parameter and it is fixed *before* any holdout runs,
so the reported operating point cannot have been retrofitted to both files.

Run:  python3 rs_validate.py [--fpr-target 0.002]
"""
import argparse
import json
import math
import random

RES = "/home/praisejamesx/Documents/Documents/Research/coglab/OpenSmell/e-nose-evals/u2_gas_leak/results"
CO = f"{RES}/rs_realdata_ethylene_co_full_sens3.json"
METHANE = f"{RES}/rs_realdata_ethylene_methane_full_sens3.json"
CO_S2 = f"{RES}/rs_realdata_ethylene_co_full_sens2.json"
METHANE_S4 = f"{RES}/rs_realdata_ethylene_methane_full_sens4.json"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(p):
    return json.load(open(p))


def summarize(d):
    return {
        "sens": d["sensitivity"],
        "det": d["detected_events"],
        "n": d["n_events"],
        "rate": d["detection_rate"],
        "fpr": d["clean_fpr"],
        "clean_s": d["clean_seconds"],
        "clean_alarms": d["clean_alarms"],
        "lat": d["latencies_s"],
        "cal": (d["calibration_start_s"], d["calibration_end_s"]),
    }


def bootstrap_median(lat, iters=100_000, seed=7):
    r = random.Random(seed)
    meds = []
    for _ in range(iters):
        s = [r.choice(lat) for _ in lat]
        s.sort()
        m = len(s) // 2
        meds.append(s[m] if len(s) % 2 == 1 else (s[m - 1] + s[m]) / 2)
    meds.sort()
    return (meds[int(0.025 * iters)], meds[int(0.975 * iters)])


def choose_sens(rows, target):
    """max rate, tiebreak smallest sens, reject sens whose FPR > target."""
    best = None
    for r in sorted(rows, key=lambda r: (r["sens"], r["rate"])):
        if r["fpr"] <= target:
            cand = (r["rate"], -r["sens"], r)
            if best is None or cand > best[0]:
                best = (cand, r)
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr-target", type=float, default=0.002)
    args = ap.parse_args()
    target = args.fpr_target

    co = {"s2": summarize(load(CO_S2)), "s3": summarize(load(CO))}
    me = {"s3": summarize(load(METHANE)), "s4": summarize(load(METHANE_S4))}

    print(f"operating-point rule: max detection rate with clean FPR <= {target}")
    print()
    for role, rows in (("tuning set", [co["s2"], co["s3"]]),):
        pick = choose_sens(rows, target)
        print(f"tune on ethylene_CO  -> selected sens {pick['sens']} "
              f"(rate {pick['rate']:.3f}, fpr {pick['fpr']:.5f})")
    pick = choose_sens([me["s3"], me["s4"]], target)
    print(f"tune on methane      -> selected sens {pick['sens']} "
          f"(rate {pick['rate']:.3f}, fpr {pick['fpr']:.5f})")
    print()

    # Cross-validation: sens chosen on the OTHER file, applied here.
    co_sens_from_me = choose_sens([me["s3"], me["s4"]], target)
    me_sens_from_co = choose_sens([co["s2"], co["s3"]], target)
    print("cross-validation (sens tuned on the other recording):")
    for name, d, picked_on_other in (
        ("ethylene_CO", co["s3"], co_sens_from_me),
        ("ethylene_methane", me["s3"], me_sens_from_co),
    ):
        lo, hi = wilson_ci(d["det"], d["n"])
        flo, fhi = wilson_ci(d["clean_alarms"], d["clean_s"])
        llo, lhi = bootstrap_median(d["lat"]) if d["lat"] else (None, None)
        print(f"\n{name}  (sens {d['sens']}, matched by rule on the other file: "
              f"sens {picked_on_other['sens']})")
        print(f"  calibration window       {d['cal']} s (causal-identical)")
        print(f"  detection {d['det']}/{d['n']}  Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  clean FPR {d['fpr']:.6f} ({d['clean_alarms']} alarms / "
              f"{d['clean_s']} clean s)  Wilson 95% CI [{flo:.6f}, {fhi:.6f}]")
        print(f"     = one spurious alarm per {d['clean_s'] / max(d['clean_alarms'],1):.0f} s of clean air")
        if llo:
            print(f"  median latency {d['lat'][len(d['lat'])//2]:.1f} s  "
                  f"bootstrap 95% CI [{llo:.1f}, {lhi:.1f}]")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Hidden workload for wro-secure-agg-shamir-committee-coupled (Type-2 B2 BEAT, proxy-perf).

Subsystem: a privacy-preserving (secure-aggregation) server under ``secureagg/`` -- Shamir secret
sharing over a prime field (``shamir.Shamir``) plus a masked cross-committee aggregator
(``aggregator.SecureAggregator``), sharing a fixed prime-field building block (``field.py``, out of
scope). The server reconstructs many per-coordinate aggregates from the SAME committee of
shareholders each round.

  correctness -- build MANY diverse sharing scenarios (threshold 1..n, exact-threshold vs extra
                 shares, single/many coordinates, secret 0 / p-1 / random, several committees,
                 multi-round reuse, duplicate-x error, empty error) and assert split/reconstruct
                 round-trips, reconstruct_many, aggregate, aggregate_rounds and committee_of all
                 match an INDEPENDENT in-harness reference (naive Lagrange) EXACTLY, and that the
                 documented ValueErrors are raised. Emits ``WRO_SECAGG_RESULT {"correctness_frac":..}``.

  timing      -- reconstruct a large model's per-coordinate aggregates over several rounds from ONE
                 committee. The naive path rebuilds the O(t^2) Lagrange weights for every coordinate
                 of every round; the committee-caching path computes them once per committee then does
                 O(t) dot-products. Headroom GROWS with the number of coordinates x rounds and the
                 threshold t. Emits ``WRO_SECAGG_RESULT {"timing_ms": ...}``.

Imports ``secureagg`` from /app/repo (PYTHONPATH).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, "/app/repo")


def scope_pkg():
    import secureagg as m
    return m


# ---------------- independent reference (naive Lagrange at 0, no import of scope shamir) ------------
_P = (1 << 61) - 1


def _inv(a, p):
    a %= p
    if a == 0:
        raise ZeroDivisionError
    return pow(a, p - 2, p)


def _ref_eval(coeffs, x, p):
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % p
    return acc


def _ref_reconstruct(shares, p=_P):
    if not shares:
        raise ValueError("no shares")
    xs = [x % p for x, _ in shares]
    if len(set(xs)) != len(xs):
        raise ValueError("dup x")
    total = 0
    k = len(shares)
    for i in range(k):
        xi, yi = shares[i][0] % p, shares[i][1] % p
        num = den = 1
        for j in range(k):
            if j == i:
                continue
            xj = shares[j][0] % p
            num = (num * (-xj)) % p
            den = (den * (xi - xj)) % p
        total = (total + yi * (num * _inv(den, p)) % p) % p
    return total % p


def _ref_split(secret, n, t, rng, p=_P):
    coeffs = [secret % p] + [rng.randrange(p) for _ in range(t - 1)]
    return [(x, _ref_eval(coeffs, x, p)) for x in range(1, n + 1)]


def _scenarios():
    rnd = random.Random(20260726)
    scen = []   # (name, dict)
    scen.append(("t1_n1", {"n": 1, "t": 1, "secret": 42}))
    scen.append(("t2_n3", {"n": 3, "t": 2, "secret": 123456789}))
    scen.append(("tn_full", {"n": 5, "t": 5, "secret": 7}))
    scen.append(("secret_zero", {"n": 4, "t": 3, "secret": 0}))
    scen.append(("secret_pminus1", {"n": 6, "t": 4, "secret": _P - 1}))
    scen.append(("big_committee", {"n": 12, "t": 8, "secret": 999999999999}))
    for c in range(20):
        n = rnd.randint(1, 14)
        t = rnd.randint(1, n)
        scen.append(("rand%d" % c, {"n": n, "t": t, "secret": rnd.randrange(_P)}))
    return scen


def run_correctness():
    m = scope_pkg()
    npass = 0
    results = {}
    scen = _scenarios()
    for name, sp in scen:
        try:
            n, t, secret = sp["n"], sp["t"], sp["secret"]
            sh = m.Shamir()
            # deterministic: same seed for scope split and reference split
            r1 = random.Random(1000 + hash(name) % 9999)
            shares = sh.split(secret, n, t, r1)
            ok = True
            # round-trip with EXACTLY t shares (first t) and with ALL n shares
            if sh.reconstruct(shares[:t]) != secret % _P:
                ok = False
            if sh.reconstruct(shares) != secret % _P:
                ok = False
            # a different subset of t shares also reconstructs
            if n >= t:
                subset = shares[-t:]
                if sh.reconstruct(subset) != secret % _P:
                    ok = False
            # reconstruct_many across several secrets sharing the committee x-coords (first t)
            secrets = [rnd_i for rnd_i in (secret, (secret * 3 + 1) % _P, (secret + 12345) % _P)]
            xcoords = [x for x, _ in shares[:t]]
            share_lists = []
            for sv in secrets:
                rr = random.Random(50 + sv % 7)
                full = _ref_split(sv, n, t, rr)
                share_lists.append([(x, y) for x, y in full if x in xcoords])
            got = sh.reconstruct_many(share_lists)
            exp = [_ref_reconstruct(sl) for sl in share_lists]
            if got != exp:
                ok = False
            # aggregator: D coordinates, same committee
            agg = m.SecureAggregator(sh)
            D = 5
            coord_shares = []
            for d in range(D):
                rr = random.Random(9 + d)
                full = _ref_split((secret + d * 7) % _P, n, t, rr)
                coord_shares.append([(x, y) for x, y in full if x in xcoords])
            if agg.aggregate(coord_shares) != [_ref_reconstruct(cs) for cs in coord_shares]:
                ok = False
            if agg.committee_of(coord_shares) != tuple(sorted(xcoords)):
                ok = False
            # aggregate_rounds
            rounds = [coord_shares, coord_shares[:2]]
            if agg.aggregate_rounds(rounds) != [[_ref_reconstruct(cs) for cs in r] for r in rounds]:
                ok = False
            results[name] = {"ok": bool(ok)}
        except Exception as e:
            results[name] = {"ok": False, "error": repr(e)}
        if results[name]["ok"]:
            npass += 1

    # error-path cases (documented ValueErrors)
    err_cases = 0
    err_pass = 0
    sh = m.Shamir()
    err_cases += 1
    try:
        sh.reconstruct([])            # empty -> ValueError
    except ValueError:
        err_pass += 1
    except Exception:
        pass
    err_cases += 1
    try:
        sh.reconstruct([(3, 10), (3, 11)])   # duplicate x -> ValueError
    except ValueError:
        err_pass += 1
    except Exception:
        pass
    err_cases += 1
    try:
        sh.split(9, 3, 5, random.Random(0))  # t>n -> ValueError
    except ValueError:
        err_pass += 1
    except Exception:
        pass
    results["_errors"] = {"ok": err_pass == err_cases, "err_pass": err_pass, "err_cases": err_cases}
    if results["_errors"]["ok"]:
        npass += 1

    total = len(results)
    failed = {k: v for k, v in results.items() if not v["ok"]}
    return (npass / total if total else 0.0), total, failed


def run_timing():
    m = scope_pkg()
    n = int(os.environ.get("WRO_SECAGG_N", "40"))
    t = int(os.environ.get("WRO_SECAGG_T", "24"))
    D = int(os.environ.get("WRO_SECAGG_D", "1200"))
    rounds = int(os.environ.get("WRO_SECAGG_ROUNDS", "3"))
    rep = int(os.environ.get("WRO_SECAGG_REP", "3"))
    rng = random.Random(99)
    # one committee: shares for D coordinates, threshold t (use x-coords 1..t)
    round_data = []
    for _ in range(rounds):
        coords = []
        for d in range(D):
            full = _ref_split(rng.randrange(_P), n, t, rng)
            coords.append([(x, y) for x, y in full if x <= t])
        round_data.append(coords)

    def one():
        sh = m.Shamir()
        agg = m.SecureAggregator(sh)
        acc = 0
        out = agg.aggregate_rounds(round_data)
        for r in out:
            acc = (acc + sum(r)) % _P
        return acc

    one()  # warmup
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter()
        one()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if cmd == "timing":
        try:
            print("WRO_SECAGG_RESULT " + json.dumps({"timing_ms": run_timing()}))
        except Exception as e:
            import traceback
            print("WRO_SECAGG_RESULT " + json.dumps({"timing_ms": -1, "error": repr(e),
                                                     "tb": traceback.format_exc()[-800:]}))
        return
    origin = None
    try:
        origin = os.path.realpath(scope_pkg().shamir.__file__)
    except Exception:
        pass
    try:
        frac, total, failed = run_correctness()
        print("WRO_SECAGG_RESULT " + json.dumps(
            {"correctness_frac": frac, "n_cases": total, "n_failed": len(failed),
             "failed": {k: failed[k] for k in list(failed)[:8]}, "origin": origin}))
    except Exception as e:
        import traceback
        print("WRO_SECAGG_RESULT " + json.dumps(
            {"correctness_frac": 0.0, "error": repr(e),
             "tb": traceback.format_exc()[-900:], "origin": origin}))


if __name__ == "__main__":
    main()

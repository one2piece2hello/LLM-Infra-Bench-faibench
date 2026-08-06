#!/usr/bin/env python3
"""/opt/loop/private/dev_eval.py — per-round DEV proxy for e2e-a8-peft-adapter-byte-golf.

Root-owned 0600, called ONLY by /opt/loop/score_engine.sh (0700). The solver's uid cannot read it.

WHAT IT DOES (a faithful, CHEAP, leak-free mirror of the graded path, on the PUBLIC dev corpus):
  1. Re-measures the ADAPTER-BYTE budget the disclosed way: stat(adapter.bin) + stat(adapter_entry.py)
     <= max_total_bytes, with the per-file caps — the SAME dual-measured byte gate the grade enforces
     (the disclosed hard rule; not a self-report).
  2. Loads the FROZEN base model from /app/base_model itself (never a solver copy), measures base CE
     on the PUBLIC dev corpus (/data/datasets/a8_syscode/val.txt), then loads the candidate's
     byte-capped adapter via the SAME build_adapted_model(base, artifact, device) entry the grade
     calls and measures candidate CE on the same public corpus.
  3. Runs a first-principles self-check (param floor, plausible CE band, logits vary across positions,
     OOD real << arbitrary) — the SAME anti-spoof invariants the grade uses — and reports the RAW
     adaptation gain (base_ce - candidate_ce) as the best-of-k ranking signal (higher = a better
     adapter). NO strong-baseline denominator, NO calibrated anchor.

WHAT IT NEVER TOUCHES / BAKES: /tests, the held-out corpus (holdout_corpus.txt), the calibrated
strong_baseline_ce / ref_speedup, or the graded manifest. The GRADED reward normalises the gain
against a HIDDEN strong_baseline_ce and feeds it through a calibrated log curve; this dev signal is
the RAW gain on the PUBLIC dev corpus, never normalised. Both rise as the adapter improves, so
best-of-k ranking is faithful; the absolute number differs. Authoritative score = the runner grade.

🔴 FIDELITY NOTE: the grade also runs a whole-filesystem STASH SWEEP + quarantine to make the byte
budget information-theoretically airtight (a candidate cannot stash weights outside the declared
files). This dev proxy does NOT run that sweep (it is heavy and its baked inventory is graded-surface)
— it enforces only the DECLARED-file byte budget. So a candidate that stashes bytes in a side file can
look within-budget on the dev proxy and still fail the grade's sweep. best-of-k still prefers
higher-gain within-declared-budget adapters; the authoritative byte accounting is the grade's.

REDUCED WORKLOAD (the GPU constraint): eval-only (NO training in the candidate path — matching the
grade), CE over a SMALL public-corpus token budget (dev_eval_tokens), median of a couple of shifted
passes. A per-round pass is two forward-only CE evals of a 0.5B model — well under a minute on H20.

OUTPUT: /logs/loop/dev/{verifier_state.json, reward.json}; on infra failure, harness_error.txt.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys
from pathlib import Path

LOOP_PRIVATE = Path("/opt/loop/private")
MANIFEST = LOOP_PRIVATE / "manifest.json"
DEV_OUT = Path("/logs/loop/dev")
DEV_OUT.mkdir(parents=True, exist_ok=True)

INV_MISSING = "adapter_artifact_or_entry_missing"
INV_BUDGET = "adapter_byte_budget_exceeded"
INV_BASE_MISSING = "frozen_base_model_missing"
INV_BUILD = "build_adapted_model_failed"
INV_PARAM_FLOOR = "param_below_floor"
INV_DEGENERATE = "logits_constant_across_positions"
INV_CE_BAND = "ce_implausible"
INV_OOD = "model_ignores_input_content"
INV_HARNESS = "harness_error"

ARTIFACT_NAME = "adapter.bin"
ENTRY_NAME = "adapter_entry.py"


def _cfg() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _write_state(correctness_ok, failing_invariant, dev_gain, base_ce, cand_ce, extra=None):
    reasons = [] if correctness_ok else ([failing_invariant] if failing_invariant else ["unknown"])
    state = {"correctness_ok": bool(correctness_ok), "hard_fail_reasons": reasons,
             "failing_invariant": failing_invariant or ""}
    if extra:
        state.update(extra)
    # dev_score: HIGHER is better for best-of-k = the raw adaptation gain (base_ce - candidate_ce).
    reward = {"dev_score": float(dev_gain) if (correctness_ok and dev_gain is not None) else 0.0,
              "dev_metric": "adaptation gain base_ce - candidate_ce on the PUBLIC dev corpus (higher is better; NOT your graded score)",
              "dev_gain": dev_gain, "dev_base_ce": base_ce, "dev_candidate_ce": cand_ce,
              "correctness_ok": bool(correctness_ok)}
    (DEV_OUT / "verifier_state.json").write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    (DEV_OUT / "reward.json").write_text(json.dumps(reward, sort_keys=True, indent=2) + "\n")


def _harness_error(msg: str) -> None:
    (DEV_OUT / "harness_error.txt").write_text(msg + "\n")
    _write_state(False, INV_HARNESS, None, None, None, {"harness_message": msg})


def _import_entry(path: Path):
    spec = importlib.util.spec_from_file_location("dev_adapter_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dev_adapter_entry"] = module
    ws = str(path.parent.resolve())
    inserted = ws not in sys.path
    if inserted:
        sys.path.insert(0, ws)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(ws)
            except ValueError:
                pass
    return module


def _load_base(base_dir: Path, device):
    from transformers import AutoModelForCausalLM
    kw = dict(low_cpu_mem_usage=True)
    try:
        m = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype="auto", **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(str(base_dir), **kw)
    m.config.use_cache = False
    return m.to(device).eval()


def _tokenizer(base_dir: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(base_dir), use_fast=True)


def _logits_of(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    lg = getattr(out, "logits", None)
    if lg is None and isinstance(out, (tuple, list)) and out:
        lg = out[0]
    if lg is None and isinstance(out, dict):
        lg = out.get("logits")
    return lg


def _ce_over(model, tokens, device, seq_len, batch_seqs):
    import torch
    import torch.nn.functional as F
    n = tokens.numel()
    usable = ((n - 1) // seq_len) * seq_len
    total_seqs = usable // seq_len
    if total_seqs <= 0:
        raise ValueError("corpus too short for seq_len")
    loss_sum, tok = 0.0, 0
    with torch.inference_mode():
        for s in range(0, total_seqs, batch_seqs):
            e = min(s + batch_seqs, total_seqs)
            lo, hi = s * seq_len, e * seq_len + 1
            local = tokens[lo:hi].to(device=device)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out = model(x)
            lg = _logits_of(out)
            if not isinstance(lg, torch.Tensor):
                raise AssertionError("forward returned no logits tensor")
            V = lg.shape[-1]
            flat = lg.reshape(-1, V)
            tgt = y.reshape(-1).clamp_max(V - 1)
            tot = torch.zeros((), dtype=torch.float64, device=flat.device)
            for i in range(0, flat.shape[0], 4096):
                tot += F.cross_entropy(flat[i:i + 4096].float(), tgt[i:i + 4096],
                                       reduction="sum").to(torch.float64)
            loss_sum += float(tot.item())
            tok += int(flat.shape[0])
            del out, lg
    return loss_sum / max(tok, 1)


def _pos_var_and_arb_ce(model, device, seq_len, vocab):
    import torch
    import torch.nn.functional as F
    ar = torch.arange(seq_len, device=device, dtype=torch.int64) % max(vocab, 1)
    x = ar.reshape(1, -1)
    y = ((ar + 1) % max(vocab, 1)).reshape(1, -1)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        out = model(x)
    lg = _logits_of(out)
    sub = lg[:, :, : min(8192, lg.shape[-1])].float()
    pos_var = float(sub.var(dim=1).mean().item())
    V = lg.shape[-1]
    ce = float(F.cross_entropy(lg.reshape(-1, V).float(), y.reshape(-1).clamp_max(V - 1),
                               reduction="mean").item())
    return pos_var, ce


def main() -> int:
    try:
        cfg = _cfg()
    except Exception as exc:
        _harness_error(f"could not read dev manifest: {type(exc).__name__}: {exc}")
        return 0
    sub = Path(cfg["submission_dir"])
    art = sub / ARTIFACT_NAME
    ent = sub / ENTRY_NAME
    if not art.is_file() or not ent.is_file():
        _write_state(False, INV_MISSING, None, None, None,
                     {"detail": f"need {ARTIFACT_NAME} + {ENTRY_NAME} under {sub}"})
        return 0
    # byte-budget gate (the disclosed hard rule; declared-file dual-measurement)
    art_b, ent_b = art.stat().st_size, ent.stat().st_size
    total = art_b + ent_b
    if (art_b > int(cfg["max_artifact_bytes"]) or ent_b > int(cfg["max_entry_bytes"])
            or total > int(cfg["max_total_bytes"])):
        _write_state(False, INV_BUDGET, None, None, None,
                     {"detail": f"bytes adapter={art_b} entry={ent_b} total={total} "
                                f"caps=({cfg['max_artifact_bytes']},{cfg['max_entry_bytes']},{cfg['max_total_bytes']})"})
        return 0

    base_dir = Path(cfg["base_model_dir"])
    if not (base_dir / "model.safetensors").is_file():
        _harness_error(f"frozen base model missing at {base_dir}")
        return 0
    corpus = Path(cfg["dev_corpus"])
    if not corpus.exists():
        _harness_error(f"public dev corpus missing at {corpus}")
        return 0

    try:
        import torch
    except Exception as exc:
        _harness_error(f"torch import failed: {type(exc).__name__}: {exc}")
        return 0
    if not torch.cuda.is_available():
        _harness_error("CUDA not available for the dev proxy")
        return 0
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    seq_len = int(cfg["seq_len"])
    batch_seqs = int(cfg.get("batch_seqs", 2))

    try:
        tok = _tokenizer(base_dir)
        raw = corpus.read_text(encoding="utf-8", errors="replace")
        text = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("# provenance-marker"))
        want = int(cfg["dev_eval_tokens"])
        ids = []
        for i in range(0, len(text), 400_000):
            ids.extend(tok(text[i:i + 400_000], add_special_tokens=False)["input_ids"])
            if len(ids) >= want + seq_len + 2:
                break
        if len(ids) < seq_len + 2:
            _harness_error(f"public dev corpus too short: {len(ids)} tokens")
            return 0
        tokens = torch.tensor(ids[: want + seq_len + 2], dtype=torch.int64)
    except Exception as exc:
        _harness_error(f"tokenization failed: {type(exc).__name__}: {exc}")
        return 0

    # ---- base CE (the 0.0 anchor, measured live) ----
    try:
        base = _load_base(base_dir, device)
        base_ce = _ce_over(base, tokens, device, seq_len, batch_seqs)
        del base
        torch.cuda.empty_cache()
    except Exception as exc:
        _harness_error(f"base model eval failed: {type(exc).__name__}: {exc}")
        return 0

    # ---- candidate: fresh base + the byte-capped adapter ----
    try:
        base2 = _load_base(base_dir, device)
        module = _import_entry(ent)
        fn = getattr(module, "build_adapted_model", None)
        if fn is None:
            _write_state(False, INV_BUILD, None, base_ce, None,
                         {"detail": "adapter_entry.py has no build_adapted_model"})
            return 0
        sig = inspect.signature(fn)
        kwargs, positional = {}, []
        for name, p in sig.parameters.items():
            if name in {"base_model", "model", "base"}:
                kwargs[name] = base2
            elif name in {"artifact_path", "adapter_path", "path", "artifact"}:
                kwargs[name] = art
            elif name == "device":
                kwargs[name] = device
            elif p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                positional.append([base2, art, device][min(len(positional), 2)])
        model = fn(*positional, **kwargs)
        if isinstance(model, tuple) and len(model) == 2:
            model = model[1]
        if not isinstance(model, torch.nn.Module):
            _write_state(False, INV_BUILD, None, base_ce, None,
                         {"detail": "build_adapted_model did not return an nn.Module"})
            return 0
        model = model.to(device).eval()
    except Exception as exc:
        _write_state(False, INV_BUILD, None, base_ce, None,
                     {"detail": f"{type(exc).__name__}: {exc}"})
        return 0

    # self-checks (same anti-spoof invariants the grade enforces)
    try:
        n_params = sum(int(p.numel()) for p in model.parameters())
        if n_params < int(cfg["min_param_count"]):
            _write_state(False, INV_PARAM_FLOOR, None, base_ce, None,
                         {"detail": f"{n_params:,} < floor {int(cfg['min_param_count']):,}"})
            return 0
        vocab = int(getattr(getattr(model, "config", object()), "vocab_size", 0) or 151936)
        pos_var, arb_ce = _pos_var_and_arb_ce(model, device, seq_len, vocab)
        if not math.isfinite(pos_var) or pos_var <= float(cfg["logits_position_variance_floor"]):
            _write_state(False, INV_DEGENERATE, None, base_ce, None,
                         {"detail": f"logits ~constant across positions (var={pos_var:.2e})"})
            return 0
        cand_ce = _ce_over(model, tokens, device, seq_len, batch_seqs)
        lo, hi = float(cfg["min_plausible_ce"]), float(cfg["max_plausible_ce"])
        if not (math.isfinite(cand_ce) and lo <= cand_ce <= hi):
            _write_state(False, INV_CE_BAND, None, base_ce, cand_ce,
                         {"detail": f"candidate CE={cand_ce} outside [{lo},{hi}]"})
            return 0
        if not (cand_ce < float(cfg["ood_loss_ratio"]) * arb_ce):
            _write_state(False, INV_OOD, None, base_ce, cand_ce,
                         {"detail": f"CE(real)={cand_ce:.4f} >= {cfg['ood_loss_ratio']}*CE(arbitrary)={arb_ce:.4f}"})
            return 0
    except Exception as exc:
        _harness_error(f"candidate eval failed: {type(exc).__name__}: {exc}")
        return 0

    dev_gain = base_ce - cand_ce
    _write_state(True, None, dev_gain, base_ce, cand_ce,
                 {"detail": f"bytes total={total}/{cfg['max_total_bytes']}; base_ce={base_ce:.4f} "
                            f"cand_ce={cand_ce:.4f} gain={dev_gain:.4f}"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _harness_error(f"dev_eval crashed: {type(exc).__name__}: {exc}")
        raise SystemExit(0)

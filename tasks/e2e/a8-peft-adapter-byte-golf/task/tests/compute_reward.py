"""e2e-a8-peft-adapter-byte-golf verifier (family C, quality-under-an-ADAPTER-BYTE budget).

中topic A8 · 参数高效微调 PEFT. Single-shot, EVAL-ONLY: the harness never re-runs the
solver's fine-tune. It loads the FROZEN pretrained base model ITSELF (digest-pinned,
never the solver's copy), hands it to the solver's declared entry hook together with
the solver's byte-capped adapter artifact, and measures held-out cross-entropy on a
hidden code corpus the solver never sees.

The anti-spoof + stabilized-eval kit is lifted from the validated family-C siblings
(the sparsity-budget / quant-golf siblings, themselves verbatim from the
parameter-golf task family).

WHY THE BYTE BUDGET IS INFORMATION-THEORETICALLY AIRTIGHT (harness-owned measurement):
  * the harness loads the base weights from a root-owned, digest-verified directory,
    so the solver cannot pre-bake anything into them;
  * the ONLY bytes that travel from the solver's training run into the scored eval are
    the declared artifact + the declared entry module — the harness COPIES exactly those
    two files into a fresh staging dir and QUARANTINES every solver-writable path
    (/app/submission, /app/repo, the visible training corpus) for the duration of the
    eval, so no side-channel file can be read at load time;
  * both staged files are re-measured with stat() and summed against the cap (dual
    measurement: per-file caps + a total cap) — a self-reported size is never trusted;
  * the corpus quarantine also kills the "fine-tune inside build_adapted_model" evasion
    (no training data is reachable), and a wall-clock cap on the hook kills a long
    data-free search.

REWARD (the bench reward spec 性能类, BOUNDED to [0,1]):

    gain_ratio = (base_ce - candidate_ce) / (base_ce - strong_baseline_ce)
    reward     = min(1.0, ln(gain_ratio/ref_speedup)/ln(ref_speedup)) if gain_ratio > ref_speedup else 0.0

  where base_ce is MEASURED LIVE on the frozen base model in the same pass, strong_baseline_ce
  and ref_speedup are FROZEN CONSTANTS read from the manifest, and the scorer NEVER runs an
  oracle. gain_ratio 0.0 = no better than the frozen base; 1.0 = matched the strong reference at
  the same byte budget => reward EXACTLY 0 (tying the reference is worth nothing); ref_speedup =
  the MEASURED in-budget ceiling => reward 0.5; ref_speedup**2 => the reward saturates at 1.0.

  🔴 METRIC-CLASS CAVEAT, recorded not hidden: reward.md's 性能类 formula is defined on a
  WALL-CLOCK speedup measured by ABBA pairing. This task times NOTHING: the quantity fed to
  ln() is a QUALITY ratio (an adaptation-gain fraction on a hidden held-out split at a fixed
  327,680-byte adapter budget), so the result JSON reports `metric_kind`,
  `timing_measured: false` and `cv: 0` (the eval is deterministic, not a placeholder) alongside
  the schema-mandated `speedup` field. The arithmetic is legal: the measured in-budget ceiling
  is 1.2327x, comfortably above reward.md's ref_speedup >= 1.15 authoring floor, and the
  eval-side noise floor is at the float-repr level.

  Any hard-fail (byte budget / base tampered / build hook crash / anti-spoof / stash sweep /
  gain_ratio <= 1 / an invalid ref_speedup) => reward 0.0 with a NAMED reason.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import os
import resource
import shutil
import signal
import stat
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Paths (frozen surface resolution: env override -> fresh /tests mount -> baked
# root-0700 /opt/verifier fallback used by the in-build pre-push smoke)
# -----------------------------------------------------------------------------
SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/submission"))
BASE_MODEL_DIR = Path(os.environ.get("BASE_MODEL_DIR", "/app/base_model"))
VISIBLE_DATA_DIR = Path(os.environ.get("A8_DATA_DIR", "/data/datasets/a8_syscode"))
REPO_DIR = Path(os.environ.get("A8_REPO_DIR", "/app/repo"))
PEFT_RO = Path(os.environ.get("A8_PEFT_RO", "/opt/peft_ro/src"))
STAGE_DIR = Path(os.environ.get("A8_STAGE_DIR", "/tmp/e2e_a8_stage"))
QUARANTINE_ROOT = Path(os.environ.get("A8_QUARANTINE", "/tmp/e2e_a8_quarantine"))

ARTIFACT_NAME = "adapter.bin"
ENTRY_NAME = "adapter_entry.py"

# ---------------------------------------------------------------------------
# STASH SWEEP (2026-07-26 corpus-wide directive): a byte cap that only looks at the
# declared files is NOT enforceable — a candidate can stash adapter weights ANYWHERE
# (another directory, a tmpfs like /dev/shm or /tmp, a symlink/hardlink farm) and read
# them back from the load hook. So before the eval the harness sweeps EVERY writable
# mount, deduped by (dev, inode), diffs against a baked inventory of the pristine image,
# and for every file the solver created or modified either QUARANTINES it (data-ish, or
# larger than SMALL_FILE_LIMIT) or COUNTS its bytes against the budget (small sources).
# Nothing solver-made survives the eval window unaccounted for.
# ---------------------------------------------------------------------------
DATA_EXT = {".pt", ".pth", ".bin", ".safetensors", ".npy", ".npz", ".ckpt", ".gguf", ".h5",
            ".msgpack", ".pkl", ".pickle", ".joblib", ".zip", ".tar", ".tgz", ".gz", ".xz",
            ".zst", ".7z", ".arrow", ".parquet", ".onnx", ".engine", ".plan", ".dat"}
SWEEP_SKIP_PREFIXES = ("/proc", "/sys", "/dev/pts", "/dev/mqueue", "/run", "/logs", "/tests",
                       "/opt/verifier", "/opt/strong_baseline", "/opt/negative", "/app/base_model",
                       "/var/log")
SMALL_FILE_LIMIT = 32 * 1024          # a solver-created file above this is quarantined outright
# System paths churn at container start (package metadata, resolv.conf, apt lists) and the
# pristine inventory lists REGULAR FILES only, so a mismatch there is not evidence of a stash:
# those files are CHARGED to the budget instead of being moved (moving them breaks the
# interpreter - an earlier sweep quarantined /usr/bin/python3 because a symlink is never in a
# `find -type f` inventory). Symlinks are skipped outright: a symlink carries no bytes and its
# TARGET is swept independently, so a link into a stash dangles once the stash is quarantined.
SWEEP_CHARGE_ONLY_PREFIXES = ("/etc", "/var", "/usr", "/lib", "/lib64", "/bin", "/sbin", "/opt/conda")
INVENTORY_NAME = "baseline_inventory.tsv.gz"


def _frozen(name: str, env: str | None = None) -> Path:
    if env and os.environ.get(env):
        return Path(os.environ[env])
    for base in ("/tests", "/opt/verifier"):
        p = Path(base) / name
        if p.exists():
            return p
    return Path("/opt/verifier") / name


MANIFEST_PATH = _frozen("verifier-correctness-manifest.json", "E2E_MANIFEST_PATH")
HOLDOUT_CORPUS_PATH = _frozen("holdout_corpus.txt", "E2E_HOLDOUT_PATH")

# Fallback defaults — OVERRIDDEN by the frozen manifest once calibrated on NVIDIA H20.
_DEFAULTS: dict[str, Any] = {
    # --- the ADAPTER-BYTE budget (the disclosed rule; harness-remeasured, G2) ---
    "max_total_bytes": 327_680,          # adapter.bin + adapter_entry.py
    "max_artifact_bytes": 262_144,       # adapter.bin alone
    "max_entry_bytes": 65_536,           # adapter_entry.py alone (code, not a payload)
    "build_timeout_sec": 90,             # wall-clock cap on build_adapted_model
    # --- eval shape ---
    "seq_len": 1024,
    "batch_seqs": 2,
    "eval_tokens": 196_608,
    # --- the reward anchors ---
    "strong_baseline_ce": None,          # the gain_ratio == 1.0 denominator (CALIBRATED on NVIDIA H20)
    "ref_speedup": None,                 # FROZEN reward.md 性能类 constant: gain_ratio at reward 0.5.
                                         # The scorer only READS it — it never runs the oracle.
                                         # Missing/<=1 => HARD FAIL 0 with a named reason, never 1.0.
    "expected_base_ce": None,            # drift re-check for the frozen base model
    "base_ce_drift_tol": 0.05,           # 5% relative drift => flag (soft)
    # --- anti-spoof (parameter-golf kit) ---
    "min_param_count": 300_000_000,      # the frozen 0.5B base must actually be in there
    "min_plausible_ce": 0.20,            # below => forged-loss spoof (HARD)
    "max_plausible_ce": 12.0,            # above => not an LM at all (HARD)
    "ood_loss_ratio": 0.7,               # loss(real) < 0.7 * loss(arbitrary ids)
    "logits_position_variance_floor": 1e-4,
    "labels_loss_agreement_tol": 0.15,   # HF labels= loss vs harness CE (soft-ish)
    "base_model_sha256": None,           # digest of the frozen base weights (G2)
}


def load_manifest() -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    try:
        if MANIFEST_PATH.exists():
            m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            th = m.get("thresholds", m)
            for k in _DEFAULTS:
                if k in th and th[k] is not None:
                    cfg[k] = th[k]
    except Exception:
        pass
    return cfg


# -----------------------------------------------------------------------------
# Check / result helpers (sibling-verbatim)
# -----------------------------------------------------------------------------
class Check:
    def __init__(self, name, passed, message, details=None, hard=True):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.hard = hard


def result(name, passed, message, hard=True, **details):
    return Check(name, passed, message, details, hard)


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sanitize_python_path(*blocked_dirs: Path) -> None:
    """Drop cwd + every solver-writable dir from sys.path so a submitted module can
    never shadow the verifier's own imports (parameter-golf's guard)."""
    blocked = {Path.cwd().resolve()}
    for d in blocked_dirs:
        try:
            blocked.add(d.resolve())
        except OSError:
            pass
    clean = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            r = Path(entry).resolve()
        except OSError:
            continue
        if r in blocked:
            continue
        clean.append(entry)
    sys.path[:] = clean


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout("build_adapted_model exceeded its wall-clock cap")


# -----------------------------------------------------------------------------
# Step 1 — the ADAPTER-BYTE budget (dual-measured) + staging + quarantine
# -----------------------------------------------------------------------------
def stage_and_measure(cfg: dict[str, Any]) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    info: dict[str, Any] = {}
    art = SUBMISSION_DIR / ARTIFACT_NAME
    ent = SUBMISSION_DIR / ENTRY_NAME
    ok_art, ok_ent = art.is_file(), ent.is_file()
    checks.append(result(f"Required file: {ARTIFACT_NAME}", ok_art,
                         "present" if ok_art else f"missing at {art}"))
    checks.append(result(f"Required file: {ENTRY_NAME}", ok_ent,
                         "present" if ok_ent else f"missing at {ent}"))
    if not (ok_art and ok_ent):
        return checks, info

    art_bytes = art.stat().st_size
    ent_bytes = ent.stat().st_size
    total = art_bytes + ent_bytes
    info.update(adapter_bytes=art_bytes, entry_bytes=ent_bytes, total_bytes=total,
                adapter_sha256=sha256_file(art), entry_sha256=sha256_file(ent))
    cap_a, cap_e, cap_t = (int(cfg["max_artifact_bytes"]), int(cfg["max_entry_bytes"]),
                           int(cfg["max_total_bytes"]))
    checks.append(result("Adapter artifact within per-file cap", art_bytes <= cap_a,
                         f"{art_bytes:,} bytes <= {cap_a:,}" if art_bytes <= cap_a
                         else f"BUDGET EXCEEDED: {ARTIFACT_NAME} {art_bytes:,} > {cap_a:,} bytes",
                         adapter_bytes=art_bytes, max_artifact_bytes=cap_a))
    checks.append(result("Entry module within per-file cap", ent_bytes <= cap_e,
                         f"{ent_bytes:,} bytes <= {cap_e:,}" if ent_bytes <= cap_e
                         else f"BUDGET EXCEEDED: {ENTRY_NAME} {ent_bytes:,} > {cap_e:,} bytes",
                         entry_bytes=ent_bytes, max_entry_bytes=cap_e))
    checks.append(result("Total adaptation bytes within budget", total <= cap_t,
                         f"{total:,} bytes <= budget {cap_t:,} "
                         f"({ARTIFACT_NAME}={art_bytes:,} + {ENTRY_NAME}={ent_bytes:,})"
                         if total <= cap_t else
                         f"BUDGET EXCEEDED: total {total:,} > {cap_t:,} bytes",
                         total_bytes=total, max_total_bytes=cap_t))
    if not all(c.passed for c in checks):
        return checks, info

    # stage exactly the two declared files into a fresh dir
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(art, STAGE_DIR / ARTIFACT_NAME)
    shutil.copy2(ent, STAGE_DIR / ENTRY_NAME)
    info["staged"] = str(STAGE_DIR)
    return checks, info


def _writable_mounts() -> list[str]:
    """Every mount we should sweep, INCLUDING tmpfs (/dev/shm, /tmp, ...)."""
    roots = {"/"}
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            _dev, mnt, fstype, opts = parts[0], parts[1], parts[2], parts[3]
            if fstype in {"proc", "sysfs", "devpts", "cgroup", "cgroup2", "mqueue", "debugfs",
                          "tracefs", "securityfs", "pstore", "bpf", "configfs", "fusectl",
                          "hugetlbfs", "binfmt_misc", "nsfs", "autofs"}:
                continue
            if "rw" not in opts.split(","):
                continue
            roots.add(mnt)
    except OSError:
        pass
    return sorted(roots)


def _readonly_mount_targets() -> set[str]:
    """Mount targets whose mount is READ-ONLY, from /proc/self/mountinfo.

    🔴 Why this exists (MEASURED NVIDIA H20 2026-07-28). The container runtime bind-mounts NVIDIA
    driver assets over individual paths inside the image:

        /usr/lib/firmware/nvidia/<ver>/gsp_ga10x.bin   74,942,448 B
        /usr/share/nvidia/nvoptix.bin                  60,858,204 B
        /usr/lib/firmware/nvidia/<ver>/gsp_tu10x.bin   30,438,488 B

    They are absent from any image inventory (they do not exist at build time), their extension is
    in DATA_EXT, so the sweep tried to QUARANTINE them; `os.rename` fails EBUSY on a bind mount and
    `shutil.move` fails EBUSY too, so all 166,239,140 B were CHARGED to a 327,680 B budget. Result:
    EVERY submission - including the honest reference - failed the byte gate. This is the same
    "161 MB" the earlier EXDEV fix note refers to: that fix addressed cross-device renames, but the
    real cause is a bind mount, which no quarantine directory placement can ever move.

    Skipping them is NOT a cache denylist (which would rot the next time a library moves its cache).
    It is a CAPABILITY argument: these mounts are `ro` (mode 0444, `open(..., "r+b")` -> EROFS), and
    the container's CapEff is 0xa80425fb, i.e. WITHOUT CAP_SYS_ADMIN, so the solver can neither write
    them nor create a bind mount of its own. Bytes that the solver provably cannot author cannot be
    part of its adaptation. A WRITABLE bind mount (e.g. /dev/termination-log, mode 0666, `rw`) is
    deliberately NOT excluded and is still charged - that one IS a usable stash channel.
    """
    ro: set[str] = set()
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            if len(fields) < 6:
                continue
            target, opts = fields[4], fields[5]
            if "ro" in opts.split(","):
                ro.add(target)
    except OSError:
        pass
    return ro


def load_inventory() -> dict[str, tuple[int, int]] | None:
    """path -> (inode, size) for the pristine image, baked root-0700 at build time."""
    inv_path = _frozen(INVENTORY_NAME, "E2E_INVENTORY_PATH")
    if not inv_path.exists():
        return None
    inv: dict[str, tuple[int, int]] = {}
    try:
        with gzip.open(inv_path, "rt", errors="replace") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) != 3:
                    continue
                try:
                    inv[f[2]] = (int(f[0]), int(f[1]))
                except ValueError:
                    continue
    except OSError:
        return None
    return inv


def stash_sweep(cfg: dict[str, Any], keep: set[str]) -> tuple[list[Check], dict[str, Any], list[tuple[Path, Path]]]:
    """Sweep every writable mount; quarantine or count everything the solver created."""
    checks: list[Check] = []
    info: dict[str, Any] = {}
    inv = load_inventory()
    if inv is None:
        checks.append(result("Whole-filesystem stash sweep", False,
                             f"baked baseline inventory ({INVENTORY_NAME}) missing — the byte "
                             f"budget cannot be enforced, so the run FAILS CLOSED"))
        return checks, info, []
    QUARANTINE_ROOT.mkdir(parents=True, exist_ok=True)
    # a quarantine dir PER MOUNT: os.rename cannot cross devices (tmpfs vs the overlay), and an
    # EXDEV failure used to be charged to the budget, which failed the honest path (MEASURED
    # 161 MB charged). Same-device rename first, shutil.move only as a fallback.
    mount_q: dict[str, Path] = {}

    def _quarantine_dir_for(root: str) -> Path:
        q = mount_q.get(root)
        if q is None:
            q = Path(root) / ".e2e_sweep_quarantine"
            try:
                q.mkdir(parents=True, exist_ok=True)
            except OSError:
                q = QUARANTINE_ROOT
            mount_q[root] = q
        return q

    ro_mounts = _readonly_mount_targets()
    ro_prefixes = tuple(m.rstrip("/") + "/" for m in ro_mounts if m not in ("/",))
    n_ro = 0
    ro_bytes = 0
    seen_ino: set[tuple[int, int]] = set()
    moved: list[tuple[Path, Path]] = []
    counted_bytes = 0
    counted_files: list[str] = []
    system_bytes = 0
    n_system = 0
    quarantined_bytes = 0
    n_new = 0
    mounts = _writable_mounts()
    for root in mounts:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda e: None):
            if any(dirpath == p or dirpath.startswith(p + "/") for p in SWEEP_SKIP_PREFIXES):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames
                           if not any(os.path.join(dirpath, d) == p or
                                      os.path.join(dirpath, d).startswith(p + "/")
                                      for p in SWEEP_SKIP_PREFIXES)]
            if os.path.basename(dirpath) == ".e2e_sweep_quarantine":
                dirnames[:] = []
                continue
            for name in filenames:
                fp = os.path.join(dirpath, name)
                if fp in keep:
                    continue
                try:
                    st = os.lstat(fp)
                except OSError:
                    continue
                if os.path.islink(fp):         # carries no bytes; its target is swept separately
                    continue
                # 🔴 REGULAR FILES ONLY. A character/block device, FIFO or socket cannot carry
                #    solver bytes, and moving one is catastrophic: MEASURED, the
                #    sweep quarantined /dev/urandom//dev/zero, and the shutil.move fallback then
                #    COPIED an infinite device into a quarantine dir that lives on a tmpfs (i.e. in
                #    RAM) -> the verifier was OOM-killed at a 48 GiB limit and, at 128 GiB, sat
                #    blocked in the kernel's `_random_read` for 27+ minutes with VmPeak 232 GB.
                #    That is why NO honest path had ever completed on image :v3.
                #    This does NOT weaken the sweep: a stash must be a regular file, and both the
                #    symlink and the hardlink routes are still covered (hardlinks ARE regular files
                #    and are deduped by (dev, inode) above). The stash-oracle's /dev/shm/stash.bin
                #    is a regular file and is still detected.
                if not stat.S_ISREG(st.st_mode):
                    continue
                # runtime-injected READ-ONLY bind mount -> the solver provably cannot have authored
                # these bytes (ro mount + no CAP_SYS_ADMIN), and they can never be renamed (EBUSY).
                # Writable bind mounts are NOT excluded and are still charged.
                if fp in ro_mounts or fp.startswith(ro_prefixes):
                    n_ro += 1
                    ro_bytes += st.st_size
                    continue
                key = (st.st_dev, st.st_ino)
                if key in seen_ino:            # hardlink / already-seen inode -> counted once
                    continue
                seen_ino.add(key)
                base = inv.get(fp)
                if base is not None and base[1] == st.st_size:
                    continue                   # pristine image file
                n_new += 1
                size = st.st_size
                ext = os.path.splitext(name)[1].lower()
                if fp.startswith(SWEEP_CHARGE_ONLY_PREFIXES) and ext not in DATA_EXT:
                    # System paths are NOT solver territory and are NOT charged: the container
                    # runtime injects ~624 MB of CUDA driver libraries into /usr at start, which
                    # no image inventory can contain (MEASURED: charging them failed the honest
                    # path). A data-extension file here is still quarantined below.
                    system_bytes += size
                    n_system += 1
                    continue
                if ext in {".pyc", ".log"} and size <= SMALL_FILE_LIMIT:
                    continue                   # honest scratch: not importable once its source is gone
                if not stat.S_ISREG(st.st_mode):   # belt and braces: never move a device node
                    continue
                if True:
                    dst = _quarantine_dir_for(root) / f"sweep_{len(moved)}_{name[:40]}"
                    try:
                        os.rename(fp, dst)
                        moved.append((Path(fp), dst))
                        quarantined_bytes += size
                    except OSError:
                        try:
                            shutil.move(fp, str(dst))
                            moved.append((Path(fp), dst))
                            quarantined_bytes += size
                        except Exception:
                            counted_bytes += size  # truly immovable -> it must be paid for
                            counted_files.append(fp)
                else:
                    counted_bytes += size      # small solver source: readable, so it COUNTS
                    counted_files.append(fp)
    info.update(sweep_system_files=n_system, sweep_system_bytes=system_bytes,
                sweep_mounts=mounts, sweep_new_or_modified=n_new,
                sweep_quarantined_files=len(moved), sweep_quarantined_bytes=quarantined_bytes,
                sweep_counted_bytes=counted_bytes, sweep_counted_files=counted_files[:400],
                sweep_counted_files_total=len(counted_files),
                sweep_readonly_mount_files=n_ro, sweep_readonly_mount_bytes=ro_bytes)
    checks.append(result("Whole-filesystem stash sweep", True,
                         f"swept {len(mounts)} writable mounts (incl. tmpfs) deduped by "
                         f"(dev,inode): {n_new} solver-created/modified files -> "
                         f"{len(moved)} quarantined ({quarantined_bytes:,} B), "
                         f"{counted_bytes:,} B unmovable and CHARGED to the budget, "
                         f"{n_system} system/runtime-injected files ignored ({system_bytes:,} B), "
                         f"{n_ro} read-only bind-mounted runtime assets ignored ({ro_bytes:,} B)"))
    return checks, info, moved


def check_entry_source(cfg: dict[str, Any]) -> list[Check]:
    """A surviving daemon or an IPC channel would defeat a file-only sweep — reject a load
    hook that reaches for one."""
    src = (STAGE_DIR / ENTRY_NAME).read_text(errors="replace")
    banned = ["socket", "/dev/shm", "/proc/", "ptrace", "multiprocessing.connection",
              "xmlrpc", "http.client", "urllib.request", "requests.get"]
    hits = [b for b in banned if b in src]
    return [result("Load hook uses no side channel", not hits,
                   "no IPC / shared-memory / self-inspection primitives in the entry module"
                   if not hits else
                   f"the entry module references {hits} — the adaptation must arrive in the "
                   f"declared bytes, not through a channel")]


def quarantine(paths: list[Path]) -> list[tuple[Path, Path]]:
    """Move every solver-writable path aside for the eval window so the entry hook
    can read ONLY the staged artifact + the frozen base model + root-owned libs."""
    QUARANTINE_ROOT.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    for p in paths:
        if not p.exists():
            continue
        dst = QUARANTINE_ROOT / (p.name + "." + str(len(moved)))
        try:
            os.rename(p, dst)
            moved.append((p, dst))
        except OSError:
            try:
                shutil.move(str(p), str(dst))
                moved.append((p, dst))
            except Exception:
                pass
    return moved


def unquarantine(moved: list[tuple[Path, Path]]) -> None:
    for orig, dst in reversed(moved):
        try:
            if dst.exists() and not orig.exists():
                os.rename(dst, orig)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Step 2 — the frozen base model (G2: the harness owns the weights)
# -----------------------------------------------------------------------------
def check_base_model(cfg: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    w = BASE_MODEL_DIR / "model.safetensors"
    ok = w.is_file()
    checks.append(result("Frozen base weights present", ok,
                         f"{w} ({w.stat().st_size:,} bytes)" if ok else f"missing {w}"))
    if not ok:
        return checks
    exp = cfg.get("base_model_sha256")
    if exp:
        got = sha256_file(w)
        checks.append(result("Frozen base weights untampered (sha256)", got == exp,
                             "digest matches the frozen manifest" if got == exp
                             else f"BASE WEIGHTS TAMPERED: {got[:16]}... != {str(exp)[:16]}...",
                             base_model_sha256=got))
    else:
        checks.append(result("Frozen base weights untampered (sha256)", True,
                             f"manifest digest not pinned yet; measured {sha256_file(w)}",
                             hard=False, base_model_sha256=sha256_file(w)))
    return checks


def load_base_model(device: torch.device):
    from transformers import AutoModelForCausalLM
    kw = dict(low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL_DIR), torch_dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL_DIR), dtype=torch.bfloat16, **kw)
    model.config.use_cache = False
    return model.to(device).eval()


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), use_fast=True)


# -----------------------------------------------------------------------------
# Step 3 — hidden held-out tokens (G1) + stabilized CE eval
# -----------------------------------------------------------------------------
def hidden_tokens(cfg: dict[str, Any]) -> torch.Tensor:
    if not HOLDOUT_CORPUS_PATH.exists():
        raise FileNotFoundError(f"Held-out corpus not found at {HOLDOUT_CORPUS_PATH}")
    tok = load_tokenizer()
    raw = HOLDOUT_CORPUS_PATH.read_text(encoding="utf-8", errors="replace")
    text = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("# provenance-marker"))
    want = int(cfg["eval_tokens"])
    # tokenize in chunks until the token budget is met (bounded, deterministic)
    ids: list[int] = []
    step = 400_000
    for i in range(0, len(text), step):
        ids.extend(tok(text[i:i + step], add_special_tokens=False)["input_ids"])
        if len(ids) >= want + int(cfg["seq_len"]) + 2:
            break
    if len(ids) < int(cfg["seq_len"]) + 2:
        raise RuntimeError(f"held-out corpus too short: {len(ids)} tokens")
    return torch.tensor(ids[: want + int(cfg["seq_len"]) + 2], dtype=torch.int64)


def _logits_of(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    lg = getattr(out, "logits", None)
    if lg is None and isinstance(out, (tuple, list)) and out:
        lg = out[0]
    if lg is None and isinstance(out, dict):
        lg = out.get("logits")
    if not isinstance(lg, torch.Tensor):
        raise AssertionError(f"forward() returned {type(out).__name__} with no logits tensor")
    return lg


def _ce_chunked(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, int]:
    """fp32 CE with an fp64 accumulator, chunked over the flattened token dim so a
    151k-way vocab never materializes as one huge fp32 tensor."""
    V = logits.shape[-1]
    flat = logits.reshape(-1, V)
    tgt = y.reshape(-1)
    tot = torch.zeros((), dtype=torch.float64, device=flat.device)
    n = flat.shape[0]
    chunk = 4096
    for i in range(0, n, chunk):
        sl = flat[i:i + chunk].float()
        t = tgt[i:i + chunk].clamp_max(V - 1)
        tot += F.cross_entropy(sl, t, reduction="sum").to(torch.float64)
    return float(tot.item()), int(n)


def _eval_ce_one_pass(model, tokens: torch.Tensor, device, seq_len: int,
                      batch_seqs: int, offset: int) -> float:
    n = tokens.numel()
    if offset < 0 or offset >= n - 1:
        raise ValueError("offset out of range")
    usable = ((n - offset - 1) // seq_len) * seq_len
    total_seqs = usable // seq_len
    if total_seqs <= 0:
        raise ValueError("corpus too short")
    loss_sum, tok_count = 0.0, 0
    for s in range(0, total_seqs, batch_seqs):
        e = min(s + batch_seqs, total_seqs)
        lo = offset + s * seq_len
        hi = offset + e * seq_len + 1
        local = tokens[lo:hi].to(device=device, non_blocking=True)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = model(x)
        ls, nt = _ce_chunked(_logits_of(out), y)
        loss_sum += ls
        tok_count += nt
        del out
    return loss_sum / max(tok_count, 1)


def eval_ce(model, tokens, device, cfg) -> float:
    """Median of 3 shifted passes (parameter-golf stabilization)."""
    seq_len = int(cfg["seq_len"])
    bs = int(cfg["batch_seqs"])
    vals = []
    model.eval()
    with torch.inference_mode():
        for off in (0, seq_len // 3, (2 * seq_len) // 3):
            try:
                vals.append(_eval_ce_one_pass(model, tokens, device, seq_len, bs, off))
            except ValueError:
                continue
    if not vals:
        raise RuntimeError("corpus too short for any shifted pass")
    vals.sort()
    return float(vals[len(vals) // 2])


# -----------------------------------------------------------------------------
# Step 4 — anti-spoof probes (parameter-golf kit, adapted to an HF CausalLM)
# -----------------------------------------------------------------------------
def probe_model(model, device, cfg, tag: str) -> tuple[list[Check], float]:
    checks: list[Check] = []
    seq_len = int(cfg["seq_len"])
    V_cfg = int(getattr(getattr(model, "config", object()), "vocab_size", 0) or 151936)
    ar = torch.arange(seq_len, device=device, dtype=torch.int64) % V_cfg
    x = ar.reshape(1, -1)
    y = ((ar + 1) % V_cfg).reshape(1, -1)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        out = model(x)
    logits = _logits_of(out)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != seq_len:
        checks.append(result(f"[{tag}] forward(x) logits shape",
                             False, f"shape {tuple(logits.shape)} != (1,{seq_len},V)"))
        return checks, float("nan")
    if not logits.dtype.is_floating_point:
        checks.append(result(f"[{tag}] forward(x) logits dtype", False, f"{logits.dtype} not float"))
        return checks, float("nan")
    sub = logits[:, :, : min(8192, logits.shape[-1])].float()
    pos_var = float(sub.var(dim=1).mean().item())
    floor = float(cfg["logits_position_variance_floor"])
    checks.append(result(f"[{tag}] logits vary across positions",
                         math.isfinite(pos_var) and pos_var > floor,
                         f"position variance {pos_var:.3e} > {floor:.1e}" if pos_var > floor
                         else f"logits ~constant across positions (var={pos_var:.3e}) — degenerate stub",
                         position_variance=pos_var))
    loss_ar, _ = _ce_chunked(logits, y)
    loss_ar /= float(y.numel())
    del out, logits, sub
    torch.cuda.empty_cache()
    return checks, float(loss_ar)


# -----------------------------------------------------------------------------
# Step 5 — the solver's entry hook
# -----------------------------------------------------------------------------
def import_entry(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_adapter_entry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_adapter_entry"] = module
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


def call_build(module, base_model, artifact: Path, device, cfg):
    fn = getattr(module, "build_adapted_model", None)
    if fn is None:
        raise RuntimeError(f"{ENTRY_NAME} must define "
                           "build_adapted_model(base_model, artifact_path, device)")
    sig = inspect.signature(fn)
    kwargs, positional = {}, []
    for name, p in sig.parameters.items():
        if name in {"base_model", "model", "base"}:
            kwargs[name] = base_model
        elif name in {"artifact_path", "adapter_path", "path", "artifact"}:
            kwargs[name] = artifact
        elif name == "device":
            kwargs[name] = device
        elif p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            positional.append([base_model, artifact, device][min(len(positional), 2)])
    cap = int(cfg["build_timeout_sec"])
    t0 = time.monotonic()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(cap)
    try:
        model = fn(*positional, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    dt = time.monotonic() - t0
    if isinstance(model, tuple) and len(model) == 2:
        model = model[1]
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_adapted_model must return an nn.Module")
    return model.to(device).eval(), dt


# -----------------------------------------------------------------------------
# The full run
# -----------------------------------------------------------------------------
def run_all(cfg: dict[str, Any]) -> tuple[dict[str, list[Check]], dict[str, Any]]:
    checks: dict[str, list[Check]] = {}
    lb: dict[str, Any] = {}

    checks["budget"], info = stage_and_measure(cfg)
    lb.update({k: v for k, v in info.items() if k != "staged"})
    if not all(c.passed for c in checks["budget"]):
        return checks, lb

    checks["base_model"] = check_base_model(cfg)
    if not all(c.passed for c in checks["base_model"] if c.hard):
        return checks, lb

    if not torch.cuda.is_available():
        checks["env"] = [result("CUDA available", False, "CUDA required for the H20 eval")]
        return checks, lb
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    tokens = hidden_tokens(cfg)
    lb["heldout_eval_tokens"] = int(tokens.numel())

    # --- (a) the FROZEN BASE model: the 0.0 anchor, measured live every run ---
    base = load_base_model(device)
    base_probe, base_arb = probe_model(base, device, cfg, "base")
    base_ce = eval_ce(base, tokens, device, cfg)
    n_params_base = sum(int(p.numel()) for p in base.parameters())
    del base
    torch.cuda.empty_cache()
    lb["base_ce"] = base_ce
    exp = cfg.get("expected_base_ce")
    drift_ok, drift_msg = True, f"live base_ce={base_ce:.4f} (manifest value not pinned yet)"
    if exp:
        rel = abs(base_ce - float(exp)) / float(exp)
        drift_ok = rel <= float(cfg["base_ce_drift_tol"])
        drift_msg = (f"live base_ce={base_ce:.4f} vs manifest {float(exp):.4f} "
                     f"(rel drift {rel:.2%})")
    checks["base_anchor"] = base_probe + [
        result("Frozen-base anchor re-measured (drift check)", drift_ok, drift_msg,
               hard=False, base_ce=base_ce, expected_base_ce=exp),
    ]

    # --- (b) the CANDIDATE: fresh base load + the solver's byte-capped adapter ---
    cand: list[Check] = []
    keep = {str(STAGE_DIR / ARTIFACT_NAME), str(STAGE_DIR / ENTRY_NAME)}
    sweep_checks, sweep_info, sweep_moved = stash_sweep(cfg, keep)
    cand += sweep_checks
    lb.update(sweep_info)
    if not all(c.passed for c in sweep_checks):
        checks["candidate"] = cand
        return checks, lb
    # the sweep's small-file charge is part of the SAME budget the declared files pay
    charged = int(lb.get("total_bytes", 0)) + int(sweep_info.get("sweep_counted_bytes", 0))
    cap_t = int(cfg["max_total_bytes"])
    cand.append(result("Total adaptation bytes incl. swept solver files", charged <= cap_t,
                       f"{charged:,} B <= budget {cap_t:,} (declared "
                       f"{lb.get('total_bytes')} + swept-readable "
                       f"{sweep_info.get('sweep_counted_bytes')})" if charged <= cap_t else
                       f"BUDGET EXCEEDED once swept solver files are charged: {charged:,} > {cap_t:,} B",
                       charged_total_bytes=charged))
    cand += check_entry_source(cfg)
    if not all(c.passed for c in cand if c.hard):
        unquarantine(sweep_moved)
        checks["candidate"] = cand
        return checks, lb
    moved = quarantine([SUBMISSION_DIR, REPO_DIR, VISIBLE_DATA_DIR]) + sweep_moved
    try:
        base2 = load_base_model(device)
        if PEFT_RO.is_dir() and str(PEFT_RO) not in sys.path:
            sys.path.append(str(PEFT_RO))
        module = import_entry(STAGE_DIR / ENTRY_NAME)
        model, build_sec = call_build(module, base2, STAGE_DIR / ARTIFACT_NAME, device, cfg)
        cand.append(result("build_adapted_model returned an nn.Module within its cap", True,
                           f"built in {build_sec:.1f}s (cap {cfg['build_timeout_sec']}s)",
                           build_seconds=build_sec))
    except Exception as exc:
        cand.append(result("build_adapted_model returned an nn.Module within its cap", False,
                           f"{type(exc).__name__}: {exc}"))
        checks["candidate"] = cand
        unquarantine(moved)
        return checks, lb
    try:
        n_params = sum(int(p.numel()) for p in model.parameters())
        floor = int(cfg["min_param_count"])
        cand.append(result("Adapted model parameter count", n_params >= floor,
                           f"{n_params:,} params (floor {floor:,}; base has {n_params_base:,})",
                           n_params=n_params))
        if n_params < floor:
            checks["candidate"] = cand
            return checks, lb
        probes, cand_arb = probe_model(model, device, cfg, "candidate")
        cand += probes
        if not all(c.passed for c in probes):
            checks["candidate"] = cand
            return checks, lb
        cand_ce = eval_ce(model, tokens, device, cfg)
        lb["candidate_ce"] = cand_ce
        lo, hi = float(cfg["min_plausible_ce"]), float(cfg["max_plausible_ce"])
        cand.append(result("Held-out CE is plausible (anti-spoof band)",
                           math.isfinite(cand_ce) and lo <= cand_ce <= hi,
                           f"held-out CE={cand_ce:.4f} in [{lo},{hi}]"
                           if math.isfinite(cand_ce) and lo <= cand_ce <= hi
                           else f"held-out CE={cand_ce} outside the plausible band [{lo},{hi}] "
                                f"(forged-loss spoof or non-LM)",
                           candidate_ce=cand_ce))
        ratio = float(cfg["ood_loss_ratio"])
        thr = ratio * cand_arb
        cand.append(result("OOD: real held-out CE << arbitrary-ids CE",
                           math.isfinite(cand_ce) and cand_ce < thr,
                           f"CE(real)={cand_ce:.4f} < {ratio}*CE(arbitrary)={thr:.4f}"
                           if cand_ce < thr else
                           f"CE(real)={cand_ce:.4f} >= {ratio}*CE(arbitrary)={thr:.4f} "
                           f"— the model ignores input content",
                           ce_arbitrary=cand_arb))
        cand.append(result("Adaptation improved on the frozen base (soft floor)",
                           math.isfinite(cand_ce) and cand_ce < base_ce,
                           f"candidate CE {cand_ce:.4f} < base CE {base_ce:.4f}"
                           if cand_ce < base_ce else
                           f"candidate CE {cand_ce:.4f} did NOT improve on the frozen base "
                           f"{base_ce:.4f}", hard=False))
        lb["verifier_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    finally:
        unquarantine(moved)
    checks["candidate"] = cand
    return checks, lb


# -----------------------------------------------------------------------------
# Reward + the verifier output contract
# -----------------------------------------------------------------------------
def gain_ratio(lb: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    """The DIAGNOSTIC adaptation-gain ratio: (base_ce - candidate_ce) / (base_ce - strong_ce).

    1.0 == matched the strong reference recipe at the same adapter-byte budget. This is a QUALITY
    ratio, not a wall-clock speedup; it is what the bounded log reward consumes."""
    base_ce, cand_ce = lb.get("base_ce"), lb.get("candidate_ce")
    strong = cfg.get("strong_baseline_ce")
    for v in (base_ce, cand_ce, strong):
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return None
    denom = float(base_ce) - float(strong)
    if denom <= 0:
        return None
    return (float(base_ce) - float(cand_ce)) / denom


# 🔴 reward.md 性能类 前置门 -> a NAMED reason per failing gate (never a bare 0).
#    Keys are matched EXACTLY first, then as a suffix (the anti-spoof probes are named
#    "[base] ..." / "[candidate] ...").
_REASON_BY_CHECK = {
    "Required file: adapter.bin": "build_or_entry_contract_failed",
    "Required file: adapter_entry.py": "build_or_entry_contract_failed",
    "Adapter artifact within per-file cap": "adapter_byte_budget_exceeded",
    "Entry module within per-file cap": "adapter_byte_budget_exceeded",
    "Total adaptation bytes within budget": "adapter_byte_budget_exceeded",
    "Total adaptation bytes incl. swept solver files": "adapter_byte_budget_exceeded",
    "Frozen base weights present": "build_import_or_readiness_failed",
    "Frozen base weights untampered (sha256)": "forbidden_edit_path",
    "CUDA available": "build_import_or_readiness_failed",
    "Whole-filesystem stash sweep": "anti_cheat_budget_unenforceable",
    "Load hook uses no side channel": "anti_cheat_side_channel",
    "build_adapted_model returned an nn.Module within its cap": "build_import_or_readiness_failed",
    "Adapted model parameter count": "anti_cheat_param_floor",
    "Held-out CE is plausible (anti-spoof band)": "anti_cheat_implausible_ce",
    "OOD: real held-out CE << arbitrary-ids CE": "anti_cheat_not_a_real_lm",
    "Verifier error": "build_or_entry_contract_failed",
}
_REASON_BY_SUFFIX = {
    "logits vary across positions": "anti_cheat_degenerate_logits",
    "forward(x) logits shape": "build_or_entry_contract_failed",
    "forward(x) logits dtype": "build_or_entry_contract_failed",
}


def _reason_for(name: str) -> str:
    if name in _REASON_BY_CHECK:
        return _REASON_BY_CHECK[name]
    for suf, r in _REASON_BY_SUFFIX.items():
        if name.endswith(suf):
            return r
    return "correctness_case_failed"


def compute_log_reward(all_hard: bool, checks: dict[str, list[Check]], lb: dict[str, Any],
                       cfg: dict[str, Any]) -> tuple[float, list[str], float | None]:
    """reward.md 性能类: reward = min(1.0, ln(ratio/ref_speedup)/ln(ref_speedup)) if ratio > ref_speedup else 0.0, bounded to [0,1].

    Implements all SIX pre-gates with named reasons. Returns (reward, hard_fail_reasons, ratio)."""
    reasons: list[str] = []
    # pre-gates 1-4: build / import / readiness, any correctness case failing, cheating,
    # touching a forbidden edit path — all surfaced as failing HARD checks.
    for group in checks.values():
        for c in group:
            if c.hard and not c.passed:
                r = _reason_for(c.name)
                if r not in reasons:
                    reasons.append(r)
    ratio = gain_ratio(lb, cfg)
    if ratio is None or not math.isfinite(ratio):
        if "metric_unavailable" not in reasons:
            reasons.append("metric_unavailable")
    ref = cfg.get("ref_speedup")
    ref_ok = isinstance(ref, (int, float)) and math.isfinite(float(ref)) and float(ref) > 1.0
    if not ref_ok:
        # pre-gate 6: 参考解无效 / not calibrated. NEVER silently treat a missing ref as 1.0.
        reasons.append("ref_speedup_invalid_or_missing")
    if ratio is not None and math.isfinite(ratio) and ratio <= 1.0:
        # pre-gate 5: 未跨过基线 — tying or losing to the strong reference scores 0.
        reasons.append("speedup_not_above_baseline")
    if not all_hard and not reasons:
        reasons.append("zero_without_named_reason")
    if reasons:
        return 0.0, reasons, ratio
    reward = max(0.0, min(1.0, math.log(float(ratio)) / math.log(float(ref)) - 1.0))
    reward = max(0.0, min(1.0, reward))
    return float(reward), [], ratio


def write_outputs(checks: dict[str, list[Check]], lb: dict[str, Any], cfg: dict[str, Any]) -> bool:
    total = sum(len(g) for g in checks.values())
    passed = sum(1 for g in checks.values() for c in g if c.passed)
    all_hard = all(c.passed for g in checks.values() for c in g if c.hard)
    all_pass = all(c.passed for g in checks.values() for c in g)
    for group in checks.values():
        for c in group:
            for k, v in c.details.items():
                lb.setdefault(k, v)
    reward, hard_fail_reasons, ratio = compute_log_reward(all_hard, checks, lb, cfg)
    lb["gain_ratio_vs_strong_baseline"] = ratio
    payload_checks = [{"category": cat, "name": c.name, "passed": c.passed,
                       "message": c.message, "hard": c.hard, "details": c.details}
                      for cat, group in checks.items() for c in group]
    # reward.md §结果 JSON (performance) + the descriptive fields that say what `speedup` really is.
    core = {
        "task_type": "performance",
        "reward": reward,
        "hard_fail_reasons": hard_fail_reasons,
        "speedup": ratio,                       # schema-compatible name; see metric_kind
        "ref_speedup": cfg.get("ref_speedup"),
        "cv": {"baseline": 0.0, "candidate": 0.0},
        "metric_kind": "quality_ratio_NOT_time_speedup",
        "metric_name": "adaptation_gain_ratio",
        "metric_direction": "higher_is_better",
        "timing_measured": False,
    }
    metrics = dict(core)
    metrics.update({
        "partial_score": reward,
        "binary_pass": 1 if reward > 0 else 0,
        "quality_gate_passed": all_hard,
        "all_gates_pass": all_pass,
        "reward_form": ("reward.md 性能类 (bounded): reward = min(1.0, ln(gain_ratio/ref_speedup)/"
                        "ln(ref_speedup)) if gain_ratio > ref_speedup else 0.0, "
                        "gain_ratio = (base_ce - candidate_ce) / "
                        "(base_ce - strong_baseline_ce). gain_ratio <= ref_speedup (tying the strong "
                        "reference OR merely reaching the MEASURED in-budget ceiling) => 0; "
                        "gain_ratio == ref_speedup**1.5 => 0.5; "
                        "gain_ratio >= ref_speedup**2 => 1.0. ref_speedup is a "
                        "FROZEN calibrated constant read from the manifest — the scorer never "
                        "runs the oracle. BOUNDED replacement "
                        "for the old open-ended un-capped ratio."),
        "score_metric": "heldout_ce",
        "strong_baseline_ce": cfg.get("strong_baseline_ce"),
        "max_total_bytes": cfg.get("max_total_bytes"),
        "gain_ratio_vs_strong_baseline": ratio,
        "passed": passed, "total": total,
        "tests": {"passed": passed, "total": total},
        "pass_rate": passed / total if total else 0.0,
        "leaderboard": lb,
        "checks": payload_checks,
        "failed_checks": [c for c in payload_checks if not c["passed"]],
    })
    vdir = Path("/logs/verifier")
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (vdir / "verifier_state.json").write_text(json.dumps(
        {"task_id": "e2e-a8-peft-adapter-byte-golf", "mode": os.environ.get("VERIFIER_MODE", "candidate"),
         "task_type": "performance", "all_hard_pass": all_hard, "reward": reward,
         "hard_fail_reasons": hard_fail_reasons, "speedup": ratio,
         "ref_speedup": cfg.get("ref_speedup"), "ts": time.time()}, indent=2, default=str) + "\n", encoding="utf-8")
    (vdir / "correctness_results.json").write_text(json.dumps(
        {"checks": payload_checks, "passed": passed, "total": total,
         "all_hard_gates_pass": all_hard, "all_gates_pass": all_pass,
         "hard_fail_reasons": hard_fail_reasons,
         "failed_checks": [c for c in payload_checks if not c["passed"]]}, indent=2, default=str) + "\n", encoding="utf-8")
    (vdir / "benchmark_results.json").write_text(json.dumps(
        {"score_metric": "heldout_ce", "base_ce": lb.get("base_ce"),
         "candidate_ce": lb.get("candidate_ce"),
         "strong_baseline_ce": cfg.get("strong_baseline_ce"),
         "gain_ratio_vs_strong_baseline": ratio,
         "ref_speedup": cfg.get("ref_speedup"),
         "metric_kind": "quality_ratio_NOT_time_speedup", "timing_measured": False,
         "cv": {"baseline": 0.0, "candidate": 0.0},
         "adapter_bytes": lb.get("adapter_bytes"), "entry_bytes": lb.get("entry_bytes"),
         "total_bytes": lb.get("total_bytes")}, indent=2, default=str) + "\n", encoding="utf-8")
    (vdir / "reward.json").write_text(json.dumps(core, indent=2, default=str) + "\n", encoding="utf-8")
    (vdir / "reward.txt").write_text(f"{reward:.6f}\n", encoding="utf-8")
    return all_hard


def main() -> int:
    cfg = load_manifest()
    sanitize_python_path(SUBMISSION_DIR, REPO_DIR, STAGE_DIR)
    lb: dict[str, Any] = {}
    try:
        checks, lb = run_all(cfg)
    except Exception as exc:
        checks = {"verifier_error": [result("Verifier error", False, f"{type(exc).__name__}: {exc}")]}
    all_hard = write_outputs(checks, lb, cfg)
    for group in checks.values():
        for c in group:
            print(f"[{'PASS' if c.passed else 'FAIL'}{'' if c.hard else ' soft'}] {c.name}: {c.message}")
    reward, reasons, ratio = compute_log_reward(all_hard, checks, lb, cfg)
    print(f"REWARD={reward:.6f} gain_ratio={ratio} ref_speedup={cfg.get('ref_speedup')} "
          f"reasons={reasons} (base_ce={lb.get('base_ce')} candidate_ce={lb.get('candidate_ce')} "
          f"strong_baseline_ce={cfg.get('strong_baseline_ce')} "
          f"bytes={lb.get('total_bytes')}/{cfg.get('max_total_bytes')})")
    return 0 if all_hard else 1


if __name__ == "__main__":
    raise SystemExit(main())

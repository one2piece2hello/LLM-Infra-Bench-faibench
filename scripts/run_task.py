#!/usr/bin/env python3
"""fai_bench runner — build a task image, let an agent solve it, capture the submission, grade it.

Single file, standard library only. Runs one task or the whole corpus.

    python3 scripts/run_task.py --task tasks/kfc/<dir> --agent claude-code --model claude-opus-5
    python3 scripts/run_task.py --task tasks/e2e/<dir> --agent codex      --model gpt-5.6
    python3 scripts/run_task.py --tasks-root tasks --n-tasks 10 --sample-seed 0 --agent claude-code
    python3 scripts/run_task.py --task tasks/lh/<dir> --agent oracle   # self-check: run solution/solve.sh
    python3 scripts/run_task.py --task tasks/lh/<dir> --agent none     # pristine baseline (expect ~0)

Pipeline (see --help for the full contract):

    (1) build      docker buildx build --load -f environment/Dockerfile -t <docker_image> .
                   context = the PACKAGE ROOT (not environment/); .dockerignore lives there too.
    (2) agent      docker run -d <resources> <image> -> inject the agent CLI -> run the agent
    (3) artifacts  run pre_artifacts.sh in the same container -> /logs/artifacts/model.patch
    (4) verify     tests/ -> /tests (read-only), bash /tests/test.sh, always --network=none
    (5) collect    /logs/{verifier,artifacts} -> runs/<id>/{verifier,artifacts} -> run.json + summary.jsonl

Submission contract: the agent leaves its work in the WORKING TREE. The runner never asks an
agent to `git commit` and never moves HEAD — every task's verifier diffs the working tree
against the single baked baseline commit, so a moved HEAD makes a correct solution score 0.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    import tomllib as toml_r                                    # py >= 3.11
except ModuleNotFoundError:
    try:
        import tomli as toml_r                                  # type: ignore
    except ModuleNotFoundError:
        toml_r = None      # neither present -> _MiniToml.load below (stdlib-only fallback)


class _MiniToml:
    """Last-resort TOML reader for hosts on Python < 3.11 without `tomli`.

    Handles exactly what this runner reads from task.toml: top-level scalars, `[table]`
    and dotted `[a.b]` headers, and string/int/float/bool/array values (arrays may span
    lines). It is NOT a full TOML parser -- inline tables and multi-line strings are not
    supported -- but every task.toml in this corpus stays within this subset, and the
    real parser is used whenever tomllib/tomli is available."""

    @staticmethod
    def _val(tok):
        tok = tok.strip()
        if not tok:
            return None
        if tok[0] in "\"'":
            return tok[1:-1] if len(tok) >= 2 and tok[-1] == tok[0] else tok.strip("\"'")
        if tok in ("true", "false"):
            return tok == "true"
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            return tok

    @classmethod
    def _array(cls, body):
        out, buf, depth, q = [], "", 0, ""
        for ch in body:
            if q:
                buf += ch
                if ch == q:
                    q = ""
            elif ch in "\"'":
                q = ch
                buf += ch
            elif ch == "[":
                depth += 1
                buf += ch
            elif ch == "]" and depth:
                depth -= 1
                buf += ch
            elif ch == "," and depth == 0:
                if buf.strip():
                    out.append(cls._val(buf))
                buf = ""
            else:
                buf += ch
        if buf.strip():
            out.append(cls._val(buf))
        return out

    @classmethod
    def loads(cls, text):
        root, cur = {}, None
        i, lines = 0, text.split("\n")
        while i < len(lines):
            raw = lines[i]
            i += 1
            s = _strip_comment(raw).strip()
            if not s:
                continue
            if s.startswith("[") and s.endswith("]"):
                path = s[1:-1].strip().strip("[]").strip()
                node = root
                for part in path.split("."):
                    part = part.strip().strip("\"'")
                    node = node.setdefault(part, {})
                cur = node
                continue
            if "=" not in s:
                continue
            key, val = s.split("=", 1)
            key = key.strip().strip("\"'")
            val = val.strip()
            if val.startswith("["):                       # array, possibly multi-line
                while val.count("[") > val.count("]") and i < len(lines):
                    val += " " + _strip_comment(lines[i]).strip()
                    i += 1
                parsed = cls._array(val[val.index("[") + 1: val.rindex("]")])
            else:
                parsed = cls._val(val)
            (cur if cur is not None else root)[key] = parsed
        return root


def _strip_comment(line):
    """Drop a trailing # comment, but not a # that sits inside a quoted value."""
    q = ""
    for idx, ch in enumerate(line):
        if q:
            if ch == q:
                q = ""
        elif ch in "\"'":
            q = ch
        elif ch == "#":
            return line[:idx]
    return line

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # the fai_bench package root

STAGE_DIR = "/tmp/fai-runner"          # in-container staging dir for runner-generated files
TESTS_MNT = "/tests"                   # where the grading surface is mounted / copied
LOGS_MNT = "/logs"                     # container log root; bind-mounted so artifacts come back
AGENT_BIN_DIR = "/opt/fai-agent-bin"   # mount point when --agent-bin is a directory

# Conventional (NOT site-specific) locations of the NVIDIA user-space driver on hosts that have
# no nvidia-container-toolkit. Both are overridable: --nvidia-lib-dir / --nvidia-bin-dir.
DEFAULT_NVIDIA_LIB_DIR = "/usr/local/nvidia/lib64"
DEFAULT_NVIDIA_BIN_DIR = "/usr/local/nvidia/bin"
NVIDIA_CONTROL_DEVICES = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]

# --------------------------------------------------------------------------- agent adapters
# One dict entry per agent. Adding an agent = adding one entry.
#   bin       executable expected on PATH inside the container ("" = nothing to inject)
#   install   shell command that installs the CLI inside the container (needs egress)
#   cmd       shell command template run inside the container; $FAI_MODEL / $FAI_PROMPT /
#             $FAI_WORKDIR are exported for it. Override wholesale with --agent-cmd.
#   env_keys  host env vars forwarded by NAME only (`docker run -e KEY`), never by value, so
#             no credential ever reaches the process table or a log line.
#   set_env   fixed env values the CLI needs in a container
#   net_allow the minimal API allowlist this agent needs (recorded in run.json; enforced by the
#             external proxy in --agent-net proxy, NOT by docker, which has no domain allowlist)
#   needs_api True -> the agent step needs egress to a model API
AGENTS = {
    "claude-code": dict(
        bin="claude",
        install="npm install -g @anthropic-ai/claude-code@${FAI_AGENT_VERSION:-latest}",
        cmd=('claude --print --model "$FAI_MODEL" --output-format stream-json --verbose '
             '--dangerously-skip-permissions "$(cat "$FAI_PROMPT")"'),
        env_keys=["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                  "ANTHROPIC_MODEL", "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "MAX_THINKING_TOKENS"],
        # claude refuses --dangerously-skip-permissions as uid 0 unless it is told it is sandboxed
        set_env={"IS_SANDBOX": "1"},
        net_allow=["api.anthropic.com"],
        needs_api=True,
    ),
    "codex": dict(
        bin="codex",
        install="npm install -g @openai/codex@${FAI_AGENT_VERSION:-latest}",
        cmd=('codex exec --model "$FAI_MODEL" --cd "$FAI_WORKDIR" --skip-git-repo-check '
             '--dangerously-bypass-approvals-and-sandbox "$(cat "$FAI_PROMPT")"'),
        env_keys=["OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_HOME"],
        set_env={},
        net_allow=["api.openai.com"],
        needs_api=True,
    ),
    # No model in the loop: land the task's own reference solution the same way an agent would
    # (working tree, no commit). This is the runner's self-check path.
    "oracle": dict(bin="", install="", cmd="", env_keys=[], set_env={},
                   net_allow=[], needs_api=False),
    # Nothing at all: build the image and grade the pristine tree. Expect ~0.
    "none": dict(bin="", install="", cmd="", env_keys=[], set_env={},
                 net_allow=[], needs_api=False),
}

GPT_FAMILY = re.compile(r'^(gpt|o[1345]([-_.]|$)|codex)', re.I)

EPILOG = """\
NETWORK POLICY (tasks all declare allow_internet = false)
  Grading always runs with --network=none. There is no flag to relax that except
  --allow-networked-verify, which only exists to let you knowingly reuse a networked agent
  container for grading; the default never does.
  The agent step is --network=none too unless the agent needs a model API. Docker has NO
  domain allowlist, so only two honest choices exist and the conservative one is the default:
    --agent-net proxy --net-proxy URL   HTTPS_PROXY/HTTP_PROXY are injected into the agent step
                                        only; the allowlist is enforced by that proxy. RECOMMENDED.
    --agent-net host                    the agent step shares the host network namespace: FULL
                                        egress plus reachability of host-local services. Fast to
                                        set up, no allowlist at all -- recorded as a risk in run.json.
    --agent-net bridge                  full egress in its own namespace (no host services).
    --agent-net none                    offline (the default, and the only option for oracle/none).
  --agent-net auto (default) = none for oracle/none; proxy when a proxy URL is available
  (--net-proxy or HTTPS_PROXY in the environment); otherwise it FAILS the task with a clear
  message instead of silently escalating to full egress.

AGENT CLI INJECTION (no image in this corpus ships an agent CLI -- verified 91/91 Dockerfiles)
  --agent-bin PATH      bind-mount a CLI from the host, read-only. A file lands on
                        /usr/local/bin/<name>; a directory lands on %(binmnt)s and is
                        prepended to PATH. A CLI that needs its own runtime (e.g. an npm
                        install that shells out to node) needs that runtime too: add
                        --agent-extra-mount /host/node/dir:/opt/fai-node:ro and put it on PATH
                        with --agent-path-prepend /opt/fai-node/bin.
  --agent-install       install the CLI inside the container instead (needs agent egress).
                        Override the command with --agent-install-cmd; pin a version with
                        --agent-version. Nothing is installed from a private registry: the
                        default command is the public npm package name and nothing else.
  Whatever the CLI turns out to want, --agent-cmd replaces the invocation wholesale;
  $FAI_MODEL / $FAI_PROMPT / $FAI_WORKDIR are exported for it.

ANCHOR (performance tasks)
  reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0. ref_speedup is resolved by the verifier from
  tests/ref_speedup.txt FIRST, then from the in-image manifest, then 1.0 -- and ref_speedup <= 1
  is a hard gate, so a missing anchor means a loud zero, not a wrong score. The runner mounts the
  whole tests/ directory, which is what keeps the anchor resolvable; if you point --tests-dir
  somewhere else, keep ref_speedup.txt in it. The anchor is calibrated hardware (GPU tasks: H20;
  CPU tasks: the authoring CPU lane) -- on different hardware, recalibrate, do not reinterpret.

BUILD
  BuildKit is mandatory: the Dockerfiles use `RUN <cmd> <<'PY' ... PY` heredocs that the classic
  builder cannot parse. The runner therefore always calls `docker buildx build --load`.
  The build context is the PACKAGE ROOT (where task.toml and .dockerignore live), never
  environment/. Package roots differ per subset (kfc/<dir>/task, e2e/<dir>/task, lh/<dir>) and
  are read from tasks_index.json's package_root field, not guessed.

GRADING (verifier mode = shared)
  --verify-mode commit   docker commit the agent container, then grade in a FRESH
                         --network=none container from that snapshot. tests/ is bind-mounted
                         read-only, so the grading surface can never be edited and no
                         __pycache__ is written back into the package.
  --verify-mode exec     docker cp tests/ into the live agent container and docker exec the
                         verifier there. Cheaper (no commit), but a docker exec inherits the
                         container's network namespace, so it is only allowed when the agent
                         step was offline.
  --verify-mode auto     (default) exec when the agent step was offline, commit otherwise.
  The verifier's OWN mode (candidate / noop / oracle / negative / ...) is a different axis:
  --verify-task-mode. It is unset by default, so every task uses its own default (candidate).
  78 tasks read KERNELBENCH_VERIFY_MODE, 5 read VERIFIER_MODE, one each E5_MODE / B11_MODE,
  and kfc/wre-verl-grpo-advantage-loop16 takes a POSITIONAL argument
  (`MODE="${1:-${WRE_MODE:-candidate}}"`) where an env var alone does nothing -- the runner
  detects which and delivers both when needed (--verify-mode-var / --verify-argv override).
  --live-anchor is OFF by default: 2 tasks will re-measure the anchor from the reference patch
  if handed KERNELBENCH_ORACLE_PATCH, which takes the score off the published anchor's scale.

SUBMISSION CAPTURE
  Stage (3) just runs the package's own pre_artifacts.sh -- the capture roots are baked into it
  per task and are never re-derived here. 5 shapes exist: /app/repo only (76); repo + a non-git
  /app/submission (7); /app/submission ALONE, no /app/repo at all (4); two independent
  single-commit repos (2); repo + a non-git /app/workspace (1). Multi-root patches separate the
  segments with `# ==== pre_artifacts root=<dir> mode=<...> ====`. Output is always
  /logs/artifacts/model.patch plus /logs/artifacts/model_files.txt (absolute paths).

FIRST-RUN SELF-CHECK SET (one task per capture shape; --agent none is the cheapest probe)
  python3 scripts/run_task.py --agent none \\
     --task kfc/arch-tiling-dram-traffic-min      `# /app/repo only, CPU, loop16` \\
     --task e2e/a3-moe-train-budget               `# repo + non-git /app/submission` \\
     --task e2e/checkpoint-transfer-integrity     `# /app/submission only, no /app/repo` \\
     --task e2e/object-store-atomic               `# two independent single-commit repos` \\
     --task kfc/wre-verl-grpo-advantage-loop16    `# repo + non-git /app/workspace, argv mode`
  Then repeat with --agent oracle: every task should score its reference value (~0.5 for a
  performance task that exactly hits its anchor, 1.0 for a binary implementation task), and
  --agent oracle --oracle-variant negative MUST score 0. Two tasks
  (kfc/wro-offload-layer-prefetch-ring-pipeline-loop16, kfc/wro-offload-policy-grid-search-loop16)
  ship no reference implementation: their solve.sh exits 2 and the runner records status=skipped.

OUTPUT
  runs/<task>-<agent>-<model>-<ts>/{build.log,agent.log,artifacts.log,verify.log,
                                    prompt.txt,stage/,logs/,artifacts/,verifier/,run.json}
  runs/summary.jsonl   one line per task
  /logs is bind-mounted from runs/<id>/logs at mode 0777 on purpose: the container is root and
  the host user has to be able to read the artifacts back out of that mount.
"""


# ------------------------------------------------------------------------------- small helpers
def _join(argv):
    return getattr(shlex, "join", lambda a: " ".join(shlex.quote(x) for x in a))(argv)


def now_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def D(v):
    return v if isinstance(v, dict) else {}


def firstv(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def read_text(path, default=""):
    try:
        with open(path, errors="ignore") as fh:
            return fh.read()
    except OSError:
        return default


def load_toml(path):
    try:
        if toml_r is not None:
            with open(path, "rb") as fh:
                return toml_r.load(fh)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return _MiniToml.loads(fh.read())            # stdlib-only fallback (py<3.11, no tomli)
    except Exception as exc:                                     # noqa: BLE001
        sys.stderr.write(f"[warn] cannot parse {path}: {exc}\n")
        return {}


def redact_url(url):
    """Strip any userinfo from a URL so it is safe to write into a log or run.json."""
    if not url:
        return url
    return re.sub(r'(?<=//)[^/@]*@', '<redacted>@', url)


def slug(text, limit=60):
    out = re.sub(r'[^A-Za-z0-9._-]+', '-', str(text or "")).strip("-")
    return out[:limit] or "x"


def mkdir777(path):
    """Create a directory the container (root) writes into and the host user reads back."""
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass


# ---------------------------------------------------------------------------------- executor
class Exec:
    """Runs (or, with --dry-run, only prints) every external command."""

    def __init__(self, dry_run=False, quiet=False):
        self.dry_run = dry_run
        self.quiet = quiet

    def echo(self, stage, text):
        if not self.quiet:
            print(f"[{stage}] {text}", flush=True)

    def show_file(self, stage, path, body):
        self.echo(stage, f"--- generated file: {path} ---")
        if not self.quiet:
            for line in body.rstrip("\n").split("\n"):
                print(f"    | {line}", flush=True)
            print(f"    +--- end {path}", flush=True)

    def run(self, argv, *, stage, cwd=None, timeout=None, log_path=None, env=None,
            capture=False, note=None):
        """Returns (rc, stdout_text). In dry-run nothing executes; rc is 0 and stdout is ""."""
        line = _join(argv)
        suffix = []
        if cwd:
            suffix.append(f"cwd={cwd}")
        if timeout:
            suffix.append(f"timeout={int(timeout)}s")
        if log_path:
            suffix.append(f"log={os.path.basename(log_path)}")
        if note:
            suffix.append(note)
        self.echo(stage, "$ " + line + (("    # " + "  ".join(suffix)) if suffix else ""))
        if self.dry_run:
            return 0, ""
        started = time.time()
        try:
            if capture:
                proc = subprocess.run(argv, cwd=cwd, timeout=timeout, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out = proc.stdout or ""
                if log_path:
                    with open(log_path, "a") as fh:
                        fh.write(f"\n$ {line}\n{out}{proc.stderr or ''}")
                return proc.returncode, out
            if log_path:
                with open(log_path, "a") as fh:
                    fh.write(f"\n$ {line}\n")
                    fh.flush()
                    proc = subprocess.run(argv, cwd=cwd, timeout=timeout, env=env,
                                          stdout=fh, stderr=subprocess.STDOUT)
                return proc.returncode, ""
            proc = subprocess.run(argv, cwd=cwd, timeout=timeout, env=env)
            return proc.returncode, ""
        except subprocess.TimeoutExpired:
            self.echo(stage, f"TIMEOUT after {int(time.time() - started)}s: {line}")
            return 124, ""
        except FileNotFoundError as exc:
            self.echo(stage, f"MISSING TOOL: {exc}")
            return 127, ""


# ------------------------------------------------------------------------------------- task
class Task:
    """Everything the runner needs to know about one package, resolved from the package itself."""

    def __init__(self, bench, name, pkg, index_entry=None):
        self.bench, self.name, self.pkg = bench, name, os.path.abspath(pkg)
        self.index = index_entry or {}
        self.warnings = []
        self.meta = load_toml(os.path.join(self.pkg, "task.toml"))
        self.dockerfile_text = ""
        self.test_sh_text = read_text(os.path.join(self.pkg, "tests", "test.sh"))
        self.instruction = read_text(os.path.join(self.pkg, "instruction.md"))
        self._resolve()

    # ---- task.toml is heterogeneous across the corpus (schema v1) and is being unified to
    # ---- v2; read every known location so the runner works before and after that migration.
    def _m(self, *paths, default=None):
        for path in paths:
            node = self.meta
            ok = True
            for key in path.split("."):
                if not isinstance(node, dict) or key not in node:
                    ok = False
                    break
                node = node[key]
            if ok and node is not None:
                return node
        return default

    def _resolve(self):
        self.id = f"{self.bench}/{self.name}"
        self.image = self._m("environment.docker_image", "docker_image",
                             default=f"fai/{self.bench}-{self.name}:oss")
        self.dockerfile_rel, self.dockerfile_single_shot = self._dockerfiles()
        self.dockerfile_text = read_text(os.path.join(self.pkg, self.dockerfile_rel))
        self.cpus = self._m("environment.cpus", "resources.cpus", "cpus")
        mem = self._m("environment.memory_mb", "environment.memory", "resources.memory_mb",
                      "memory", "verifier.environment.memory_mb")
        if mem is None:
            gb = self._m("environment.memory_gb", "resources.memory_gb")
            mem = int(float(gb) * 1024) if gb else None
        self.memory_mb = int(mem) if mem else None
        gpus = self._m("environment.gpus", "resources.gpus", "gpus")
        if gpus is None:
            gpus = self.index.get("gpus")
        self.gpus = int(gpus or 0)
        self.build_timeout = float(self._m("environment.build_timeout_sec",
                                           "verifier.environment.build_timeout_sec",
                                           default=3600))
        self.agent_timeout = float(self._m("agent.timeout_sec", "agent_timeout_sec", default=5400))
        self.verify_timeout = float(self._m("verifier.timeout_sec", "timeout_sec", default=3600))
        self.primary_edit_paths = list(self._m("verifier.entry.primary_edit_paths",
                                               "verifier.primary_edit_paths",
                                               "primary_edit_paths", default=[]) or [])
        if not self.primary_edit_paths:
            self.primary_edit_paths = list(self.index.get("primary_edit_paths") or [])
        self.forbidden_edit_paths = list(self._m("verifier.entry.forbidden_edit_paths",
                                                 "verifier.forbidden_edit_paths", default=[]) or [])
        self.task_kind = self._m("verifier.entry.task_kind", "task.kind", "metadata.task_kind",
                                 "verifier.task_kind", default=self.index.get("task_kind"))
        self.metric_name = self._m("primary_metric.name", "primary_metric.primary_metric_name",
                                   default=None)
        # the verifier entry point as declared (all 90 declare tests/test.sh today); tests/ is
        # mounted at /tests, so the in-container path is /tests/<rest>
        entry = self._m("verifier.entrypoint", "verifier.entry.entrypoint", default="tests/test.sh")
        entry = entry if isinstance(entry, str) else "tests/test.sh"
        rest = entry.split("tests/", 1)[1] if entry.startswith("tests/") else os.path.basename(entry)
        self.verify_entry = f"{TESTS_MNT}/{rest}"
        self.repo_dir = self._guess_repo_dir()
        self.submission_dir = self._guess_submission_dir()
        self.workdir = self._guess_workdir()
        self.loop_script = os.path.join(self.pkg, "environment", "loop", "submit.sh")
        self.has_loop = os.path.isfile(self.loop_script)
        self.loop_limits = self._loop_limits()
        self.anchor, self.anchor_source = self._anchor()
        lds = re.findall(r'LD_LIBRARY_PATH=([^\s\\"\']+)', self.dockerfile_text)
        self.image_ld_static = lds[-1] if lds else ""
        self.entrypoint_honors_argv = self._entrypoint_honors_argv()
        self.pre_artifacts = self._find_pre_artifacts()
        self.solve_sh = self._find_solve_sh()
        self.capture_roots = self._capture_roots()
        self.mode_dispatch = self._mode_dispatch()
        # the graded working tree: the package's own capture roots win over any inference,
        # because 4 tasks have no /app/repo at all and 3 more capture two roots.
        git_roots = [r["path"] for r in self.capture_roots if r["kind"] == "git"]
        if git_roots and self.repo_dir not in git_roots:
            self.warnings.append(f"repo_dir_from_pre_artifacts: capture roots are "
                                 f"{[r['path'] for r in self.capture_roots]}; using "
                                 f"{git_roots[0]} instead of the inferred {self.repo_dir}")
            self.repo_dir = git_roots[0]

    def _dockerfiles(self):
        """Which recipe produces `docker_image`. One package (e2e/a4-token-efficiency-budget)
        ships two Dockerfiles and its docker_image is the loop16 overlay tag, so the recipe is
        read from task.toml `[environment] dockerfile` when declared; the loop16 tag heuristic
        and --dockerfile cover the window before that field lands."""
        def norm(v):
            if not isinstance(v, str) or not v:
                return None
            return v if "/" in v else os.path.join("environment", v)

        declared = norm(self._m("environment.dockerfile", "dockerfile"))
        single = norm(self._m("environment.dockerfile_single_shot", "dockerfile_single_shot"))
        default = os.path.join("environment", "Dockerfile")
        found = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(self.pkg, "environment", "Dockerfile*")))
        if declared and os.path.isfile(os.path.join(self.pkg, declared)):
            return declared, single
        if declared:
            self.warnings.append(f"declared_dockerfile_missing: task.toml points at {declared} "
                                 f"which is not in the package (have: {found}); using {default}")
            return default, single
        if len(found) > 1:
            image = str(self._m("environment.docker_image", "docker_image", default=""))
            for name in found:                       # e.g. Dockerfile.loop16 for an ...:oss-loop16 tag
                suffix = name.split(".", 1)[1] if "." in name else ""
                if suffix and suffix in image:
                    self.warnings.append(
                        f"dockerfile_inferred: package ships {found} and task.toml declares no "
                        f"[environment] dockerfile; docker_image {image!r} matches {name}, so that "
                        "recipe is used (override with --dockerfile)")
                    return os.path.join("environment", name), single or default
            self.warnings.append(f"multiple_dockerfiles: {found}; task.toml declares no "
                                 f"[environment] dockerfile, so {default} is used "
                                 "(override with --dockerfile)")
        return default, single

    def _mode_dispatch(self):
        """How this task's verifier picks its mode. 78 read KERNELBENCH_VERIFY_MODE, 5 read
        VERIFIER_MODE, one each E5_MODE / B11_MODE, and kfc/wre-verl-grpo-advantage-loop16 takes
        a POSITIONAL argument (`MODE="${1:-${WRE_MODE:-candidate}}"`) where an env var alone does
        nothing. 4 tasks deliberately strip inherited mode vars (frozen eval surface)."""
        m = re.search(r'^\s*(?:export\s+)?(?:MODE|VERIFIER_MODE|VMODE)=(.+)$',
                      self.test_sh_text, re.M)
        raw = m.group(1) if m else ""
        names = [n for n in re.findall(r'\$\{([A-Z][A-Z0-9_]*)(?::-|\})', raw)]
        return {"env_vars": names, "positional": "${1" in raw,
                "detected": bool(names) or "${1" in raw}

    def _capture_roots(self):
        """The submission capture roots, taken from the package's own pre_artifacts.sh (which B
        derived per task with evidence). 5 shapes exist corpus-wide: /app/repo only; repo +
        a non-git /app/submission; /app/submission ALONE (no /app/repo at all); two independent
        single-commit repos; repo + a non-git /app/workspace. Never re-derived here."""
        roots = []
        if self.pre_artifacts:
            body = read_text(self.pre_artifacts)
            for fn, kind in (("capture_git", "git"), ("capture_tree", "tree")):
                for m in re.finditer(rf'^\s*{fn}\s+"([^"]+)"', body, re.M):
                    path = m.group(1)
                    if path.startswith("/"):            # skip the internal "$root" fallback call
                        roots.append({"path": path, "kind": kind})
        return roots


    def _guess_repo_dir(self):
        val = self._m("verifier.entry.candidate_source_root", "verifier.candidate_source_root")
        if isinstance(val, str) and val.startswith("/"):
            return val
        m = re.search(r'\bREPO_DIR=([^\s\\"\']+)', self.dockerfile_text)
        if m:
            return m.group(1)
        m = re.search(r'^\s*(?:REPO|REPO_DIR)=["\']?(/[^\s"\'{}]+)', self.test_sh_text, re.M)
        if m:
            return m.group(1)
        for p in self.primary_edit_paths:
            if isinstance(p, str) and p.startswith("/app/repo/"):
                return "/app/repo"
        return "/app/repo"

    def _guess_submission_dir(self):
        for pat in (r'\bSUBMISSION_DIR=["\']?(/[^\s\\"\':]+)',
                    r'^\s*(?:SUB|SUBMISSION_DIR)=["\']?\$\{SUBMISSION_DIR:-(/[^}\s"\']+)\}',
                    r'^\s*(?:SUB|SUBMISSION_DIR)=["\']?(/[^\s"\'{}]+)'):
            for text in (self.dockerfile_text, self.test_sh_text):
                m = re.search(pat, text, re.M)
                if m:
                    return m.group(1)
        return None

    def _guess_workdir(self):
        wds = re.findall(r'^WORKDIR\s+(\S+)', self.dockerfile_text, re.M)
        return wds[-1] if wds else self.repo_dir

    def _loop_limits(self):
        """The baked submit.sh is the ground truth: task.toml's [loop] values disagree with it
        on 73/90 packages, so the script wins and the disagreement is reported."""
        toml_min = self._m("loop.min_submissions", "min_submissions")
        toml_max = self._m("loop.max_submissions", "max_submissions")
        baked_min = baked_max = None
        if self.has_loop:
            body = read_text(self.loop_script)
            m = re.search(r'^MIN_SUBMISSIONS=(\d+)', body, re.M)
            baked_min = int(m.group(1)) if m else None
            m = re.search(r'^MAX_SUBMISSIONS=(\d+)', body, re.M)
            baked_max = int(m.group(1)) if m else None
            if (toml_min, toml_max) != (baked_min, baked_max) and (toml_min or toml_max):
                self.warnings.append(
                    f"loop_limits_mismatch: task.toml says {toml_min}/{toml_max}, baked "
                    f"submit.sh enforces {baked_min}/{baked_max} (the baked script governs)")
        return dict(min=firstv(baked_min, toml_min), max=firstv(baked_max, toml_max),
                    submit="bash /opt/loop/submit.sh" if self.has_loop else None)

    def _anchor(self):
        """Mirror the verifier's own resolution chain: tests/ref_speedup.txt first (that is what
        makes a recalibration possible without a rebuild), then a manifest, then a value hardcoded
        in test.sh. Anything else means the verifier falls back to 1.0 and hard-fails."""
        p = os.path.join(self.pkg, "tests", "ref_speedup.txt")
        if os.path.isfile(p):
            digits = re.sub(r'[^0-9.]', '', read_text(p))
            try:
                return float(digits), "tests/ref_speedup.txt"
            except ValueError:
                pass
        for rel in ("tests/verifier-correctness-manifest.json",
                    "environment/verifier-correctness-manifest.json",
                    "tests/reward_manifest.json"):
            q = os.path.join(self.pkg, rel)
            if os.path.isfile(q):
                m = re.search(r'"ref_speedup"\s*:\s*(?:\{[^}]*"value"\s*:\s*)?([0-9.]+)',
                              read_text(q))
                if m and float(m.group(1)) > 1:
                    return float(m.group(1)), rel
        m = re.search(r'ref_speedup=([0-9]+\.[0-9]+)', self.test_sh_text)
        if m and float(m.group(1)) > 1:
            return float(m.group(1)), "hardcoded in tests/test.sh"
        return None, None

    def _entrypoint_honors_argv(self):
        """3 packages ship an entrypoint that ends in `exec /bin/bash -l` without honouring
        "$@" -- a `docker run <img> <cmd>` there never runs <cmd>. The script lives in different
        places per subset (environment/runtime/, environment/, environment/workspace/) or is
        written inline by the Dockerfile."""
        bodies = [read_text(p) for p in
                  glob.glob(os.path.join(self.pkg, "environment", "Dockerfile*"))]
        for p in glob.glob(os.path.join(self.pkg, "environment", "**", "entrypoint.sh"),
                           recursive=True):
            if f"{os.sep}repo{os.sep}" not in p:
                bodies.append(read_text(p))
        return any('exec "$@"' in b for b in bodies)

    def _find_pre_artifacts(self):
        for rel in ("pre_artifacts.sh", os.path.join("solution", "pre_artifacts.sh")):
            p = os.path.join(self.pkg, rel)
            if os.path.isfile(p):
                return p
        return None

    def _find_solve_sh(self):
        for rel in (os.path.join("solution", "solve.sh"), "solve.sh"):
            p = os.path.join(self.pkg, rel)
            if os.path.isfile(p):
                return p
        return None

    def oracle_artifact(self, variant):
        """Where this package keeps its reference / negative / baseline2 landing material."""
        names = {"oracle": ["oracle"], "negative": ["negative"], "baseline2": ["baseline2"]}[variant]
        cands = []
        for n in names:
            cands += [f"solution/{n}.patch", f"{n}.patch", f"tests/{n}.patch"]
        for a in (self.index.get("oracle_artifacts") or []):
            if variant == "oracle":
                cands.append(a)
        for rel in cands:
            p = os.path.join(self.pkg, rel)
            if os.path.exists(p):
                return rel, ("patch" if rel.endswith(".patch") else
                             ("dir" if os.path.isdir(p) else "file"))
        for rel in sorted(glob.glob(os.path.join(self.pkg, "solution", f"*{names[0]}*"))):
            r = os.path.relpath(rel, self.pkg)
            return r, ("patch" if r.endswith(".patch") else
                       ("dir" if os.path.isdir(rel) else "file"))
        return None, None


# ------------------------------------------------------------------------------ task discovery
def pkg_root_of(path):
    """kfc/<dir>/task, e2e/<dir>/task, lh/<dir> -- decided by where tests/ actually is."""
    if os.path.isdir(os.path.join(path, "tests")):
        return path
    if os.path.isdir(os.path.join(path, "task", "tests")):
        return os.path.join(path, "task")
    return None


def load_index(index_path):
    if not index_path or not os.path.isfile(index_path):
        return {}
    try:
        with open(index_path) as fh:
            data = json.load(fh)
        return {(e["bench"], e["task"]): e for e in data.get("tasks", [])}
    except Exception as exc:                                        # noqa: BLE001
        sys.stderr.write(f"[warn] cannot read index {index_path}: {exc}\n")
        return {}


def discover(args):
    index = load_index(args.index)
    tasks_root = os.path.abspath(args.tasks_root)
    out = []
    if args.task:
        for spec in args.task:
            cands = [spec, os.path.join(tasks_root, spec), os.path.join(ROOT, spec)]
            hit = None
            for c in cands:
                c = os.path.abspath(c)
                root = pkg_root_of(c) if os.path.isdir(c) else None
                if root:
                    hit = root
                    break
            if not hit:
                # bench/name or bare name
                for (bench, name), _e in sorted(index.items()):
                    if spec in (name, f"{bench}/{name}"):
                        hit = pkg_root_of(os.path.join(ROOT, _e["package_root"]))
                        break
            if not hit:
                sys.exit(f"error: cannot resolve --task {spec!r} to a package root "
                         f"(looked for tests/ under it and under <task>/task/)")
            parts = os.path.normpath(hit).split(os.sep)
            name = parts[-2] if parts[-1] == "task" else parts[-1]
            bench = parts[-3] if parts[-1] == "task" else parts[-2]
            out.append(Task(bench, name, hit, index.get((bench, name))))
        return out

    entries = sorted(index.items())
    if entries:
        pool = [(b, n, os.path.join(ROOT, e["package_root"]), e) for (b, n), e in entries]
    else:                                                   # index-less fallback: walk the tree
        pool = []
        for bench in sorted(os.listdir(tasks_root)):
            bdir = os.path.join(tasks_root, bench)
            if not os.path.isdir(bdir):
                continue
            for name in sorted(os.listdir(bdir)):
                root = pkg_root_of(os.path.join(bdir, name))
                if root:
                    pool.append((bench, name, root, None))
    if args.bench:
        pool = [p for p in pool if p[0] in args.bench]
    if args.n_tasks and args.n_tasks < len(pool):
        pool = sorted(random.Random(args.sample_seed).sample(pool, args.n_tasks))
    return [Task(b, n, r, e) for b, n, r, e in pool]


# ---------------------------------------------------------------------------- docker fragments
def resource_flags(task, args):
    flags = []
    cpus = args.cpus or task.cpus
    mem = args.memory_mb or task.memory_mb
    if cpus:
        flags += [f"--cpus={cpus}"]
    if mem:
        flags += [f"--memory={int(mem)}m"]
    shm = args.shm_size or ("2g" if task.gpus else None)
    if shm:
        flags += [f"--shm-size={shm}"]
    return flags


def gpu_plan(task, args):
    """Returns (flags, record). Never guesses silently: the record says what happened."""
    rec = {"requested_gpus": task.gpus, "mode": "none", "devices": [], "notes": []}
    if not task.gpus or args.gpu_passthrough == "none":
        if task.gpus and args.gpu_passthrough == "none":
            rec["notes"].append("task wants a GPU but --gpu-passthrough none was requested")
        return [], rec
    mode = args.gpu_passthrough
    if mode == "auto":
        has_toolkit = bool(shutil.which("nvidia-container-runtime")
                           or shutil.which("nvidia-container-cli"))
        if has_toolkit:
            mode = "toolkit"
        elif os.path.exists("/dev/nvidiactl"):
            mode = "manual"
            rec["notes"].append("no nvidia-container-toolkit found; falling back to manual "
                                "device+driver passthrough")
        else:
            rec["mode"] = "none"
            rec["notes"].append("no nvidia-container-toolkit and no /dev/nvidiactl: the task "
                                "declares a GPU but this host cannot provide one")
            return [], rec
    rec["mode"] = mode
    if mode == "toolkit":
        return ["--gpus", "all"], rec
    # manual: devices + the host's user-space driver, mounted at the SAME path it has on the host
    if args.gpu_devices:
        devs = [d if d.startswith("/dev/") else f"/dev/nvidia{d}"
                for d in re.split(r'[,\s]+', args.gpu_devices) if d]
    else:
        devs = sorted(glob.glob("/dev/nvidia[0-9]*"))
        if not devs:
            devs = ["/dev/nvidia0"]
            rec["notes"].append("no /dev/nvidia<N> visible; assuming /dev/nvidia0 "
                                "(override with --gpu-devices)")
    flags = []
    for d in devs + [c for c in NVIDIA_CONTROL_DEVICES if os.path.exists(c) or args.dry_run]:
        flags += ["--device", d]
    rec["devices"] = devs
    for d in (args.nvidia_lib_dir, args.nvidia_bin_dir):
        if d:
            flags += ["-v", f"{d}:{d}:ro"]
    rec["lib_dir"], rec["bin_dir"] = args.nvidia_lib_dir, args.nvidia_bin_dir
    return flags, rec


def gpu_env_flags(exec_, task, args, grec, image):
    """LD_LIBRARY_PATH for manual passthrough -- but only when the image does not already carry
    the driver directory (43 of the 44 GPU images do, so this is usually a no-op and the image's
    own value, which points at torch's bundled libs, is never clobbered)."""
    if grec.get("mode") != "manual" or not args.nvidia_lib_dir:
        return []
    have = image_env(exec_, image, "LD_LIBRARY_PATH") or task.image_ld_static
    if have and args.nvidia_lib_dir in have.split(":"):
        note = f"driver dir already on the image's LD_LIBRARY_PATH; not overriding it"
        if note not in grec["notes"]:
            grec["notes"].append(note)
        return []
    value = f"{args.nvidia_lib_dir}:{have}" if have else args.nvidia_lib_dir
    return ["-e", f"LD_LIBRARY_PATH={value}"]


def network_plan(task, agent, args):
    """Resolve the agent-step network policy. Grading is always none."""
    spec = AGENTS[agent]
    proxy = args.net_proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    want = args.agent_net
    if want == "auto":
        if not spec["needs_api"]:
            want = "none"
        elif proxy:
            want = "proxy"
        else:
            return None, {"error": (
                f"agent {agent!r} needs egress to a model API but no network policy was chosen. "
                "Pass --agent-net proxy --net-proxy URL (recommended; the proxy enforces the "
                f"allowlist {spec['net_allow']}), or --agent-net host / --agent-net bridge to "
                "accept unrestricted egress. The runner will not escalate on its own.")}
    if want == "proxy" and not proxy:
        return None, {"error": "--agent-net proxy needs --net-proxy URL (or HTTPS_PROXY in env)"}
    rec = {
        "agent_policy": want,
        "docker_network": {"none": "none", "proxy": "bridge", "bridge": "bridge",
                           "host": "host"}[want],
        "declared_allowlist": spec["net_allow"],
        "allowlist_enforced_by": {
            "none": "n/a (offline)",
            "proxy": f"external proxy {redact_url(proxy)}",
            "bridge": "NOTHING -- unrestricted egress in a private namespace",
            "host": "NOTHING -- unrestricted egress AND host-local services reachable",
        }[want],
        "proxy": redact_url(proxy) if want == "proxy" else None,
        "task_allow_internet": False,
        "verify_policy": "none",
        "risk": {
            "none": "none",
            "proxy": "low: only what the proxy allows leaves the container",
            "bridge": "MEDIUM: the agent step can reach anything the host can reach",
            "host": "HIGH: host network namespace shared with the agent step",
        }[want],
    }
    return (want, proxy), rec


def image_env(exec_, image, key):
    """The image's own value for one env var. Static in dry-run (parsed from the Dockerfile by
    the caller), live via docker image inspect otherwise."""
    if exec_.dry_run:
        return ""
    rc, out = exec_.run(["docker", "image", "inspect", "--format",
                         "{{range .Config.Env}}{{println .}}{{end}}", image],
                        stage="inspect", capture=True)
    if rc:
        return ""
    for line in out.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    return ""


# -------------------------------------------------------------------------------- the prompt
def build_prompt(task, agent, budget_sec, args):
    """instruction.md verbatim + a factual run-context block. Never contradicts the statement."""
    lines = [task.instruction.rstrip("\n"), "", "---", "",
             "## Run context (appended by the fai_bench runner; not part of the task statement)",
             ""]
    lines.append(f"- Environment: this container. Working directory: `{task.workdir}`.")
    if len(task.capture_roots) >= 2:
        shown = ", ".join(f"`{r['path']}`"
                          + (" (git working tree)" if r["kind"] == "git" else " (directory)")
                          for r in task.capture_roots)
        lines.append(f"- Your work is graded from: {shown}. Keep every change inside those.")
    else:
        lines.append(f"- Graded working tree: `{task.repo_dir}`"
                     + (f"; submission directory: `{task.submission_dir}`."
                        if task.submission_dir and task.submission_dir != task.repo_dir else "."))
    if task.primary_edit_paths:
        shown = ", ".join(f"`{p}`" for p in task.primary_edit_paths[:8])
        more = "" if len(task.primary_edit_paths) <= 8 else f" (+{len(task.primary_edit_paths) - 8} more)"
        lines.append(f"- Editable scope declared by the task: {shown}{more}. Editing anything "
                     "else can trip the scope gate and zero the score.")
    lines.append("- Leave your work in the WORKING TREE. Do **not** `git commit` and do not move "
                 "`HEAD`: grading diffs the working tree against the baked baseline commit, and a "
                 "moved `HEAD` makes even a correct solution score 0.")
    lines.append(f"- Wall-clock budget for this session: {int(budget_sec)} s. Grading happens "
                 "after the session ends, in this same environment.")
    if task.has_loop and "submit.sh" not in task.instruction:
        lim = task.loop_limits
        cap = f" (up to {lim['max']} submissions" + (f", at least {lim['min']}"
                                                     if lim.get("min") else "") + ")"
        lines.append(f"- In-session self-evaluation: `bash /opt/loop/submit.sh`{cap}. It scores "
                     "your current tree on a public dev proxy and returns feedback. When you are "
                     'done: `bash /opt/loop/submit.sh --finalize --reason "<one sentence>"`; the '
                     "harness also finalizes automatically at the maximum. Your best submission "
                     "is what gets graded.")
    if (not task.has_loop) and "/opt/loop/submit.sh" in task.instruction:
        lines.append("- Correction: `/opt/loop/submit.sh` does **not** exist in this image, so "
                     "ignore the statement's instruction to run it. There is nothing to submit: "
                     "leave your final changes in the working tree and stop when you are done.")
    if args.prompt_extra:
        lines += ["", args.prompt_extra]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- generated scripts
PRE_ARTIFACTS_FALLBACK = r'''#!/bin/bash
# Built-in fallback used when a package ships no pre_artifacts.sh. Captures the agent's work as
# a portable patch of the WORKING TREE against the baseline commit -- no commit is required of
# the agent, and HEAD is never moved (see the spec's H3).
set -uo pipefail
REPO="${REPO_DIR:-/app/repo}"
OUT=/logs/artifacts
mkdir -p "$OUT"
cd "$REPO" 2>/dev/null || { echo "[pre_artifacts] no repo at $REPO"; exit 0; }
git config --global --add safe.directory '*' 2>/dev/null || true
# intent-to-add makes NEW files show up in `git diff` without creating a commit
git add -AN . 2>/dev/null || true
git diff --binary HEAD > "$OUT/model.patch" 2>/dev/null || : > "$OUT/model.patch"
git status --porcelain | awk '{print $NF}' > "$OUT/model_files.txt" 2>/dev/null || : > "$OUT/model_files.txt"
echo "[pre_artifacts] captured $(wc -c < "$OUT/model.patch") bytes, $(wc -l < "$OUT/model_files.txt") paths"
# e2e submission trees can live outside the git repo; tar them for the record (never graded).
SUB="${SUBMISSION_DIR:-}"
if [ -n "$SUB" ] && [ -d "$SUB" ] && [ "$SUB" != "$REPO" ]; then
  tar -czf "$OUT/submission.tar.gz" -C "$(dirname "$SUB")" "$(basename "$SUB")" 2>/dev/null \
    && echo "[pre_artifacts] submission tarball: $(wc -c < "$OUT/submission.tar.gz") bytes"
fi
exit 0
'''


def gen_agent_scripts(task, agent, model, prompt_in_container, budget, args):
    """Returns {filename: body}. agent_cmd.sh holds the invocation verbatim so it is auditable."""
    spec = AGENTS[agent]
    cmd = args.agent_cmd or spec["cmd"]
    files = {}
    head = ["#!/bin/bash",
            "# generated by fai_bench scripts/run_task.py -- agent step (never baked into an image)",
            "set -uo pipefail"]
    path_prepend = []
    if args.agent_path_prepend:
        path_prepend += args.agent_path_prepend
    if args.agent_bin and os.path.isdir(args.agent_bin):
        path_prepend.append(AGENT_BIN_DIR)
    body = list(head)
    if path_prepend:
        body.append(f'export PATH={shlex.quote(":".join(path_prepend))}:$PATH')
    for k, v in (spec["set_env"] or {}).items():
        body.append(f'export {k}={shlex.quote(str(v))}')
    body += [f'export FAI_MODEL={shlex.quote(model or "")}',
             f'export FAI_PROMPT={shlex.quote(prompt_in_container)}',
             f'export FAI_WORKDIR={shlex.quote(args.workdir or task.workdir)}',
             'cd "$FAI_WORKDIR" || { echo "[fai-runner] no workdir $FAI_WORKDIR"; exit 9; }',
             f'echo "[fai-runner] agent={agent} model=$FAI_MODEL workdir=$FAI_WORKDIR '
             f'budget={int(budget)}s"',
             f'CMD={shlex.quote(STAGE_DIR + "/agent_cmd.sh")}',
             'if command -v timeout >/dev/null 2>&1; then',
             f'  timeout --signal=TERM --kill-after=60 {int(budget)} bash "$CMD"',
             'else',
             '  echo "[fai-runner] no coreutils timeout in image; relying on the host-side limit"',
             '  bash "$CMD"',
             'fi',
             'rc=$?',
             'echo "[fai-runner] agent exit rc=$rc"',
             'exit $rc']
    files["agent.sh"] = "\n".join(body) + "\n"
    files["agent_cmd.sh"] = ("#!/bin/bash\n# the agent invocation itself (override with "
                             "--agent-cmd)\nset -uo pipefail\n" + cmd + "\n")
    if args.agent_install:
        inst = args.agent_install_cmd or spec["install"]
        files["agent_install.sh"] = (
            "#!/bin/bash\n# install the agent CLI inside the container (needs agent egress)\n"
            "set -uo pipefail\n"
            f'export FAI_AGENT_VERSION={shlex.quote(args.agent_version or "latest")}\n'
            + inst + "\n")
    return files


def gen_oracle_script(task, args):
    """Land the task's reference material the way an agent would: working tree, no commit."""
    variant = args.oracle_variant
    pkg_in = f"{STAGE_DIR}/pkg"
    lines = ["#!/bin/bash",
             "# generated by fai_bench scripts/run_task.py -- oracle landing (self-check path).",
             "# The reference solution is landed in the WORKING TREE, exactly like an agent's",
             "# edit: no commit, HEAD untouched. Grading then runs in normal candidate mode.",
             "set -uo pipefail",
             f'REPO={shlex.quote(args.repo_dir or task.repo_dir)}',
             f'PKG_DIR={shlex.quote(pkg_in)}',
             f'SOLUTION_DIR={shlex.quote(pkg_in + "/solution")}',
             'export REPO_DIR="$REPO" PKG_DIR SOLUTION_DIR',
             'git config --global --add safe.directory "*" 2>/dev/null || true']
    note = None
    if task.solve_sh:
        rel = os.path.relpath(task.solve_sh, task.pkg)
        flag = {"oracle": "", "negative": " --negative", "baseline2": " --baseline2",
                "noop": " --noop"}[variant]
        lines += [f'SOLVE="$PKG_DIR/{rel}"',
                  'test -f "$SOLVE" || { echo "[oracle] missing $SOLVE"; exit 3; }',
                  f'echo "[oracle] bash $SOLVE{flag}"',
                  f'bash "$SOLVE"{flag}',
                  'rc=$?; echo "[oracle] solve.sh rc=$rc"; exit $rc']
        note = f"solution/solve.sh{flag or ' (oracle)'}"
    elif variant == "noop":
        lines += ['echo "[oracle] --noop: leaving the pristine tree alone"', 'exit 0']
        note = "noop"
    else:
        rel, form = task.oracle_artifact(variant)
        if rel and form == "patch":
            lines += [f'PATCHFILE="$PKG_DIR/{rel}"',
                      'test -f "$PATCHFILE" || { echo "[oracle] missing $PATCHFILE"; exit 3; }',
                      'git -C "$REPO" apply --check -p1 "$PATCHFILE" || '
                      '{ echo "[oracle] patch does not apply cleanly to $REPO"; exit 4; }',
                      'git -C "$REPO" apply -p1 "$PATCHFILE" || exit 4',
                      f'echo "[oracle] landed {rel} in $REPO (working tree; HEAD untouched)"',
                      'git -C "$REPO" status --porcelain', 'exit 0']
            note = f"runner git-apply of {rel} (no solve.sh in this package yet)"
        elif rel and form == "file" and len(task.primary_edit_paths) == 1:
            dest = task.primary_edit_paths[0]
            # keep $REPO expandable: only the literal part gets quoted
            dest_expr = (shlex.quote(dest) if dest.startswith("/")
                         else '"$REPO"/' + shlex.quote(dest))
            lines += [f'SRC="$PKG_DIR/{rel}"',
                      f'DEST={dest_expr}',
                      'test -f "$SRC" || { echo "[oracle] missing $SRC"; exit 3; }',
                      'test -f "$DEST" || { echo "[oracle] scope file $DEST absent"; exit 4; }',
                      'cp "$SRC" "$DEST" || exit 4',
                      f'echo "[oracle] copied {rel} over $DEST (working tree; HEAD untouched)"',
                      'git -C "$REPO" status --porcelain', 'exit 0']
            note = f"runner file-copy of {rel} -> {dest} (no solve.sh in this package yet)"
        else:
            lines += [f'echo "[oracle] this package\'s {variant} material is {rel or "absent"} '
                      f'({form or "n/a"}); the runner cannot land it without solution/solve.sh."',
                      'echo "[oracle] use the verifier-native path instead, e.g."',
                      'echo "  --verify-env KERNELBENCH_VERIFY_MODE=oracle '
                      '--verify-env KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch '
                      '--verify-mount <pkg>/solution:/patches:ro"',
                      'exit 2']
            note = f"UNSUPPORTED without solve.sh (material={rel or 'absent'} form={form})"
    return "\n".join(lines) + "\n", note


def verify_preamble():
    """Reproduce the only side effect of the image entrypoint (the harness clock daemon) without
    going through a login shell that would source a solver-owned ~/.bashrc."""
    return ("mkdir -p /app/.timer /logs/verifier /logs/artifacts /logs/agent 2>/dev/null; "
            "if [ -x /app/timer.sh ]; then export FRONTIER_TIMER_BOOTSTRAP=1; "
            "( /app/timer.sh >/tmp/fai-timer.log 2>&1 & echo $! > /app/.timer/timer.pid ) "
            "2>/dev/null; fi; ")


# ------------------------------------------------------------------------------ result readers
def read_reward(vdir):
    out = {"reward": None, "source": None, "speedup": None, "ref_speedup": None,
           "hard_fail_reasons": []}
    for name in ("reward.json", "metrics.json"):
        p = os.path.join(vdir, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:                                            # noqa: BLE001
            continue
        if isinstance(d, dict) and "reward" in d and out["reward"] is None:
            try:
                out["reward"] = float(d["reward"])
                out["source"] = name
            except (TypeError, ValueError):
                pass
        if isinstance(d, dict):
            for k in ("speedup", "ref_speedup"):
                if d.get(k) is not None and out[k] is None:
                    out[k] = d[k]
            for r in (d.get("hard_fail_reasons") or []):
                if r not in out["hard_fail_reasons"]:
                    out["hard_fail_reasons"].append(r)
    if out["reward"] is None:
        p = os.path.join(vdir, "reward.txt")
        if os.path.isfile(p):
            try:
                out["reward"] = float(re.sub(r'[^0-9eE.+-]', '', read_text(p).strip()))
                out["source"] = "reward.txt"
            except ValueError:
                pass
    p = os.path.join(vdir, "verifier_state.json")
    if os.path.isfile(p):
        try:
            with open(p) as fh:
                st = json.load(fh)
            for r in (st.get("hard_fail_reasons") or []):
                if r not in out["hard_fail_reasons"]:
                    out["hard_fail_reasons"].append(r)
        except Exception:                                            # noqa: BLE001
            pass
    return out


def list_files(root):
    try:
        return sorted(os.path.relpath(os.path.join(dp, f), root)
                      for dp, _dn, fn in os.walk(root) for f in fn)
    except OSError:
        return []


def copy_tree(src, dst, warnings, label):
    if not os.path.isdir(src):
        return []
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as exc:                                         # noqa: BLE001
        warnings.append(f"collect_{label}_copy_failed: {exc}")
    return list_files(dst)


# ------------------------------------------------------------------------------------ one task
def run_one(task, args, exec_, summary_path):
    started = time.time()
    ts = now_ts()
    agent = args.agent
    if agent == "auto":
        agent = "codex" if GPT_FAMILY.match((args.model or "").split("/")[-1]) else "claude-code"
    elif agent in ("claude-code", "codex") and args.model:
        want = "codex" if GPT_FAMILY.match(args.model.split("/")[-1]) else "claude-code"
        if want != agent:
            task.warnings.append(f"model {args.model!r} usually runs under {want!r} but "
                                 f"--agent {agent!r} was requested explicitly")
    spec = AGENTS[agent]
    model = args.model or ""
    run_id = args.run_name or f"{slug(task.name)}-{agent}-{slug(model or 'na', 32)}-{ts}"
    run_dir = os.path.join(os.path.abspath(args.runs_dir), run_id)
    logs_host = os.path.join(run_dir, "logs")
    stage_host = os.path.join(run_dir, "stage")
    image = args.image or task.image
    dockerfile = args.dockerfile or task.dockerfile_rel
    if args.single_shot and not args.dockerfile:
        if task.dockerfile_single_shot:
            dockerfile = task.dockerfile_single_shot
            if not args.image:
                image = image + "-single-shot"
        else:
            task.warnings.append("single_shot_unavailable: task.toml declares no "
                                 "[environment] dockerfile_single_shot; using " + dockerfile)
    tests_dir = os.path.abspath(args.tests_dir) if args.tests_dir else os.path.join(task.pkg, "tests")
    container = f"fai-{slug(task.bench, 8)}-{slug(task.name, 40)}-{ts}"
    cand_image = re.sub(r':[^:/]*$', '', image) + f":cand-{ts}"

    warnings = list(task.warnings)
    rec = {
        "schema": "fai_bench_run_v1", "run_id": run_id, "started_at": ts,
        "runner": os.path.relpath(os.path.abspath(__file__), ROOT),
        "task": {
            "bench": task.bench, "name": task.name, "id": task.id,
            "package_root": os.path.relpath(task.pkg, ROOT), "layout":
                "task/" if os.path.basename(task.pkg) == "task" else "flat",
            "image": image, "dockerfile": dockerfile, "task_kind": task.task_kind,
            "metric": task.metric_name, "gpus": task.gpus, "cpus": args.cpus or task.cpus,
            "memory_mb": args.memory_mb or task.memory_mb,
            "repo_dir": args.repo_dir or task.repo_dir, "submission_dir": task.submission_dir,
            "capture_roots": task.capture_roots,
            "workdir": args.workdir or task.workdir,
            "primary_edit_paths": task.primary_edit_paths,
            "anchor": {"ref_speedup": task.anchor, "source": task.anchor_source},
            "loop": task.loop_limits if task.has_loop else None,
            "timeouts": {"build_sec": args.build_timeout_sec or task.build_timeout,
                         "agent_sec": args.agent_timeout_sec or task.agent_timeout,
                         "verify_sec": args.verify_timeout_sec or task.verify_timeout},
        },
        "agent": {"kind": agent, "model": model, "cli_source": None, "timed_out": False,
                  "exit_code": None, "oracle_variant": args.oracle_variant if agent == "oracle"
                  else None, "oracle_landing": None},
        "stages": {}, "warnings": warnings, "hard_fail_reasons": [],
        "reward": None, "reward_source": None, "build_ok": None, "ok": False, "status": "pending",
    }

    # ---- preflight ---------------------------------------------------------------------
    if "ref_speedup" in task.test_sh_text and task.anchor is None:
        warnings.append("anchor_unresolvable: tests/ref_speedup.txt is absent and no manifest "
                        "carries ref_speedup>1; the verifier will fall back to 1.0 and hard-fail")
    if not task.pre_artifacts:
        warnings.append("pre_artifacts_builtin: this package ships no pre_artifacts.sh; the "
                        "runner's built-in working-tree capture is used")
    if (not task.has_loop) and "/opt/loop/submit.sh" in task.instruction:
        warnings.append("loop_interface_promised_but_absent: instruction.md tells the solver to "
                        "run /opt/loop/submit.sh but this package has no environment/loop/, so "
                        "the image has no such file (prompt carries a correction note)")
    if not task.entrypoint_honors_argv:
        warnings.append("entrypoint_ignores_argv: this package's entrypoint ends in "
                        "`exec /bin/bash -l` without honouring \"$@\"; the runner keeps the "
                        "container alive with an open stdin and overrides the entrypoint when it "
                        "grades in a fresh container")
    if spec["needs_api"]:
        have = [k for k in spec["env_keys"] if os.environ.get(k)]
        if not have:
            warnings.append(f"no credential env var set for {agent} (looked for "
                            f"{spec['env_keys'][:2]}); the agent will probably fail to authenticate")

    net, netrec = network_plan(task, agent, args)
    rec["network"] = netrec
    if net is None:
        rec["hard_fail_reasons"].append("network_policy_unresolved")
        rec["stages"]["preflight"] = {"ok": False, "detail": netrec["error"]}
        exec_.echo("plan", "ERROR " + netrec["error"])
        finish(rec, run_dir, summary_path, started, exec_, args)
        return rec
    agent_net, proxy_url = net
    gflags, grec = gpu_plan(task, args)
    rec["gpu"] = grec
    for note in grec.get("notes", []):
        warnings.append(f"gpu: {note}")

    verify_mode = args.verify_mode
    if verify_mode == "auto":
        verify_mode = "exec" if (agent != "none" and agent_net == "none") else "commit"
    if verify_mode == "exec" and agent_net != "none" and not args.allow_networked_verify:
        verify_mode = "commit"
        warnings.append("verify_mode_forced_commit: a docker exec inherits the container's "
                        "network namespace, and grading must be offline")
    if agent == "none":
        verify_mode = "image"
    # how this task's verifier takes its mode: env var, positional argument, or both
    md = task.mode_dispatch
    vmode_env, vmode_argv = [], list(args.verify_argv)
    if args.verify_task_mode:
        names = ([args.verify_mode_var] if args.verify_mode_var
                 else md["env_vars"] or ["KERNELBENCH_VERIFY_MODE"])
        vmode_env = [f"{n}={args.verify_task_mode}" for n in names]
        if md["positional"] and not vmode_argv:
            vmode_argv = [args.verify_task_mode]
        if not md["detected"]:
            warnings.append(f"verify_mode_dispatch_undetected: tests/test.sh shows no mode "
                            f"selector (some tasks strip inherited mode vars on purpose); "
                            f"passing {vmode_env} anyway -- it may be ignored")
    rec["verify"] = {"mode": verify_mode, "network": "none", "tests_source":
                     os.path.relpath(tests_dir, ROOT) if tests_dir.startswith(ROOT) else tests_dir,
                     "task_mode": args.verify_task_mode, "mode_dispatch": md,
                     "mode_env": vmode_env, "entry_argv": vmode_argv,
                     "entry": task.verify_entry, "live_anchor": bool(args.live_anchor)}
    if args.live_anchor:
        warnings.append("live_anchor: the verifier is allowed to re-measure the anchor from the "
                        "reference patch in this run; the score is then NOT on the same scale as "
                        "the published ref_speedup")

    # ---- run dir ----------------------------------------------------------------------
    if not exec_.dry_run:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(stage_host, exist_ok=True)
        # 0777: the container runs as root and writes here; the host user must read it back.
        mkdir777(logs_host)
        mkdir777(os.path.join(logs_host, "verifier"))
        mkdir777(os.path.join(logs_host, "artifacts"))
    exec_.echo("plan", f"task={task.id} agent={agent} model={model or '-'} image={image}")
    exec_.echo("plan", f"run dir={run_dir}")
    exec_.echo("plan", f"package root={task.pkg} (build context)")
    exec_.echo("plan", f"repo={rec['task']['repo_dir']} workdir={rec['task']['workdir']} "
                       f"gpus={task.gpus} gpu-mode={grec['mode']} "
                       f"agent-net={agent_net} verify-net=none verify-mode={verify_mode}")
    exec_.echo("plan", f"anchor ref_speedup={task.anchor} from {task.anchor_source}")
    for w in warnings:
        exec_.echo("plan", f"WARNING {w}")

    build_log = os.path.join(run_dir, "build.log")
    agent_log = os.path.join(run_dir, "agent.log")
    art_log = os.path.join(run_dir, "artifacts.log")
    verify_log = os.path.join(run_dir, "verify.log")

    # ---- (1) build --------------------------------------------------------------------
    t0 = time.time()
    build_skipped = False
    if args.build_mode == "never":
        build_skipped = True
        rc = 0
    else:
        if args.build_mode == "if-missing":
            rc_i, _ = exec_.run(["docker", "image", "inspect", image], stage="build",
                                capture=True, note="present -> skip the build")
            if rc_i == 0 and not exec_.dry_run:
                build_skipped = True
        rc = 0
        if not build_skipped:
            cmd = ["docker", "buildx", "build", "--load", "--progress=plain",
                   "-f", dockerfile, "-t", image]
            for ba in args.build_arg:
                cmd += ["--build-arg", ba]
            if args.build_network:
                cmd += ["--network", args.build_network]
            if args.platform:
                cmd += ["--platform", args.platform]
            if args.pull:
                cmd += ["--pull"]
            cmd += ["."]
            rc, _ = exec_.run(cmd, stage="build", cwd=task.pkg, log_path=build_log,
                              timeout=rec["task"]["timeouts"]["build_sec"],
                              note="BuildKit is required (heredoc RUN); context = package root"
                                   + ("; skipped when the inspect above succeeds"
                                      if args.build_mode == "if-missing" else ""))
    rec["stages"]["build"] = {"ok": rc == 0, "rc": rc, "skipped": build_skipped,
                              "seconds": round(time.time() - t0, 1)}
    rec["build_ok"] = rc == 0
    if rc != 0:
        rec["hard_fail_reasons"].append("build_failed")
        finish(rec, run_dir, summary_path, started, exec_, args)
        return rec

    verify_image = image
    cid = None
    try:
        # ---- (2) agent ---------------------------------------------------------------
        if agent != "none":
            t0 = time.time()
            budget = rec["task"]["timeouts"]["agent_sec"]
            prompt_txt = build_prompt(task, agent, budget, args)
            prompt_in = f"{STAGE_DIR}/prompt.txt"
            files = {}
            if agent == "oracle":
                body, note = gen_oracle_script(task, args)
                files["oracle.sh"] = body
                rec["agent"]["oracle_landing"] = note
            else:
                files.update(gen_agent_scripts(task, agent, model, prompt_in, budget, args))
            files["pre_artifacts.sh"] = (read_text(task.pre_artifacts) if task.pre_artifacts
                                         else PRE_ARTIFACTS_FALLBACK)
            if not exec_.dry_run:
                with open(os.path.join(run_dir, "prompt.txt"), "w") as fh:
                    fh.write(prompt_txt)
                for name, body in files.items():
                    with open(os.path.join(stage_host, name), "w") as fh:
                        fh.write(body)
            else:
                for name, body in files.items():
                    exec_.show_file("agent", f"{STAGE_DIR}/{name}", body)
                exec_.echo("agent", f"prompt.txt = instruction.md + run context "
                                    f"({len(prompt_txt)} chars, written to {run_dir}/prompt.txt)")

            # start the container
            cmd = ["docker", "run", "-d", "-i", "--name", container,
                   f"--network={netrec['docker_network']}"]
            cmd += resource_flags(task, args) + gflags
            cmd += gpu_env_flags(exec_, task, args, grec, image)
            cmd += ["-v", f"{logs_host}:{LOGS_MNT}"]
            cli_source = "already in image / not needed"
            if agent in ("claude-code", "codex"):
                if args.agent_bin:
                    src = os.path.abspath(args.agent_bin)
                    if os.path.isdir(src):
                        cmd += ["-v", f"{src}:{AGENT_BIN_DIR}:ro"]
                        cli_source = f"bind-mount dir {src} -> {AGENT_BIN_DIR} (ro)"
                    else:
                        dest = f"/usr/local/bin/{os.path.basename(src)}"
                        cmd += ["-v", f"{src}:{dest}:ro"]
                        cli_source = f"bind-mount {src} -> {dest} (ro)"
                elif args.agent_install:
                    cli_source = "installed in container (--agent-install)"
                else:
                    cli_source = "MISSING"
            rec["agent"]["cli_source"] = cli_source
            for m in args.agent_extra_mount:
                cmd += ["-v", m]
            # credentials and proxy travel by NAME so no value lands in argv or a log
            env_names = list(spec["env_keys"]) + [e.split("=")[0] for e in args.agent_env
                                                  if "=" not in e]
            child_env = os.environ.copy()
            if agent_net == "proxy":
                for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
                    child_env[k] = proxy_url
                child_env.setdefault("NO_PROXY", args.no_proxy or "localhost,127.0.0.1")
                child_env.setdefault("no_proxy", child_env["NO_PROXY"])
                env_names += ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                              "NO_PROXY", "no_proxy"]
            for name in dict.fromkeys(env_names):
                cmd += ["-e", name]
            for kv in args.agent_env:
                if "=" in kv:
                    cmd += ["-e", kv]
            cmd += [image, "sleep", "infinity"]
            rc, out = exec_.run(cmd, stage="agent", capture=True, env=child_env,
                                note="-i keeps the 3 packages whose entrypoint ignores \"$@\" alive")
            cid = container if rc == 0 else None
            if rc != 0:
                rec["stages"]["agent"] = {"ok": False, "rc": rc, "detail": "container start failed",
                                          "seconds": round(time.time() - t0, 1)}
                rec["hard_fail_reasons"].append("container_start_failed")
                finish(rec, run_dir, summary_path, started, exec_, args)
                return rec
            if not exec_.dry_run:
                time.sleep(2)
                rc_s, state = exec_.run(["docker", "inspect", "-f", "{{.State.Running}}", container],
                                        stage="agent", capture=True)
                if state.strip() != "true":
                    exec_.run(["docker", "logs", "--tail", "40", container], stage="agent",
                              log_path=agent_log)
                    rec["stages"]["agent"] = {"ok": False, "rc": 1,
                                             "detail": "container exited immediately",
                                             "seconds": round(time.time() - t0, 1)}
                    rec["hard_fail_reasons"].append("container_exited_early")
                    finish(rec, run_dir, summary_path, started, exec_, args)
                    return rec

            exec_.run(["docker", "exec", container, "mkdir", "-p", STAGE_DIR], stage="agent")
            exec_.run(["docker", "cp", f"{stage_host}/.", f"{container}:{STAGE_DIR}/"],
                      stage="agent")
            if agent == "oracle":
                # the package's own reference material, staged where solve.sh expects a package
                # tree. Only ever mounted/copied on this path -- never for a model agent.
                exec_.run(["docker", "exec", container, "mkdir", "-p", f"{STAGE_DIR}/pkg"],
                          stage="agent")
                for rel in ("solution", "oracle.patch", "negative.patch", "baseline2.patch",
                            "solve.sh"):
                    src = os.path.join(task.pkg, rel)
                    if os.path.exists(src):
                        exec_.run(["docker", "cp", src, f"{container}:{STAGE_DIR}/pkg/{rel}"],
                                  stage="agent", note="reviewer-only material, oracle path only")
                # Mode-dispatch oracles keep their reference IN tests/ (e.g. tests/oracles/<MODE>.py,
                # tests/scenarios/) rather than in solution/. tests/ is NOT mounted during the agent
                # step, so stage those subdirs into the pkg tree where solve.sh looks for them
                # (SOLVE_ORACLES_DIR / $PKG/tests/oracles). Oracle path only -- a model agent never
                # gets these. It is a copy of reviewer material, and grading still runs from the
                # real read-only tests/ mount, so nothing here can leak into or bias the score.
                for rel in ("tests/oracles", "tests/scenarios"):
                    src = os.path.join(task.pkg, rel)
                    if os.path.isdir(src):
                        exec_.run(["docker", "exec", container, "mkdir", "-p",
                                   f"{STAGE_DIR}/pkg/tests"], stage="agent")
                        exec_.run(["docker", "cp", src, f"{container}:{STAGE_DIR}/pkg/{rel}"],
                                  stage="agent",
                                  note="mode-dispatch reference (oracle self-check only)")
            else:
                exec_.run(["docker", "cp", os.path.join(run_dir, "prompt.txt"),
                           f"{container}:{prompt_in}"], stage="agent")
                if args.agent_install:
                    rc_i, _ = exec_.run(["docker", "exec", container, "bash",
                                         f"{STAGE_DIR}/agent_install.sh"], stage="agent",
                                        log_path=agent_log, timeout=1800)
                    if rc_i != 0:
                        warnings.append(f"agent_install_rc={rc_i}")
                rc_c, _ = exec_.run(["docker", "exec", container, "bash", "-lc",
                                     f"command -v {spec['bin']}"], stage="agent", capture=True)
                if rc_c != 0 and not exec_.dry_run:
                    rec["stages"]["agent"] = {
                        "ok": False, "rc": 127, "seconds": round(time.time() - t0, 1),
                        "detail": (f"{spec['bin']!r} is not on PATH inside the container. No image "
                                   "in this corpus ships an agent CLI: pass --agent-bin <host "
                                   "path> to bind-mount one, or --agent-install to install it in "
                                   "the container (needs agent egress).")}
                    rec["hard_fail_reasons"].append("agent_cli_missing")
                    finish(rec, run_dir, summary_path, started, exec_, args)
                    return rec

            script = "oracle.sh" if agent == "oracle" else "agent.sh"
            ex = ["docker", "exec"]
            if agent == "oracle":
                # solve.sh looks for its reference assets under SOLVE_ASSET_DIR first, then
                # relative to itself; both point at the staged package tree. SOLVE_ORACLES_DIR
                # points a mode-dispatch solve.sh straight at the staged tests/oracles.
                ex += ["-e", f"REPO_DIR={rec['task']['repo_dir']}",
                       "-e", f"SOLVE_ASSET_DIR={STAGE_DIR}/pkg",
                       "-e", f"SOLVE_ORACLES_DIR={STAGE_DIR}/pkg/tests/oracles"]
            ex += [container, "bash", f"{STAGE_DIR}/{script}"]
            rc_a, _ = exec_.run(ex, stage="agent", log_path=agent_log,
                                timeout=(budget + 300) if agent != "oracle" else 1800,
                                note="in-container `timeout` owns the budget; this is slack")
            rec["agent"]["exit_code"] = rc_a
            rec["agent"]["timed_out"] = rc_a in (124, 137)
            if rc_a in (124, 137):
                warnings.append("agent_timed_out: the session hit its wall-clock budget; "
                                "grading continues on whatever is in the working tree")
                # a host-side timeout only kills the docker client, so make sure nothing of ours
                # is still editing the tree while pre_artifacts runs (container-local match only)
                exec_.run(["docker", "exec", container, "bash", "-c",
                           f"pkill -f {STAGE_DIR}/agent 2>/dev/null; "
                           f"pkill -f {shlex.quote(spec['bin'] or 'nothing-to-kill')} "
                           "2>/dev/null; true"], stage="agent", capture=True)
            rec["stages"]["agent"] = {"ok": rc_a == 0 or rc_a in (124, 137), "rc": rc_a,
                                      "seconds": round(time.time() - t0, 1)}
            if agent == "oracle" and rc_a == 2:
                # solve.sh exits 2 when the package has no reference implementation at all
                # (2 tasks corpus-wide). That is a SKIP, not a failure.
                rec["status"] = "skipped"
                rec["skip_reason"] = "oracle_unavailable_for_this_task"
                warnings.append("oracle_unavailable: solution/solve.sh exited 2 -- this package "
                                "ships no reference implementation, so the oracle self-check does "
                                "not apply. Nothing was graded.")
                finish(rec, run_dir, summary_path, started, exec_, args)
                return rec
            if agent == "oracle" and rc_a != 0:
                rec["hard_fail_reasons"].append(f"oracle_landing_failed_rc{rc_a}")

            # ---- (3) artifacts -------------------------------------------------------
            t0 = time.time()
            envs = ["-e", f"REPO_DIR={rec['task']['repo_dir']}"]
            if task.submission_dir:
                envs += ["-e", f"SUBMISSION_DIR={task.submission_dir}"]
            rc_p, _ = exec_.run(["docker", "exec"] + envs +
                                [container, "bash", f"{STAGE_DIR}/pre_artifacts.sh"],
                                stage="artifacts", log_path=art_log, timeout=900,
                                note=("package pre_artifacts.sh" if task.pre_artifacts
                                      else "runner built-in fallback"))
            rec["stages"]["artifacts"] = {"ok": rc_p == 0, "rc": rc_p,
                                          "source": (os.path.relpath(task.pre_artifacts, task.pkg)
                                                     if task.pre_artifacts else "runner-builtin"),
                                          "seconds": round(time.time() - t0, 1)}

        # ---- (4) verify --------------------------------------------------------------
        t0 = time.time()
        vt = rec["task"]["timeouts"]["verify_sec"]
        entry_cmd = " ".join([f"bash {task.verify_entry}"] + [shlex.quote(a) for a in vmode_argv])
        inner = (verify_preamble() +
                 f"exec timeout --signal=TERM --kill-after=60 {int(vt)} {entry_cmd}")
        venv_flags = []
        for kv in vmode_env + args.verify_env:
            venv_flags += ["-e", kv]
        vmounts = list(args.verify_mount)
        if args.live_anchor:
            # 2 tasks re-measure the anchor in candidate mode when handed the reference patch.
            # Off by default so a score stays comparable to the published ref_speedup.
            rel, form = task.oracle_artifact("oracle")
            if rel and form == "patch":
                vmounts.append(f"{os.path.join(task.pkg, os.path.dirname(rel)) or task.pkg}"
                               f":/patches:ro")
                venv_flags += ["-e", f"KERNELBENCH_ORACLE_PATCH=/patches/{os.path.basename(rel)}"]
            else:
                warnings.append(f"live_anchor_unavailable: no oracle patch in this package "
                                f"(found {rel!r} form={form!r})")
        if verify_mode == "exec":
            exec_.run(["docker", "cp", tests_dir, f"{container}:{TESTS_MNT}"], stage="verify",
                      note="tests/ reach the container only AFTER the session ends")
            rc_v, _ = exec_.run(["docker", "exec"] + venv_flags +
                                [container, "bash", "-c",
                                 f"exec timeout --signal=TERM --kill-after=60 {int(vt)} "
                                 f"{entry_cmd}"],
                                stage="verify", log_path=verify_log, timeout=vt + 300,
                                note="offline: the container itself has --network=none")
            rec["verify"]["tests_delivery"] = "docker cp (container is offline)"
            if vmounts:
                warnings.append("verify_mounts_ignored_in_exec_mode: docker exec cannot add "
                                f"mounts ({vmounts}); use --verify-mode commit")
        else:
            if verify_mode == "commit":
                rc_c, _ = exec_.run(["docker", "commit", container, cand_image], stage="verify",
                                    capture=True,
                                    note="snapshot the agent's tree, then grade it offline")
                if rc_c != 0:
                    rec["hard_fail_reasons"].append("commit_failed")
                verify_image = cand_image
                rec["verify"]["candidate_image"] = cand_image
            vcmd = ["docker", "run", "--name", f"{container}-verify", "--network=none"]
            vcmd += resource_flags(task, args) + gflags
            vcmd += gpu_env_flags(exec_, task, args, grec, verify_image)
            vcmd += ["-v", f"{tests_dir}:{TESTS_MNT}:ro", "-v", f"{logs_host}:{LOGS_MNT}"]
            for m in vmounts:
                vcmd += ["-v", m]
            vcmd += venv_flags
            vcmd += ["--entrypoint", "/bin/bash", verify_image, "-c", inner]
            rc_v, _ = exec_.run(vcmd, stage="verify", log_path=verify_log, timeout=vt + 300,
                                note="entrypoint overridden: the timer daemon is started by the "
                                     "preamble, no login shell sources a solver-owned rc file")
            rec["verify"]["tests_delivery"] = f"bind-mount {TESTS_MNT}:ro"
            if not args.keep_container:
                exec_.run(["docker", "rm", "-f", f"{container}-verify"], stage="verify",
                          capture=True)
        rec["stages"]["verify"] = {"ok": rc_v == 0, "rc": rc_v,
                                  "seconds": round(time.time() - t0, 1)}
        if rc_v in (124, 137):
            rec["hard_fail_reasons"].append("verifier_timed_out")

        # ---- (5) collect -------------------------------------------------------------
        t0 = time.time()
        vhost = os.path.join(run_dir, "verifier")
        ahost = os.path.join(run_dir, "artifacts")
        vfiles = afiles = []
        if not exec_.dry_run:
            vfiles = copy_tree(os.path.join(logs_host, "verifier"), vhost, warnings, "verifier")
            afiles = copy_tree(os.path.join(logs_host, "artifacts"), ahost, warnings, "artifacts")
            if not vfiles and cid:                      # bind mount unreadable -> stream it out
                os.makedirs(vhost, exist_ok=True)
                exec_.run(["docker", "cp", f"{container}:{LOGS_MNT}/verifier/.", vhost],
                          stage="collect", capture=True,
                          note="fallback: the bind mount was not readable back")
                vfiles = list_files(vhost)
        else:
            exec_.echo("collect", f"copy {logs_host}/verifier -> {vhost}")
            exec_.echo("collect", f"copy {logs_host}/artifacts -> {ahost}")
        res = read_reward(vhost)
        rec["reward"], rec["reward_source"] = res["reward"], res["source"]
        rec["speedup"], rec["measured_ref_speedup"] = res["speedup"], res["ref_speedup"]
        for r in res["hard_fail_reasons"]:
            if r not in rec["hard_fail_reasons"]:
                rec["hard_fail_reasons"].append(r)
        if rec["reward"] is None and not exec_.dry_run:
            rec["hard_fail_reasons"].append("no_reward_file")
        rec["verifier_files"], rec["artifact_files"] = vfiles, afiles
        rec["stages"]["collect"] = {"ok": bool(vfiles) or exec_.dry_run,
                                    "seconds": round(time.time() - t0, 1)}
    finally:
        if cid and not args.keep_container:
            exec_.run(["docker", "rm", "-f", container], stage="cleanup", capture=True)
        if (verify_image != image) and not args.keep_verify_image:
            exec_.run(["docker", "image", "rm", "-f", verify_image], stage="cleanup", capture=True)

    rec["ok"] = bool(rec["build_ok"]) and rec["reward"] is not None and not any(
        r in rec["hard_fail_reasons"] for r in ("build_failed", "agent_cli_missing",
                                                "container_start_failed", "no_reward_file"))
    if rec.get("status") != "skipped":
        rec["status"] = "ok" if rec["ok"] else "failed"
    finish(rec, run_dir, summary_path, started, exec_, args)
    return rec


def finish(rec, run_dir, summary_path, started, exec_, args):
    rec["finished_at"] = now_ts()
    if rec.get("status") == "pending":
        rec["status"] = "failed" if rec["hard_fail_reasons"] else "incomplete"
    rec["seconds"] = {k: v.get("seconds") for k, v in rec["stages"].items()}
    rec["seconds"]["total"] = round(time.time() - started, 1)
    line = {k: rec.get(k) for k in ("run_id", "reward", "reward_source", "build_ok", "ok",
                                    "status", "hard_fail_reasons", "seconds")}
    line.update({"task": rec["task"]["id"], "bench": rec["task"]["bench"],
                 "agent": rec["agent"]["kind"], "model": rec["agent"]["model"],
                 "image": rec["task"]["image"],
                 "agent_net": rec.get("network", {}).get("agent_policy"),
                 "verify_mode": rec.get("verify", {}).get("mode"),
                 "warnings": len(rec.get("warnings") or []),
                 "run_dir": os.path.relpath(run_dir, ROOT) if run_dir.startswith(ROOT) else run_dir})
    if exec_.dry_run:
        exec_.echo("plan", "dry run: nothing executed, no run.json / summary.jsonl written")
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "a") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    exec_.echo("done", f"reward={rec['reward']} hard_fail={rec['hard_fail_reasons']} "
                       f"-> {os.path.join(run_dir, 'run.json')}")


# ------------------------------------------------------------------------------------- CLI
def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="run_task.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=EPILOG % {"binmnt": AGENT_BIN_DIR})

    g = p.add_argument_group("task selection")
    g.add_argument("--task", action="append", default=[],
                   help="package root, task dir, or <bench>/<name>; repeatable")
    g.add_argument("--tasks-root", default=os.path.join(ROOT, "tasks"))
    g.add_argument("--index", default=os.path.join(ROOT, "tasks_index.json"),
                   help="tasks_index.json; package_root comes from here (the three subsets "
                        "have different layouts, so it is never guessed)")
    g.add_argument("--bench", action="append", default=[], choices=["kfc", "lh", "e2e"])
    g.add_argument("--n-tasks", type=int, help="deterministic random subset size")
    g.add_argument("--sample-seed", type=int, default=0)
    g.add_argument("--list", action="store_true", help="print the resolved selection and exit")

    g = p.add_argument_group("agent")
    g.add_argument("--agent", default="auto",
                   choices=["auto", "claude-code", "codex", "oracle", "none"],
                   help="auto: gpt-family models -> codex, everything else -> claude-code")
    g.add_argument("--model", default="")
    g.add_argument("--agent-bin", help="host path to the CLI (file or dir), bind-mounted read-only")
    g.add_argument("--agent-install", action="store_true",
                   help="install the CLI inside the container instead (needs agent egress)")
    g.add_argument("--agent-install-cmd", help="override the install command")
    g.add_argument("--agent-version", help="version for the default install command")
    g.add_argument("--agent-cmd", help="override the whole invocation; $FAI_MODEL / $FAI_PROMPT / "
                                       "$FAI_WORKDIR are exported")
    g.add_argument("--agent-env", action="append", default=[], metavar="KEY[=VAL]",
                   help="extra env for the agent step; KEY alone forwards the host value by name")
    g.add_argument("--agent-extra-mount", action="append", default=[], metavar="H:C[:ro]")
    g.add_argument("--agent-path-prepend", action="append", default=[], metavar="DIR")
    g.add_argument("--agent-timeout-sec", type=float, help="override [agent] timeout_sec")
    g.add_argument("--oracle-variant", default="oracle",
                   choices=["oracle", "negative", "baseline2", "noop"],
                   help="--agent oracle only: which reference material to land "
                        "(negative must score 0; noop is the no-op control)")
    g.add_argument("--prompt-extra", help="extra text appended to the run-context block")

    g = p.add_argument_group("network (see the epilog for the trade-offs)")
    g.add_argument("--agent-net", default="auto", choices=["auto", "none", "proxy", "bridge", "host"])
    g.add_argument("--net-proxy", help="proxy URL for --agent-net proxy; it enforces the allowlist")
    g.add_argument("--no-proxy", help="NO_PROXY value for the agent step")
    g.add_argument("--allow-networked-verify", action="store_true",
                   help="permit --verify-mode exec in a networked container (NOT recommended)")

    g = p.add_argument_group("build")
    g.add_argument("--build-mode", default="if-missing", choices=["always", "if-missing", "never"])
    g.add_argument("--image", help="override task.toml's docker_image")
    g.add_argument("--dockerfile", help="relative to the package root; default = task.toml "
                                        "[environment] dockerfile, else environment/Dockerfile")
    g.add_argument("--single-shot", action="store_true",
                   help="build task.toml [environment] dockerfile_single_shot instead (the "
                        "non-loop16 recipe, where a package has one); the image tag gets a "
                        "-single-shot suffix so it cannot overwrite the declared image")
    g.add_argument("--build-arg", action="append", default=[], metavar="K=V")
    g.add_argument("--build-network", choices=["default", "host", "none"],
                   help="buildx --network for RUN steps. Leave unset for docker's default "
                        "(works with public egress). Use host when the build must reach a "
                        "PyPI/apt mirror that is only routable on the host network.")
    g.add_argument("--platform")
    g.add_argument("--pull", action="store_true")
    g.add_argument("--build-timeout-sec", type=float)

    g = p.add_argument_group("resources / GPU")
    g.add_argument("--cpus")
    g.add_argument("--memory-mb", type=int)
    g.add_argument("--shm-size", help="default 2g for GPU tasks, docker's default otherwise")
    g.add_argument("--gpu-passthrough", default="auto",
                   choices=["auto", "toolkit", "manual", "none"],
                   help="toolkit: --gpus all. manual: /dev/nvidia* devices plus the host driver "
                        "user-space (for hosts without nvidia-container-toolkit)")
    g.add_argument("--gpu-devices", help="e.g. 0,1 or /dev/nvidia0 (manual mode)")
    g.add_argument("--nvidia-lib-dir", default=DEFAULT_NVIDIA_LIB_DIR,
                   help=f"manual mode driver user-space dir (default {DEFAULT_NVIDIA_LIB_DIR})")
    g.add_argument("--nvidia-bin-dir", default=DEFAULT_NVIDIA_BIN_DIR,
                   help=f"manual mode nvidia-smi dir (default {DEFAULT_NVIDIA_BIN_DIR})")

    g = p.add_argument_group("grading")
    g.add_argument("--verify-mode", default="auto", choices=["auto", "exec", "commit"])
    g.add_argument("--verify-task-mode", metavar="MODE",
                   help="the VERIFIER's own mode (candidate|noop|oracle|negative|...). Left unset "
                        "by default so each task uses its own default (candidate). The runner "
                        "delivers it the way that task accepts it: its detected env var and, for "
                        "the one task that reads a positional argument, as argv too")
    g.add_argument("--verify-mode-var", metavar="NAME",
                   help="force the env var name that carries --verify-task-mode")
    g.add_argument("--verify-argv", action="append", default=[], metavar="ARG",
                   help="extra argv appended to the verifier entry point (repeatable)")
    g.add_argument("--live-anchor", action="store_true",
                   help="allow the verifier to re-measure the anchor from the package's reference "
                        "patch during grading (2 tasks support it). OFF by default: with it on the "
                        "score is no longer on the same scale as the published ref_speedup")
    g.add_argument("--verify-env", action="append", default=[], metavar="K=V",
                   help="e.g. KERNELBENCH_VERIFY_MODE=oracle (low-level escape hatch)")
    g.add_argument("--verify-mount", action="append", default=[], metavar="H:C[:ro]",
                   help="e.g. <pkg>/solution:/patches:ro for the verifier-native oracle path")
    g.add_argument("--verify-timeout-sec", type=float)
    g.add_argument("--tests-dir", help="override the grading surface (keep ref_speedup.txt in it)")
    g.add_argument("--repo-dir", help="override the graded working tree path")
    g.add_argument("--workdir", help="override the agent's working directory")

    g = p.add_argument_group("output / misc")
    g.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    g.add_argument("--run-name", help="fixed run directory name (single task only)")
    g.add_argument("--dry-run", action="store_true",
                   help="print every command that would run, execute nothing, write nothing")
    g.add_argument("--keep-container", action="store_true")
    g.add_argument("--keep-verify-image", action="store_true")
    g.add_argument("--stop-on-error", action="store_true")
    g.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    if args.run_name and len(args.task) > 1:
        p.error("--run-name only makes sense with a single --task")
    return args


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    tasks = discover(args)
    if not tasks:
        sys.exit("error: no tasks selected")
    if args.list:
        for t in tasks:
            print(f"{t.bench}/{t.name}\tgpus={t.gpus}\timage={t.image}\t"
                  f"root={os.path.relpath(t.pkg, ROOT)}")
        print(f"# {len(tasks)} task(s)")
        return 0
    exec_ = Exec(dry_run=args.dry_run, quiet=args.quiet)
    summary_path = os.path.join(os.path.abspath(args.runs_dir), "summary.jsonl")
    results = []
    for i, t in enumerate(tasks, 1):
        print(f"\n===== [{i}/{len(tasks)}] {t.bench}/{t.name} =====", flush=True)
        try:
            rec = run_one(t, args, exec_, summary_path)
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
            return 130
        results.append(rec)
        if args.stop_on_error and not rec.get("ok"):
            break
    if args.dry_run:
        print(f"\ndry run: {len(results)} task(s) planned, nothing executed")
        return 0
    print("\n===== summary =====")
    for r in results:
        print(f"{r['task']['id']:<58} {r.get('status','?'):<9} reward={str(r['reward']):<8} "
              f"build={r['build_ok']} "
              f"hard_fail={','.join(r['hard_fail_reasons']) or '-'}")
    print(f"appended {len(results)} line(s) to {summary_path}")
    failed = [r for r in results if r.get("status") == "failed"
              or (r.get("status") == "incomplete")]
    skipped = [r for r in results if r.get("status") == "skipped"]
    if skipped:
        print(f"skipped {len(skipped)}: " + ", ".join(r["task"]["id"] for r in skipped))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

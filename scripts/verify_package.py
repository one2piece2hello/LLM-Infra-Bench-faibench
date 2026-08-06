#!/usr/bin/env python3
"""fai_bench 题包自检 —— 只读,任何失败退出码非 0。

完全自洽:所有路径从本脚本位置推导,不依赖任何外部台账或源树,
所以在你 clone 下来的任何位置都能直接跑。

  python3 scripts/verify_package.py            # 全量
  python3 scripts/verify_package.py kfc lh     # 只查某些子集

检查项(逐题):
  1  必备件齐全        instruction.md · task.toml · environment/Dockerfile · tests/
  2  task.toml 可解析  且可编辑范围 / 资源声明在
  3  判分面可用        tests/test.sh 存在;性能题的锚点能被解析到(不会静默回落 1.0)
  4  Dockerfile 可解析 复刻 Dockerfile 解析器:heredoc 配对、无悬挂续行、
                       COPY 源都在构建上下文里且没被 .dockerignore 排除,
                       每个 RUN 的 shell 体过 `bash -n`
  5  锚点自洽          tests/ref_speedup.txt(若有)与 caveat 记录的值一致
  6  挖洞树自证        有 oracle.patch 的题:补丁必须能正向打在 environment/repo/ 上、反向打不上
  7  卫生              无 __pycache__ / *.pyc / *.bak* / .DS_Store / 编辑器残留
  8  可运行三件套      pre_artifacts.sh 存在且 `bash -n` 过;solution/solve.sh 存在、可执行、
                       四态 CLI(--negative/--baseline2/--noop)且 `bash -n` 过;
                       task.toml 是 schema_version="2.0"
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import tempfile

try:
    import tomllib as toml_r                                     # py >= 3.11
except ModuleNotFoundError:
    try:
        import tomli as toml_r                                   # type: ignore
    except ModuleNotFoundError:
        toml_r = None                                            # stdlib-only: schema check degrades


def _toml_load(path):
    """Parse task.toml. Prefer tomllib/tomli; fall back to a scan for the handful of
    keys this checker needs so the checker itself stays dependency-free on py<3.11."""
    if toml_r is not None:
        with open(path, "rb") as fh:
            return toml_r.load(fh)
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    meta: dict = {}
    m = re.search(r'^\s*schema_version\s*=\s*["\']([^"\']+)', txt, re.M)
    if m:
        meta["schema_version"] = m.group(1)
    return meta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "tasks")
BENCHES = [b for b in ("kfc", "lh", "e2e") if os.path.isdir(os.path.join(TASKS, b))]
WANT = [a for a in sys.argv[1:] if not a.startswith("-")] or BENCHES

KW = ("FROM RUN CMD LABEL MAINTAINER EXPOSE ENV ADD COPY ENTRYPOINT VOLUME USER "
      "WORKDIR ARG ONBUILD STOPSIGNAL HEALTHCHECK SHELL").split()
KWRE = re.compile(r'^\s*(' + "|".join(KW) + r')(\s|$)', re.I)
HDRE = re.compile(r'<<(-?)\s*(["\']?)([A-Za-z_][A-Za-z0-9_]*)\2')
JUNK = re.compile(r'(^|/)__pycache__(/|$)|\.pyc$|\.pyo$|\.bak(_|\.|$)|\.orig(_|$)'
                  r'|\.DS_Store$|~$|\.swp$|\.pre_')

fails: list[str] = []
warns: list[str] = []


def fail(pkg: str, msg: str) -> None:
    fails.append(f"{pkg}: {msg}")


def warn(pkg: str, msg: str) -> None:
    warns.append(f"{pkg}: {msg}")


# ---------------------------------------------------------------- .dockerignore
def _pat2re(pp: str) -> re.Pattern:
    """Docker 的 .dockerignore 语义:相对上下文根整路径匹配(支持 ** / * / ?)。"""
    out, i = "", 0
    while i < len(pp):
        if pp.startswith("**", i):
            out += ".*"
            i += 2
            if i < len(pp) and pp[i] == "/":
                out += "/?"
                i += 1
        elif pp[i] == "*":
            out += "[^/]*"
            i += 1
        elif pp[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pp[i])
            i += 1
    return re.compile(r'^' + out + r'$')


def dockerignored(rel: str, pats: list[str]) -> bool:
    rel = rel.strip("/")
    hit = False
    for p in pats:
        neg = p.startswith("!")
        rx = _pat2re((p[1:] if neg else p).strip("/"))
        m = bool(rx.match(rel))
        if not m:                                   # 命中父目录 -> 内容也被排除
            parts = rel.split("/")
            m = any(rx.match("/".join(parts[:k])) for k in range(1, len(parts)))
        if m:
            hit = not neg
    return hit


# ---------------------------------------------------------------- Dockerfile
def check_dockerfile(pkg: str, df: str) -> None:
    ctx = os.path.dirname(os.path.dirname(df))      # 上下文 = 题包根
    lines = open(df, errors="ignore").read().split("\n")
    di = os.path.join(ctx, ".dockerignore")
    pats = [l.strip() for l in open(di, errors="ignore")] if os.path.isfile(di) else []
    pats = [p for p in pats if p and not p.startswith("#")]
    rel_df = os.path.relpath(df, ROOT)
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        if not KWRE.match(raw):
            fail(pkg, f"{rel_df}:{i+1} 非指令行出现在指令位置(Dockerfile 解析会失败): "
                      f"«{raw.strip()[:70]}»")
            i += 1
            continue
        instr = KWRE.match(raw).group(1).upper()
        cont = [raw]
        while cont[-1].rstrip().endswith("\\"):
            if i + 1 >= n:
                fail(pkg, f"{rel_df}:{i+1} 文件末尾悬挂 `\\` 续行")
                break
            i += 1
            cont.append(lines[i])
        joined = "\n".join(cont)
        i += 1
        hds: list[tuple[str, list[str]]] = []
        for m in HDRE.finditer(joined):
            term, dash, body, ok = m.group(3), m.group(1) == "-", [], False
            while i < n:
                s = lines[i]
                if (s.lstrip("\t") if dash else s).rstrip("\r") == term:
                    ok = True
                    i += 1
                    break
                body.append(s)
                i += 1
            if not ok:
                fail(pkg, f"{rel_df} heredoc `{term}` 未终止")
            hds.append((term, body))
        if instr == "RUN":
            first = re.sub(r'^\s*RUN\s+', '', joined, flags=re.I)
            if not first.lstrip().startswith("["):          # exec-form 不是 shell
                if hds and re.match(r'^<<-?\s*["\']?[A-Za-z_]', first.strip()):
                    body = "\n".join(hds[0][1])             # RUN <<'SH' 脚本形式
                else:
                    body = re.sub(r'\\\s*\n', ' ', first)   # shell 语义:\+换行 整对删除
                    for term, b in hds:
                        body += "\n" + "\n".join(b) + "\n" + term
                    for k, l in enumerate(joined.split("\n")):
                        if k and re.search(r'(^|\s)#', l) and l.rstrip().endswith("\\"):
                            fail(pkg, f"{rel_df} `\\`-续行的 RUN 里有内联 `#`,会吃掉后续命令")
                            break
                with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
                    f.write(body)
                    tmp = f.name
                r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
                os.unlink(tmp)
                if r.returncode:
                    tail = (r.stderr.strip().split("\n") or [""])[-1][:90]
                    fail(pkg, f"{rel_df} RUN 的 shell 体语法错: {tail}")
        if instr in ("COPY", "ADD") and "--from=" not in joined:
            toks = [t for t in re.sub(r'^\s*(COPY|ADD)\s+', '', joined, flags=re.I)
                    .replace("\\\n", " ").replace("\n", " ").split()
                    if not t.startswith("--")]
            for src in toks[:-1]:
                if re.match(r'^https?://', src):
                    continue
                s = src.rstrip("/")
                if not glob.glob(os.path.join(ctx, s)):
                    fail(pkg, f"{rel_df} COPY 源不在构建上下文里: {src}")
                elif dockerignored(s, pats):
                    fail(pkg, f"{rel_df} COPY 源被 .dockerignore 排除: {src}")


# ---------------------------------------------------------------- runnable three-piece
def _bash_n(path: str) -> str:
    """Return '' if `bash -n <path>` is clean, else the last stderr line."""
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    if r.returncode == 0:
        return ""
    return (r.stderr.strip().split("\n") or [""])[-1][:90] or f"rc={r.returncode}"


def check_runnable(pkg: str, root: str, meta: dict) -> None:
    """Gate 8: the runnable contract layered on top of the read/build package —
    pre_artifacts.sh, solution/solve.sh (four-mode CLI), and schema_version 2.0."""
    pre = os.path.join(root, "pre_artifacts.sh")
    if not os.path.isfile(pre):
        fail(pkg, "缺 pre_artifacts.sh(提交契约:把工作树捕获成 model.patch)")
    else:
        err = _bash_n(pre)
        if err:
            fail(pkg, f"pre_artifacts.sh 语法错: {err}")

    solve = os.path.join(root, "solution", "solve.sh")
    if not os.path.isfile(solve):
        fail(pkg, "缺 solution/solve.sh(oracle 一键落地脚本)")
    else:
        if not os.access(solve, os.X_OK):
            fail(pkg, "solution/solve.sh 没有可执行位")
        err = _bash_n(solve)
        if err:
            fail(pkg, f"solution/solve.sh 语法错: {err}")
        body = open(solve, errors="ignore").read()
        # a solve.sh that only prints "no reference implementation" and exits 2 is a
        # legitimate shape for the 2 tasks that ship no oracle -- accept it as-is.
        if re.search(r'exit\s+2\b', body) and "--negative" not in body:
            pass
        else:
            for flag in ("--negative", "--baseline2", "--noop"):
                if flag not in body:
                    warn(pkg, f"solution/solve.sh 未处理 {flag} 模式(四态 CLI 不全)")

    sv = str(meta.get("schema_version") or "")
    if sv != "2.0":
        warn(pkg, f"task.toml schema_version={sv or '缺失'}(期望 '2.0')")


# ---------------------------------------------------------------- 每题
def check_pkg(bench: str, task: str) -> None:
    base = os.path.join(TASKS, bench, task)
    pkg = f"{bench}/{task}"
    root = base if os.path.isdir(os.path.join(base, "tests")) else os.path.join(base, "task")
    meta: dict = {}

    for rel in ("instruction.md", "task.toml", "environment", "tests"):
        if not os.path.exists(os.path.join(root, rel)):
            fail(pkg, f"缺 {rel}")
    dfs = sorted(glob.glob(os.path.join(root, "environment", "Dockerfile*")))
    if not dfs:
        fail(pkg, "缺 environment/Dockerfile")

    tt = os.path.join(root, "task.toml")
    if os.path.isfile(tt):
        try:
            meta = _toml_load(tt)
        except Exception as e:
            fail(pkg, f"task.toml 解析失败: {e}")
            meta = {}
        D = lambda v: v if isinstance(v, dict) else {}
        gpus = D(meta.get("environment")).get("gpus",
               D(meta.get("resources")).get("gpus", meta.get("gpus")))
        if gpus is None:
            warn(pkg, "task.toml 没有声明 gpus(检查过 [environment] / [resources] / 顶层)")
        vr = D(meta.get("verifier"))
        scope = (D(vr.get("entry")).get("primary_edit_paths")
                 or vr.get("primary_edit_paths") or meta.get("primary_edit_paths"))
        if not scope:                       # 部分题的可编辑范围只在 tests/test.sh 里声明
            tsp = os.path.join(root, "tests", "test.sh")
            if os.path.isfile(tsp):
                body = open(tsp, errors="ignore").read()
                if re.search(r'^\s*SCOPE=\(', body, re.M) or re.search(r'^\s*SUB=', body, re.M):
                    scope = ["<declared in tests/test.sh>"]
        if not scope:
            frozen = any(os.path.isfile(os.path.join(root, x)) for x in
                         ("tests/.frozen_hashes.json", "tests/verifier-correctness-manifest.json",
                          "tests/reward_manifest.json"))
            if not frozen:
                warn(pkg, "可编辑范围无处声明,且没有 sha256 冻结面 —— 无从判断哪些文件可改")
            # 有冻结面 = Type-3 端到端题的正常形制(整树可改,只冻结评分面),不告警

    ts = os.path.join(root, "tests", "test.sh")
    if not os.path.isfile(ts):
        fail(pkg, "缺 tests/test.sh")
    else:
        s = open(ts, errors="ignore").read()
        if "ref_speedup" in s:
            got = None
            rt = os.path.join(root, "tests", "ref_speedup.txt")
            if os.path.isfile(rt):
                try:
                    got = float(re.sub(r'[^0-9.]', '', open(rt).read()) or 0)
                except ValueError:
                    got = None
            if got is None:
                for mf in ("tests/verifier-correctness-manifest.json",
                           "environment/verifier-correctness-manifest.json",
                           "tests/reward_manifest.json"):
                    p = os.path.join(root, mf)
                    if os.path.isfile(p):
                        m = re.search(r'"ref_speedup"\s*:\s*(?:\{[^}]*"value"\s*:\s*)?([0-9.]+)',
                                      open(p, errors="ignore").read())
                        if m and float(m.group(1)) > 1:
                            got = float(m.group(1))
                            break
            hard = re.search(r'ref_speedup=([0-9]+\.[0-9]+)', s)
            if got is None and not hard:
                fail(pkg, "性能题但锚点无处可解析(会静默回落 1.0 → 所有 reward 归 0)")
            elif got is not None and got <= 1:
                fail(pkg, f"锚点 <= 1({got}) → reward 恒为 0")

    for df in dfs:
        check_dockerfile(pkg, df)

    rt = os.path.join(root, "tests", "ref_speedup.txt")
    cv = os.path.join(root, "tests", "ref_speedup.caveat.md")
    if os.path.isfile(rt) and os.path.isfile(cv):
        v = re.sub(r'[^0-9.]', '', open(rt).read())
        if v and v not in open(cv, errors="ignore").read():
            warn(pkg, f"caveat.md 里没出现 ref_speedup.txt 的值 {v}(两处可能已脱钩)")

    repo = os.path.join(root, "environment", "repo")
    for cand in ("solution/oracle.patch", "oracle.patch", "tests/oracle.patch"):
        op = os.path.join(root, cand)
        if os.path.isfile(op) and os.path.isdir(repo):
            fwd = subprocess.run(["git", "apply", "--check", "-p1", op],
                                 cwd=repo, capture_output=True).returncode == 0
            rev = subprocess.run(["git", "apply", "--check", "-R", "-p1", op],
                                 cwd=repo, capture_output=True).returncode == 0
            if not fwd:
                fail(pkg, f"{cand} 无法正向打在 environment/repo/ 上(树与参考补丁不匹配)")
            elif rev:
                fail(pkg, f"{cand} 反向也能打上 —— environment/repo/ 似乎已是修好的状态")
            break

    for dp, dn, fn in os.walk(root):
        for f in list(dn) + list(fn):
            rel = os.path.relpath(os.path.join(dp, f), root)
            if JUNK.search(rel):
                fail(pkg, f"残留构建/编辑器产物: {rel}")

    check_runnable(pkg, root, meta)


tasks = [(b, t) for b in WANT
         for t in sorted(os.listdir(os.path.join(TASKS, b)))
         if not t.startswith("_") and not t.startswith("delete_")
         and os.path.isdir(os.path.join(TASKS, b, t))]
for b, t in tasks:
    check_pkg(b, t)

print(f"fai_bench 自检 — 根目录 {ROOT}")
print(f"  子集 {', '.join(WANT)} / 共 {len(tasks)} 题")
if warns:
    print(f"\n提示 {len(warns)} 条:")
    for w in warns[:40]:
        print(f"  · {w}")
    if len(warns) > 40:
        print(f"  · … 另 {len(warns) - 40} 条")
if fails:
    print(f"\n失败 {len(fails)} 条:")
    for f in fails:
        print(f"  ✗ {f}")
    sys.exit(1)
print(f"\n全部通过:{len(tasks)} 题,0 失败。")

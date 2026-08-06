"""Shared harness for the graph normalize + idiom-collapse fusion task
(CPU, pure Python stdlib -- no numpy / torch).

Provides:
  * candidate loader (imports ``fuse`` from the repo under test);
  * an INDEPENDENT, obviously-correct graph EVALUATOR (the ground truth): the
    candidate's output graph is scored by whether it evaluates to the same
    external outputs as the input graph -- never by matching the oracle's graph;
  * a deterministic op-count metric (number of compute nodes);
  * deterministic corpus / idiom builders and an equivalence checker.

Graph representation: a plain dict
    {"inputs":[...], "outputs":[...], "constants":{name:[floats]},
     "nodes":[{"op","name","inputs","outputs","attrs"}...]}.
Tensors are 1-D float vectors; a length-1 vector is a scalar that broadcasts.
"""

import importlib.util
import math
import os
import random


# --------------------------------------------------------------------------- #
# Candidate / module loading.
# --------------------------------------------------------------------------- #
def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    path = os.path.join(repo_dir(), "dag_fusion.py")
    return load_module(path, "candidate_dag_fusion")


def load_module(path, name=None):
    name = name or ("kb_dag_mod_" + str(abs(hash(path))))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "fuse"):
        raise AttributeError(f"{path} does not define fuse(graph)")
    return mod


# --------------------------------------------------------------------------- #
# Independent evaluator (ground truth). Order-independent: resolves nodes whose
# inputs are all available until every external output is produced. Raises if a
# referenced tensor is never defined (e.g. a fusion deleted a still-needed value).
# --------------------------------------------------------------------------- #
def _broadcast(a, b):
    if len(a) == len(b):
        return a, b
    if len(a) == 1:
        return a * len(b), b
    if len(b) == 1:
        return a, b * len(a)
    raise ValueError(f"incompatible vector lengths {len(a)} vs {len(b)}")


def _elt(a, b, f):
    a, b = _broadcast(a, b)
    return [f(x, y) for x, y in zip(a, b)]


def _apply(op, ins, attrs):
    if op == "Sub":
        return _elt(ins[0], ins[1], lambda x, y: x - y)
    if op == "Add":
        return _elt(ins[0], ins[1], lambda x, y: x + y)
    if op == "Mul":
        return _elt(ins[0], ins[1], lambda x, y: x * y)
    if op == "Div":
        return _elt(ins[0], ins[1], lambda x, y: x / y)
    if op == "Pow":
        e = ins[1][0]
        return [x ** e for x in ins[0]]
    if op == "Sqrt":
        return [math.sqrt(x) for x in ins[0]]
    if op == "ReduceMean":
        v = ins[0]
        return [sum(v) / len(v)]
    if op in ("Identity", "Cast"):
        return list(ins[0])
    if op == "FusedNorm":
        x, w, b = ins[0], ins[1], ins[2]
        eps = attrs["epsilon"]
        mean = sum(x) / len(x)
        c = [xi - mean for xi in x]
        var = sum(ci * ci for ci in c) / len(c)
        std = math.sqrt(var + eps)
        norm = [ci / std for ci in c]
        w, _ = _broadcast(w, norm)
        b, _ = _broadcast(b, norm)
        return [norm[i] * w[i] + b[i] for i in range(len(norm))]
    raise ValueError(f"unknown op {op!r}")


def evaluate(graph, feeds):
    """feeds: {input_name: vector}. Returns {output_name: vector}."""
    values = {}
    for k, v in graph["constants"].items():
        values[k] = list(v)
    for k in graph["inputs"]:
        if k in feeds:
            values[k] = list(feeds[k])
    pending = list(graph["nodes"])
    progress = True
    while pending and progress:
        progress = False
        still = []
        for nd in pending:
            if all(t in values for t in nd["inputs"]):
                outs = _apply(nd["op"], [values[t] for t in nd["inputs"]], nd.get("attrs", {}))
                if not isinstance(outs, list) or (outs and isinstance(outs[0], list)):
                    raise ValueError(f"node {nd['name']} produced a non-vector")
                values[nd["outputs"][0]] = outs
                progress = True
            else:
                still.append(nd)
        pending = still
    if pending:
        raise ValueError(f"graph did not resolve; unresolved nodes: {[n['name'] for n in pending]}")
    result = {}
    for o in graph["outputs"]:
        if o not in values:
            raise ValueError(f"external output {o!r} is undefined in the graph")
        result[o] = values[o]
    return result


def op_count(graph):
    """The metric: number of compute nodes (constants live in a side table)."""
    return len(graph["nodes"])


def assert_valid_graph(graph):
    """Structural sanity: single-assignment outputs, external outputs produced."""
    seen = set(graph["inputs"]) | set(graph["constants"].keys())
    produced = set()
    for nd in graph["nodes"]:
        for o in nd["outputs"]:
            if o in produced:
                raise AssertionError(f"tensor {o!r} produced by more than one node")
            produced.add(o)
    defined = seen | produced
    for o in graph["outputs"]:
        if o not in defined:
            raise AssertionError(f"external output {o!r} is not defined")


def random_feeds(graph, rng, dim=8):
    # strictly positive inputs so that, for every graph shape in the suite, the
    # variance-plus-epsilon fed to Sqrt stays positive (mean-centering still
    # produces mixed-sign intermediates, so equivalence is exercised on both signs).
    feeds = {}
    for name in graph["inputs"]:
        feeds[name] = [rng.uniform(0.25, 3.0) for _ in range(dim)]
    return feeds


def graphs_equivalent(g_in, g_out, rng, trials=8, dim=8, tol=1e-6):
    """True iff g_out evaluates identically to g_in on random inputs."""
    assert_valid_graph(g_out)
    if list(g_in["inputs"]) != list(g_out["inputs"]) or list(g_in["outputs"]) != list(g_out["outputs"]):
        return False, "input/output signature changed"
    for _ in range(trials):
        feeds = random_feeds(g_in, rng, dim=dim)
        ref = evaluate(g_in, feeds)
        got = evaluate(g_out, feeds)
        for o in g_in["outputs"]:
            a, b = ref[o], got[o]
            if len(a) != len(b):
                return False, f"output {o} length {len(b)} != {len(a)}"
            for x, y in zip(a, b):
                if abs(x - y) > tol + tol * abs(x):
                    return False, f"output {o} differs: {x} vs {y}"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Deterministic idiom / corpus builders.
# --------------------------------------------------------------------------- #
def _n(op, name, ins, outs, attrs=None):
    return {"op": op, "name": name, "inputs": list(ins), "outputs": list(outs),
            "attrs": dict(attrs or {})}


def new_graph():
    return {"inputs": [], "outputs": [], "constants": {}, "nodes": []}


def build_idiom(g, prefix, x, form="canonical", weight=None, bias=None,
                eps=1e-5, exponent=2.0):
    """Append one normalization idiom on input tensor ``x`` to graph ``g`` and
    return the idiom's output tensor name. ``form`` in
    {canonical, cast, dupsub}. weight/bias default to fresh 1-D constants."""
    p = prefix
    two = f"{p}_two"
    epsn = f"{p}_eps"
    g["constants"][two] = [float(exponent)]
    g["constants"][epsn] = [float(eps)]
    if weight is None:
        weight = f"{p}_w"
        g["constants"][weight] = [1.3, 0.7, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8]
    if bias is None:
        bias = f"{p}_b"
        g["constants"][bias] = [0.1, -0.2, 0.05, 0.0, -0.1, 0.2, -0.05, 0.15]
    mean, c, sq, var, ve, std, norm, sc, y = (f"{p}_{t}" for t in
                                              ("mean", "c", "sq", "var", "ve", "std", "norm", "sc", "y"))
    g["nodes"].append(_n("ReduceMean", f"{p}_rm1", [x], [mean]))
    if form == "dupsub":
        c2 = f"{p}_c2"
        g["nodes"].append(_n("Sub", f"{p}_sub1", [x, mean], [c]))
        g["nodes"].append(_n("Sub", f"{p}_sub2", [x, mean], [c2]))
        pow_in, div_num = c, c2
    else:
        g["nodes"].append(_n("Sub", f"{p}_sub", [x, mean], [c]))
        pow_in = div_num = c
    if form == "cast":
        cc = f"{p}_cc"
        g["nodes"].append(_n("Cast", f"{p}_cast", [pow_in], [cc]))
        pow_in = cc
    g["nodes"].append(_n("Pow", f"{p}_pow", [pow_in, two], [sq]))
    g["nodes"].append(_n("ReduceMean", f"{p}_rm2", [sq], [var]))
    g["nodes"].append(_n("Add", f"{p}_adde", [var, epsn], [ve]))
    g["nodes"].append(_n("Sqrt", f"{p}_sqrt", [ve], [std]))
    g["nodes"].append(_n("Div", f"{p}_div", [div_num, std], [norm]))
    g["nodes"].append(_n("Mul", f"{p}_mul", [norm, weight], [sc]))
    g["nodes"].append(_n("Add", f"{p}_addb", [sc, bias], [y]))
    return y


def add_identity_noise(g, prefix, src, count=1):
    """Append ``count`` chained pass-through nodes ending in a fresh output."""
    prev = src
    last = src
    for i in range(count):
        out = f"{prefix}_id{i}"
        g["nodes"].append(_n("Identity", f"{prefix}_idn{i}", [prev], [out]))
        prev = out
        last = out
    return last


def make_bench_corpus(num_graphs=12, idioms_per_graph=4, seed=20260720):
    """A corpus dominated by idioms written in variant forms (cast / duplicated
    Sub) plus pass-through noise, so a matcher that only handles the single rigid
    canonical shape leaves most idioms expanded while a normalize+collapse pass
    reduces every graph. Deterministic."""
    rng = random.Random(seed)
    forms = ["cast", "dupsub", "canonical", "cast", "dupsub"]
    graphs = []
    for gi in range(num_graphs):
        g = new_graph()
        outs = []
        for k in range(idioms_per_graph):
            x = f"x{gi}_{k}"
            g["inputs"].append(x)
            form = forms[(gi + k) % len(forms)]
            y = build_idiom(g, f"g{gi}_i{k}", x, form=form)
            # a couple of stray pass-through nodes only the normalizer removes
            y = add_identity_noise(g, f"g{gi}_i{k}", y, count=1 + (k % 2))
            outs.append(y)
        g["outputs"] = outs
        graphs.append(g)
    return graphs

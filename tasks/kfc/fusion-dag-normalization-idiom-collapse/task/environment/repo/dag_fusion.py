"""Computation-graph fusion pass: normalize a DAG and collapse a normalization
idiom into a single fused node, emitting an equivalent graph with fewer ops.

Public entry point
------------------
    ``fuse(graph) -> graph``

A *graph* is a plain dict (JSON-shaped; no third-party types) describing a
computation DAG:

    {
      "inputs":    ["x", "weight", "bias"],   # external input tensor names
      "outputs":   ["y"],                      # external output tensor names
      "constants": {"two": [2.0], "eps": [1e-5],
                    "weight": [...], "bias": [...]},   # compile-time constants
      "nodes": [                               # topologically ordered
        {"op": "ReduceMean", "name": "n0", "inputs": ["x"],    "outputs": ["mean"], "attrs": {}},
        {"op": "Sub",        "name": "n1", "inputs": ["x","mean"], "outputs": ["xc"], "attrs": {}},
        ...
      ],
    }

Each *node* has an ``op`` (op-type string), a unique ``name``, ``inputs`` and
``outputs`` (tensor-name lists), and an ``attrs`` dict. A tensor is defined either
as a graph ``input``, as a key of ``constants``, or as some node's output. Edges
are implicit through shared tensor names.

Tensor / op semantics (the value model the fused node must preserve)
--------------------------------------------------------------------
Tensors are 1-D float vectors; a length-1 vector is a scalar that broadcasts.
    Sub(a,b)=a-b   Add(a,b)=a+b   Mul(a,b)=a*b   Div(a,b)=a/b   (elementwise,
    with length-1 broadcast); Pow(a,e)=a**e (``e`` a length-1 constant exponent);
    ReduceMean(a)=[mean(a)]; Sqrt(a)=[sqrt(ai)]; Identity(a)=a; Cast(a)=a
    (dtype-only, a numeric no-op here).
The *normalization idiom* is the chain
    mean=ReduceMean(x); c=Sub(x,mean); s=Pow(c,2); v=ReduceMean(s);
    ve=Add(v,eps); std=Sqrt(ve); n=Div(c,std); sc=Mul(n,weight); y=Add(sc,bias)
i.e. subtract the mean, square, mean, add a small constant, reciprocal-sqrt,
scale by a 1-D constant weight, shift by a 1-D constant bias. It collapses to a
single fused node
    {"op":"FusedNorm","inputs":["x","weight","bias"],"outputs":["y"],
     "attrs":{"epsilon": <eps value>}}
whose value equals the chain's output *exactly* (variance = mean of squares,
epsilon carried as the attribute).

Contract
--------
``fuse`` returns a NEW graph with the same ``inputs``, ``outputs`` and external
behaviour (every external output evaluates identically for every input), but with
fewer ``nodes`` where possible. It must never delete a tensor another node still
reads. It must NOT collapse a subgraph when doing so would change the result:

  * the square's exponent constant is not exactly 2;
  * the epsilon operand is not a compile-time constant scalar;
  * the scale/shift operands are not 1-D constants;
  * an interior tensor of the idiom is consumed outside the matched region
    (fusing would drop a value that is still needed).

In any of those cases the subgraph is left as-is (still correct, just not
collapsed). A graph containing no idiom is returned unchanged (same node count).

Why this implementation is weak
-------------------------------
This naive pass matches only the single canonical idiom on a rigid path: one
``Sub`` feeding both the square and the divide, and no intervening dtype/no-op
nodes. It does NOT normalize the graph first, so it misses idioms written in a
different-but-equivalent shape -- a dtype-only node sitting in the middle of the
chain, a duplicated ``Sub`` (the same ``x-mean`` computed twice, once for each
branch), and stray pass-through / no-op nodes it could have deleted. Those are
left expanded, so the output keeps far more nodes than necessary. Make the pass
collapse MORE (fewer nodes in the output) while keeping every external output
identical: normalize the graph (drop pass-through / dtype-only nodes, canonicalize
the duplicated branch) so more idiom occurrences become matchable, then collapse
each one -- always honouring the four safety conditions above.

Do not delegate the match or rewrite to a graph-optimizer / pattern-rewrite
engine or import one; build the traversal, the multi-condition match, the safety
check and the node replacement yourself. The scoring harness scans the submitted
file and scores it 0 if such a dependency appears.
"""


def _consumers(nodes):
    """tensor name -> list of nodes that read it as an input."""
    table = {}
    for nd in nodes:
        for t in nd["inputs"]:
            table.setdefault(t, []).append(nd)
    return table


def _producer(nodes):
    """tensor name -> the node that produces it (single-assignment DAG)."""
    table = {}
    for nd in nodes:
        for t in nd["outputs"]:
            table[t] = nd
    return table


def _is_const_scalar(graph, name):
    v = graph["constants"].get(name)
    return v is not None and len(v) == 1


def _is_const_vector(graph, name):
    return name in graph["constants"]


def _only_consumer(consumers, tensor, matched_names):
    """True iff every node reading ``tensor`` is inside the matched set."""
    for c in consumers.get(tensor, ()):
        if c["name"] not in matched_names:
            return False
    return True


def _match_canonical(graph, mean_node, cons, prod):
    """Try to match the single canonical idiom anchored at a ReduceMean whose
    output is the mean of a root tensor. Return a match record or None.

    Rigid: exactly one Sub(x, mean) feeding both Pow and Div, no dtype/no-op
    nodes in the path."""
    x = mean_node["inputs"][0]
    mean = mean_node["outputs"][0]

    # the single centering Sub(x, mean)
    subs = [n for n in cons.get(mean, ())
            if n["op"] == "Sub" and n["inputs"] == [x, mean]]
    if len(subs) != 1:
        return None
    sub = subs[0]
    c = sub["outputs"][0]

    c_readers = cons.get(c, ())
    pow_nodes = [n for n in c_readers if n["op"] == "Pow"]
    div_nodes = [n for n in c_readers if n["op"] == "Div" and n["inputs"][0] == c]
    if len(pow_nodes) != 1 or len(div_nodes) != 1:
        return None
    pow_node, div_node = pow_nodes[0], div_nodes[0]

    # exponent must be a constant exactly 2
    exp_name = pow_node["inputs"][1]
    exp_val = graph["constants"].get(exp_name)
    if exp_val is None or len(exp_val) != 1 or exp_val[0] != 2.0:
        return None
    sq = pow_node["outputs"][0]

    rm2 = prod.get(sq)
    if rm2 is None or rm2["op"] != "ReduceMean" or rm2["inputs"] != [sq]:
        return None
    var = rm2["outputs"][0]

    add_eps = prod.get(_single(cons.get(var)))
    if add_eps is None or add_eps["op"] != "Add":
        return None
    # epsilon operand: the Add input that is not `var`
    eps_name = add_eps["inputs"][1] if add_eps["inputs"][0] == var else add_eps["inputs"][0]
    if not _is_const_scalar(graph, eps_name):
        return None
    ve = add_eps["outputs"][0]

    sqrt_node = prod.get(_single(cons.get(ve)))
    if sqrt_node is None or sqrt_node["op"] != "Sqrt":
        return None
    std = sqrt_node["outputs"][0]

    if div_node["inputs"][1] != std:
        return None
    n = div_node["outputs"][0]

    mul_node = prod.get(_single(cons.get(n)))
    if mul_node is None or mul_node["op"] != "Mul":
        return None
    weight = mul_node["inputs"][1] if mul_node["inputs"][0] == n else mul_node["inputs"][0]
    if not _is_const_vector(graph, weight):
        return None
    sc = mul_node["outputs"][0]

    add_bias = prod.get(_single(cons.get(sc)))
    if add_bias is None or add_bias["op"] != "Add":
        return None
    bias = add_bias["inputs"][1] if add_bias["inputs"][0] == sc else add_bias["inputs"][0]
    if not _is_const_vector(graph, bias):
        return None
    y = add_bias["outputs"][0]

    matched = [mean_node, sub, pow_node, rm2, add_eps, sqrt_node, div_node, mul_node, add_bias]
    matched_names = {m["name"] for m in matched}
    interior = [mean, c, sq, var, ve, std, n, sc]
    outs = set(graph["outputs"])
    for t in interior:
        if t in outs or not _only_consumer(cons, t, matched_names):
            return None  # interior escape -> not safe to fuse

    eps_val = graph["constants"][eps_name][0]
    return {"matched": matched_names, "x": x, "weight": weight, "bias": bias,
            "y": y, "eps": eps_val}


def _single(seq):
    """The sole element of a 1-element list, else a name that never resolves."""
    if seq and len(seq) == 1:
        return seq[0]["outputs"][0]
    return None


def fuse(graph):
    nodes = [dict(n, inputs=list(n["inputs"]), outputs=list(n["outputs"]),
                  attrs=dict(n.get("attrs", {}))) for n in graph["nodes"]]
    work = {"inputs": list(graph["inputs"]), "outputs": list(graph["outputs"]),
            "constants": {k: list(v) for k, v in graph["constants"].items()},
            "nodes": nodes}

    cons = _consumers(nodes)
    prod = _producer(nodes)
    used = set()
    matches = []
    for nd in nodes:
        if nd["op"] != "ReduceMean":
            continue
        rec = _match_canonical(work, nd, cons, prod)
        if rec is None or rec["matched"] & used:
            continue
        matches.append(rec)
        used |= rec["matched"]

    if not matches:
        return work

    fused_nodes = []
    counter = 0
    kept = [n for n in nodes if n["name"] not in used]
    for rec in matches:
        counter += 1
        fused_nodes.append({
            "op": "FusedNorm",
            "name": "fused_norm_%d" % counter,
            "inputs": [rec["x"], rec["weight"], rec["bias"]],
            "outputs": [rec["y"]],
            "attrs": {"epsilon": rec["eps"]},
        })
    work["nodes"] = kept + fused_nodes
    return work

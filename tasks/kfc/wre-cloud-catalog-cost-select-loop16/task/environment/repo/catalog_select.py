"""Choosing cloud instance types for a batch of resource requests (CPU, numpy only).

A cloud catalog is a flat table of *offerings*: one row per (cloud, region, instance
type), carrying that type's vCPU count, its memory, the accelerator it comes with, and
the two prices it is billed at -- on demand and, where the cloud offers one, spot.  A
resource request does not name an instance type; it states what it needs -- optionally a
particular cloud, at least so many vCPUs, at least so much memory, exactly so many
accelerators of one kind -- how it is willing to be billed, and how much it is willing
to pay.  Turning a batch of such requests into a plan means, for every request,
narrowing the catalog to the offerings that satisfy it, working out what each of those
would actually cost under the request's billing mode, picking the cheapest, and -- because
the cheapest region can be out of capacity when the request is really launched -- laying
out the ordered ladder of regions it should fall back through.

The catalog and the requests arrive as flat integer columns:

* ``row_cloud`` / ``row_region`` / ``row_type`` -- who offers the row and under what name,
* ``row_vcpu`` / ``row_mem``      -- the shape of the offering,
* ``row_acc`` / ``row_accn``      -- the accelerator kind and count it comes with,
* ``row_price`` / ``row_spot``    -- what it is billed on demand and at spot,
* ``req_cloud``                   -- the cloud a request insists on, or none,
* ``req_vcpu`` / ``req_mem``      -- the floors a request states,
* ``req_acc`` / ``req_accn``      -- the accelerator a request insists on, or none,
* ``req_mode``                    -- how the request is willing to be billed,
* ``req_cap``                     -- what the request is willing to pay, or no ceiling,
* ``max_ladder``                  -- how long a fallback ladder may get,
* ``query_cloud`` / ``query_type`` -- an optional batch of catalog name lookups.

Two entry points are exposed:

* ``select_offerings``             -- the picks, the ladders and the statistics.
* ``select_offerings_with_lookup`` -- the same, plus resolve a batch of catalog names.

Both are thin wrappers over the single core ``_select_core``.
"""

import numpy as np

__all__ = ["select_offerings", "select_offerings_with_lookup", "MAX_ROWS",
           "MAX_REQUESTS", "MAX_CLOUDS", "MAX_REGIONS", "MAX_TYPES", "MAX_ACCS",
           "MAX_ACC_COUNT", "MAX_VCPU", "MAX_MEM", "MAX_PRICE", "MAX_LADDER",
           "MAX_QUERIES"]

MAX_ROWS = 1 << 22                  # catalog rows
MAX_REQUESTS = 1 << 22              # resource requests per batch
MAX_CLOUDS = 1 << 12                # cloud-id table size
MAX_REGIONS = 1 << 16               # region-id table size
MAX_TYPES = 1 << 20                 # instance-type-name table size
MAX_ACCS = 1 << 12                  # accelerator-kind table size
MAX_ACC_COUNT = 1 << 12             # accelerators on one offering
MAX_VCPU = 1 << 20                  # vCPUs on one offering
MAX_MEM = 1 << 32                   # memory on one offering
MAX_PRICE = 1 << 40                 # price, in the catalog's own smallest unit
MAX_LADDER = 64                     # longest fallback ladder
MAX_QUERIES = 1 << 22               # name-lookup batch cap


def select_offerings(row_cloud, row_region, row_type, row_vcpu, row_mem, row_acc,
                     row_accn, row_price, row_spot, req_cloud, req_vcpu, req_mem,
                     req_acc, req_accn, req_mode, req_cap, max_ladder):
    """Choose an instance type for every request and lay out its fallback ladder.

    Parameters
    ----------
    row_cloud, row_region, row_type : ndarray, shape (n_rows,), dtype int64
        The cloud, the region and the instance-type name of every catalog row.
    row_vcpu, row_mem : ndarray, shape (n_rows,), dtype int64
        The vCPU count and the memory of every catalog row.
    row_acc, row_accn : ndarray, shape (n_rows,), dtype int64
        The accelerator kind of every catalog row, ``-1`` for a row that comes with
        none, and how many of them it comes with, ``0`` in that case.
    row_price, row_spot : ndarray, shape (n_rows,), dtype int64
        What every catalog row is billed on demand and at spot, ``-1`` where the cloud
        does not offer that billing for it.
    req_cloud : ndarray, shape (n_req,), dtype int64
        The cloud every request insists on, ``-1`` when it does not care.
    req_vcpu, req_mem : ndarray, shape (n_req,), dtype int64
        The vCPU and memory floors every request states, ``0`` for no floor.
    req_acc, req_accn : ndarray, shape (n_req,), dtype int64
        The accelerator kind every request insists on, ``-1`` when it wants a row with
        none, and how many it insists on, ``0`` in that case.
    req_mode : ndarray, shape (n_req,), dtype int64
        How every request is willing to be billed: ``0`` on demand only, ``1`` spot
        where it is offered and on demand otherwise, ``2`` spot only.
    req_cap : ndarray, shape (n_req,), dtype int64
        The most every request is willing to pay, ``-1`` for no ceiling.
    max_ladder : int
        The most regions a fallback ladder may name.

    Returns
    -------
    dict
        The ten plan/statistics keys documented on :func:`_select_core` plus
        ``query_row``, which is ``None`` in this flavour.

    Raises
    ------
    ValueError
        On any contract violation listed in :func:`_select_core`.
    """
    return _select_core(row_cloud, row_region, row_type, row_vcpu, row_mem, row_acc,
                        row_accn, row_price, row_spot, req_cloud, req_vcpu, req_mem,
                        req_acc, req_accn, req_mode, req_cap, max_ladder, None, None)


def select_offerings_with_lookup(row_cloud, row_region, row_type, row_vcpu, row_mem,
                                 row_acc, row_accn, row_price, row_spot, req_cloud,
                                 req_vcpu, req_mem, req_acc, req_accn, req_mode,
                                 req_cap, max_ladder, query_cloud, query_type):
    """Choose instance types, then resolve a batch of catalog names.

    Parameters
    ----------
    row_cloud, row_region, row_type, row_vcpu, row_mem, row_acc, row_accn, row_price, row_spot, req_cloud, req_vcpu, req_mem, req_acc, req_accn, req_mode, req_cap, max_ladder
        As on :func:`select_offerings`.
    query_cloud : ndarray, shape (n_query,), dtype int64, or None
        Which cloud each looked-up instance-type name belongs to.  ``None`` disables the
        lookup output and must then be paired with ``query_type`` also ``None``.
    query_type : ndarray, shape (n_query,), dtype int64, or None
        The instance-type names to resolve against the catalog.  May be empty.

    Returns
    -------
    dict
        All eleven keys documented on :func:`_select_core`.

    Raises
    ------
    ValueError
        On any contract violation listed in :func:`_select_core`.
    """
    return _select_core(row_cloud, row_region, row_type, row_vcpu, row_mem, row_acc,
                        row_accn, row_price, row_spot, req_cloud, req_vcpu, req_mem,
                        req_acc, req_accn, req_mode, req_cap, max_ladder, query_cloud,
                        query_type)


def _select_core(row_cloud, row_region, row_type, row_vcpu, row_mem, row_acc, row_accn,
                 row_price, row_spot, req_cloud, req_vcpu, req_mem, req_acc, req_accn,
                 req_mode, req_cap, max_ladder, query_cloud, query_type):
    """The single derivation core shared by both public entry points.

    Notation
    --------
    ``n_rows`` is ``row_cloud.shape[0]``, ``n_req`` is ``req_cloud.shape[0]`` and
    ``n_query`` is ``query_type.shape[0]``.  ``mode(q)`` is ``req_mode[q]``.

    Validation (all of it before any output is produced; ``ValueError`` on failure)
    ------------------------------------------------------------------------------
    * the nine ``row_*`` and the seven ``req_*`` arrays are 1-D int64 arrays;
    * every ``row_*`` array has the same length ``n_rows``, and
      ``0 <= n_rows <= MAX_ROWS``;
    * every ``req_*`` array has the same length ``n_req``, and
      ``0 <= n_req <= MAX_REQUESTS``;
    * ``max_ladder`` is a real python ``int`` -- a ``bool`` does not count -- in
      ``[0, MAX_LADDER]``;
    * ``row_cloud`` entries are in ``[0, MAX_CLOUDS)``, ``row_region`` entries in
      ``[0, MAX_REGIONS)`` and ``row_type`` entries in ``[0, MAX_TYPES)``;
    * ``row_vcpu`` entries are in ``[1, MAX_VCPU]`` and ``row_mem`` entries in
      ``[1, MAX_MEM]``;
    * ``row_acc`` entries are ``-1`` or in ``[0, MAX_ACCS)``, ``row_accn`` entries are
      in ``[0, MAX_ACC_COUNT]``, and a row carries an accelerator kind exactly when it
      carries a positive count -- ``row_acc[r] == -1`` if and only if
      ``row_accn[r] == 0``;
    * ``row_price`` and ``row_spot`` entries are ``-1`` or in ``[0, MAX_PRICE]``;
    * ``req_cloud`` entries are ``-1`` or in ``[0, MAX_CLOUDS)``;
    * ``req_vcpu`` entries are in ``[0, MAX_VCPU]`` and ``req_mem`` entries in
      ``[0, MAX_MEM]``;
    * ``req_acc`` entries are ``-1`` or in ``[0, MAX_ACCS)``, ``req_accn`` entries are
      in ``[0, MAX_ACC_COUNT]``, and ``req_acc[q] == -1`` if and only if
      ``req_accn[q] == 0``;
    * ``req_mode`` entries are ``0``, ``1`` or ``2``;
    * ``req_cap`` entries are ``-1`` or in ``[0, MAX_PRICE]``;
    * ``query_cloud`` and ``query_type`` are either both ``None`` or both not ``None``;
      when they are given, both are 1-D int64 arrays of the same length
      ``n_query <= MAX_QUERIES``, every ``query_cloud`` entry is in
      ``[0, MAX_CLOUDS)`` and every ``query_type`` entry is in ``[0, MAX_TYPES)``.

    Behaviour
    ---------
    1. **What a row costs a request.**  Under ``mode(q) == 0`` a row is billable only if
       it has an on-demand price, and that price is what it costs.  Under
       ``mode(q) == 2`` a row is billable only if it has a spot price, and that price is
       what it costs.  Under ``mode(q) == 1`` a row costs its spot price when it has
       one and its on-demand price otherwise, and is billable unless it has neither.
       A row billed at its spot price is said to be *taken at spot*.

    2. **Feasibility.**  Row ``r`` is *feasible* for request ``q`` when ALL of

       - ``req_cloud[q] == -1`` or ``req_cloud[q] == row_cloud[r]``;
       - ``row_vcpu[r] >= req_vcpu[q]`` and ``row_mem[r] >= req_mem[q]`` -- the request
         states floors, not exact sizes;
       - the accelerator matches: when ``req_acc[q] == -1`` the row must carry no
         accelerator either, and otherwise the row's kind AND count must both equal the
         request's -- a request for four of a kind is not served by eight of it;
       - the row is billable under ``mode(q)``;
       - and what it costs is at most ``req_cap[q]``, unless that is ``-1``.

    3. **The pick.**  ``req_pick[q]`` is the feasible row that costs the least, ties
       broken by the smaller ``row_vcpu``, then the smaller ``row_mem``, then the
       smaller row index; ``-1`` when nothing is feasible.  ``req_price[q]`` is what
       that row costs and ``req_spot[q]`` is ``1`` when it is taken at spot and ``0``
       when it is not; both are ``-1`` when nothing is feasible.

    4. **The fallback ladder.**  A request's *candidate regions* are the distinct
       ``row_region`` values of its feasible rows, and a candidate region's own cost is
       the least a feasible row of that region costs the request.  The ladder lists
       the candidate regions ordered by ascending own cost, then by ascending region,
       and keeps only the first ``max_ladder`` of them.  The ladders of all requests are
       concatenated in ascending request order into ``ladder_region`` and
       ``ladder_cost``, with ``ladder_ptr`` the group boundaries: request ``q`` owns
       ``ladder_ptr[q] .. ladder_ptr[q + 1]``.

    5. **Statistics.**  ``req_n_feasible[q]`` counts request ``q``'s feasible rows and
       ``req_n_regions[q]`` counts its candidate regions BEFORE the ladder is truncated.
       ``row_n_feasible[r]`` counts the requests row ``r`` is feasible for and
       ``row_n_picked[r]`` counts the requests that pick it.

    6. **Name lookup.**  When ``query_cloud`` is ``None`` the lookup output is ``None``.
       Otherwise ``query_row[q]`` is the SMALLEST row index whose ``row_cloud`` and
       ``row_type`` are exactly ``query_cloud[q]`` and ``query_type[q]``, and ``-1``
       when the catalog has no such row.

    Returns
    -------
    dict with

    ``req_pick``        int64 (n_req,)
    ``req_price``       int64 (n_req,)
    ``req_spot``        int64 (n_req,)
    ``req_n_feasible``  int64 (n_req,)
    ``req_n_regions``   int64 (n_req,)
    ``ladder_ptr``      int64 (n_req + 1,)
    ``ladder_region``   int64 (n_ladder,)
    ``ladder_cost``     int64 (n_ladder,)
    ``row_n_feasible``  int64 (n_rows,)
    ``row_n_picked``    int64 (n_rows,)
    ``query_row``       int64 (n_query,)  or ``None``

    No input array is modified.
    """
    rows = (("row_cloud", row_cloud), ("row_region", row_region), ("row_type", row_type),
            ("row_vcpu", row_vcpu), ("row_mem", row_mem), ("row_acc", row_acc),
            ("row_accn", row_accn), ("row_price", row_price), ("row_spot", row_spot))
    reqs = (("req_cloud", req_cloud), ("req_vcpu", req_vcpu), ("req_mem", req_mem),
            ("req_acc", req_acc), ("req_accn", req_accn), ("req_mode", req_mode),
            ("req_cap", req_cap))
    for nm, a in rows + reqs:
        if not isinstance(a, np.ndarray) or a.ndim != 1 or a.dtype != np.int64:
            raise ValueError("%s must be a 1-D int64 ndarray" % nm)
    n_rows = int(row_cloud.shape[0])
    if n_rows > MAX_ROWS:
        raise ValueError("n_rows out of range")
    for nm, a in rows[1:]:
        if int(a.shape[0]) != n_rows:
            raise ValueError("%s length mismatch" % nm)
    n_req = int(req_cloud.shape[0])
    if n_req > MAX_REQUESTS:
        raise ValueError("n_req out of range")
    for nm, a in reqs[1:]:
        if int(a.shape[0]) != n_req:
            raise ValueError("%s length mismatch" % nm)
    if not isinstance(max_ladder, int) or isinstance(max_ladder, bool):
        raise ValueError("max_ladder must be a python int")
    if max_ladder < 0 or max_ladder > MAX_LADDER:
        raise ValueError("max_ladder out of range")
    rcl = np.ascontiguousarray(row_cloud)
    rrg = np.ascontiguousarray(row_region)
    rty = np.ascontiguousarray(row_type)
    rvc = np.ascontiguousarray(row_vcpu)
    rmm = np.ascontiguousarray(row_mem)
    rac = np.ascontiguousarray(row_acc)
    ran = np.ascontiguousarray(row_accn)
    rpr = np.ascontiguousarray(row_price)
    rsp = np.ascontiguousarray(row_spot)
    qcl = np.ascontiguousarray(req_cloud)
    qvc = np.ascontiguousarray(req_vcpu)
    qmm = np.ascontiguousarray(req_mem)
    qac = np.ascontiguousarray(req_acc)
    qan = np.ascontiguousarray(req_accn)
    qmd = np.ascontiguousarray(req_mode)
    qcp = np.ascontiguousarray(req_cap)
    if n_rows:
        if int(rcl.min()) < 0 or int(rcl.max()) >= MAX_CLOUDS:
            raise ValueError("row_cloud entries out of range")
        if int(rrg.min()) < 0 or int(rrg.max()) >= MAX_REGIONS:
            raise ValueError("row_region entries out of range")
        if int(rty.min()) < 0 or int(rty.max()) >= MAX_TYPES:
            raise ValueError("row_type entries out of range")
        if int(rvc.min()) < 1 or int(rvc.max()) > MAX_VCPU:
            raise ValueError("row_vcpu entries out of range")
        if int(rmm.min()) < 1 or int(rmm.max()) > MAX_MEM:
            raise ValueError("row_mem entries out of range")
        if int(rac.min()) < -1 or int(rac.max()) >= MAX_ACCS:
            raise ValueError("row_acc entries out of range")
        if int(ran.min()) < 0 or int(ran.max()) > MAX_ACC_COUNT:
            raise ValueError("row_accn entries out of range")
        if bool(((rac < 0) != (ran == 0)).any()):
            raise ValueError("row_acc and row_accn disagree about carrying an accelerator")
        if int(rpr.min()) < -1 or int(rpr.max()) > MAX_PRICE:
            raise ValueError("row_price entries out of range")
        if int(rsp.min()) < -1 or int(rsp.max()) > MAX_PRICE:
            raise ValueError("row_spot entries out of range")
    if n_req:
        if int(qcl.min()) < -1 or int(qcl.max()) >= MAX_CLOUDS:
            raise ValueError("req_cloud entries out of range")
        if int(qvc.min()) < 0 or int(qvc.max()) > MAX_VCPU:
            raise ValueError("req_vcpu entries out of range")
        if int(qmm.min()) < 0 or int(qmm.max()) > MAX_MEM:
            raise ValueError("req_mem entries out of range")
        if int(qac.min()) < -1 or int(qac.max()) >= MAX_ACCS:
            raise ValueError("req_acc entries out of range")
        if int(qan.min()) < 0 or int(qan.max()) > MAX_ACC_COUNT:
            raise ValueError("req_accn entries out of range")
        if bool(((qac < 0) != (qan == 0)).any()):
            raise ValueError("req_acc and req_accn disagree about wanting an accelerator")
        if int(qmd.min()) < 0 or int(qmd.max()) > 2:
            raise ValueError("req_mode entries out of range")
        if int(qcp.min()) < -1 or int(qcp.max()) > MAX_PRICE:
            raise ValueError("req_cap entries out of range")
    given = [x is not None for x in (query_cloud, query_type)]
    if any(given) and not all(given):
        raise ValueError("query inputs must be both given or both omitted")
    do_query = all(given)
    n_query = 0
    kcl = None
    kty = None
    if do_query:
        for nm, a in (("query_cloud", query_cloud), ("query_type", query_type)):
            if not isinstance(a, np.ndarray) or a.ndim != 1 or a.dtype != np.int64:
                raise ValueError("%s must be a 1-D int64 ndarray" % nm)
        n_query = int(query_type.shape[0])
        if int(query_cloud.shape[0]) != n_query:
            raise ValueError("query_cloud length mismatch")
        if n_query > MAX_QUERIES:
            raise ValueError("n_query out of range")
        kcl = np.ascontiguousarray(query_cloud)
        kty = np.ascontiguousarray(query_type)
        if n_query:
            if int(kcl.min()) < 0 or int(kcl.max()) >= MAX_CLOUDS:
                raise ValueError("query_cloud entries out of range")
            if int(kty.min()) < 0 or int(kty.max()) >= MAX_TYPES:
                raise ValueError("query_type entries out of range")

    raise NotImplementedError("_select_core is not implemented yet")

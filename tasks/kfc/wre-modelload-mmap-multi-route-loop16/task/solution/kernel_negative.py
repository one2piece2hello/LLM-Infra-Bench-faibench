# NEGATIVE variant (reviewer-only; never baked into the image). FAST-but-WRONG.
# Breaks the last-write-wins invariant: it builds the routing table from the declarations in
# REVERSED scan order, so the FIRST declaration of each name ends up winning instead of the last.
# When a name is declared by more than one file the wrong (earlier) file is returned -> correctness
# FAILS. Fast (vectorized) but wrong.
import numpy as np


def resolve_tensor_files(decl_name, decl_file, n_names, query):
    dn = np.asarray(decl_name, dtype=np.int64)[::-1]     # BUG: reversed -> first-write wins
    df = np.asarray(decl_file, dtype=np.int64)[::-1]
    route = np.full(int(n_names), -1, dtype=np.int64)
    route[dn] = df
    q = np.asarray(query, dtype=np.int64)
    return route[q].astype(np.int64)


def custom_kernel(data):
    decl_name, decl_file, n_names, query = data
    return resolve_tensor_files(decl_name, decl_file, n_names, query)

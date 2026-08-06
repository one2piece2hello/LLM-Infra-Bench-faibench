# Performance Optimization Task — submission entry point.
#
# Implement `resolve_tensor_files` to the contract in instruction.md, then make it as fast as
# possible. This is the ONLY file you edit. Leaving the NotImplementedError in place scores 0.
import numpy as np  # noqa: F401  (available; you may use it)


def resolve_tensor_files(decl_name, decl_file, n_names, query):
    """Resolve tensor names to the shard file that ultimately owns them (last-write-wins routing).

    A multi-file memory-mapped weight loader deserializes several shard files IN ORDER and, for
    every tensor name each file declares, records that name as routed to that file. When the same
    tensor name is declared by more than one file, the LAST file to declare it wins (its entry
    overwrites the earlier ones). After all files are scanned, tensors are fetched by name and must
    map to the file that finally owns them.

    Contract (deterministic; all correct implementations agree exactly):
      * ``decl_name``: a 1-D ``numpy`` integer array of ``D`` name ids, each in ``[0, n_names)``.
        ``decl_name[k]`` is the tensor name declared by the ``k``-th declaration, given in scan
        order (increasing ``k`` == later in the load).
      * ``decl_file``: a 1-D ``numpy`` integer array of ``D`` file ids; ``decl_file[k]`` is the file
        that made declaration ``k``.
      * ``n_names``: the total number of distinct name ids (name ids run over ``[0, n_names)``).
      * ``query``: a 1-D ``numpy`` integer array of ``Q`` name ids to resolve, each in
        ``[0, n_names)`` and guaranteed to have been declared at least once.
      * For each ``query[q]``, return the file id of the LAST declaration (largest ``k``) whose
        ``decl_name[k] == query[q]``.
      * Return a 1-D ``numpy`` ``int64`` array of length ``Q`` (one file id per query), in order.

    Args:
        decl_name: 1-D numpy int array of D name ids in scan order.
        decl_file: 1-D numpy int array of D file ids.
        n_names:   int, number of distinct name ids.
        query:     1-D numpy int array of Q name ids to resolve.

    Return:
        numpy.ndarray[int64] of shape (Q,): the owning file id per queried name.
    """
    raise NotImplementedError("implement resolve_tensor_files to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls. data = (decl_name, decl_file, n_names, query). Already wired
    — returns the per-query owning-file-id array from resolve_tensor_files."""
    decl_name, decl_file, n_names, query = data
    return resolve_tensor_files(decl_name, decl_file, n_names, query)

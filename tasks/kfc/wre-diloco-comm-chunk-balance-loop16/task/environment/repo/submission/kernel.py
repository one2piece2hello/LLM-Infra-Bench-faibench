# Performance Optimization Task — submission entry point.
#
# Implement `balance_chunks` to the contract in instruction.md so your partition is VALID, then make
# the bottleneck (largest) chunk's byte total as SMALL as possible. This is the ONLY file you edit.
# The verifier first checks that your boundaries validly split the gradient tensors into at most
# num_chunks contiguous chunks, then scores the largest chunk's byte total (smaller = higher
# reward). A submission that leaves NotImplementedError in place scores 0.


def balance_chunks(sizes, num_chunks):
    """Partition gradient tensors (in parameter order) into ring-all-reduce chunks, balanced.

    In decentralized / data-parallel training (DiLoCo outer step, ring all-reduce, reduce-scatter),
    the flattened gradient is transmitted as a small number of contiguous CHUNKS in parameter
    order. A ring all-reduce proceeds in lock-step rounds, so its wall-time is bounded by the
    LARGEST chunk (the bottleneck link transfers that many bytes each round). With at most
    ``num_chunks`` chunks available (one per ring slot / comm buffer), you want to place the chunk
    boundaries so the largest chunk's byte total is as small as possible — i.e. balance the chunks.
    Because a big tensor (e.g. an embedding) can dominate a chunk, where you cut matters.

    Args:
        sizes:      list[int] of length N (N >= 1); ``sizes[i] >= 1`` is gradient tensor i's byte
                    count, given in parameter order (chunks must be contiguous in this order).
        num_chunks: int P (>= 1); the maximum number of contiguous chunks.

    Return:
        boundaries: list[int]. The EXCLUSIVE end index of each contiguous chunk, in strictly
                    increasing order. Chunk k spans ``[boundaries[k-1], boundaries[k])`` (chunk 0
                    starts at 0), so ``boundaries[-1]`` must equal N. Validity (hard, checked first):
                    the list is non-empty, strictly increasing, every value in ``1..N``, the last
                    value is exactly N (all tensors covered), and ``len(boundaries) <= P``.
    Bottleneck chunk bytes (the score): ``max`` over chunks of the chunk's summed sizes — minimize it.
    """
    raise NotImplementedError("implement balance_chunks to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls.

    data = (sizes, num_chunks, config) where config = {"N": int, "num_chunks": int}. Already wired
    to call balance_chunks(sizes, num_chunks) and return the boundaries list.
    """
    sizes, num_chunks, config = data
    return balance_chunks(sizes, num_chunks)

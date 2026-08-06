# Performance Optimization Task — submission entry point.
#
# Implement `plan_arena` to the contract in instruction.md, then make the arena high-water it
# produces as SMALL as possible. This is the ONLY file you edit. The verifier first checks that
# every plan you return is VALID (every block placed; blocks that are simultaneously live never
# occupy overlapping bytes), then scores the peak arena size your plan achieves on the hidden
# workloads (smaller arena = higher reward). A submission that leaves NotImplementedError in
# place scores 0.

def plan_arena(sizes, alloc_step, free_step):
    """Assign each block a byte offset in one arena for an ONLINE alloc/free stream.

    A runtime services a time-ordered stream of block allocations and frees (like a small
    ``malloc`` / an inference-runtime memory arena). Block ``b`` (``0 .. N-1``) is allocated at
    time ``alloc_step[b]`` and freed at time ``free_step[b]`` (``alloc_step[b] < free_step[b]``),
    so it is LIVE during the half-open interval ``[alloc_step[b], free_step[b])``. Two blocks
    whose live intervals overlap must occupy disjoint byte ranges; once a block is freed its bytes
    may be reused by a later allocation. Your job is to place every block into the smallest
    possible contiguous arena by reusing the bytes of blocks that have already been freed.

    Args:
        sizes:      list[int] of length N. ``sizes[b]`` is the byte size of block b (> 0).
        alloc_step: list[int] of length N. ``alloc_step[b]`` is the step at which block b is
                    allocated.
        free_step:  list[int] of length N. ``free_step[b]`` is the step at which block b is freed
                    (``alloc_step[b] < free_step[b]``).

    Lifetime overlap: blocks b and c are simultaneously live iff their half-open intervals
        overlap: ``alloc_step[b] < free_step[c] and alloc_step[c] < free_step[b]``.

    Return:
        offsets: list[int] of length N. ``offsets[b] >= 0`` is the start byte of block b, which
                 occupies ``[offsets[b], offsets[b] + sizes[b])``.

    Validity (hard, checked first): for every pair (b, c) of simultaneously-live blocks,
        NOT (offsets[b] < offsets[c] + sizes[c] and offsets[c] < offsets[b] + sizes[b]).
    Arena size (the score): ``max_b (offsets[b] + sizes[b])`` — minimize it.
    """
    raise NotImplementedError("implement plan_arena to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls.

    data = (sizes, alloc_step, free_step, config) where config = {"N": int}. Already wired to
    call plan_arena and return the offsets list.
    """
    sizes, alloc_step, free_step, config = data
    return plan_arena(sizes, alloc_step, free_step)

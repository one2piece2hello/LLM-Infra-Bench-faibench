# Reviewer-only NEGATIVE (not baked into the image): fast but WRONG. Reuses a tiny fixed arena by placing
# blocks round-robin over just two slots WITHOUT respecting lifetimes, so simultaneously-live
# blocks are assigned overlapping byte ranges. It produces a very small arena (looks great on the
# metric) but the plan is INVALID -- the validity gate rejects it, so it scores 0.

def plan_arena(sizes, alloc_step, free_step):
    n = len(sizes)
    if n == 0:
        return []
    # place everything at offset 0 (and a second slot) ignoring liveness -> co-live blocks
    # (blocks whose alloc/free intervals overlap) collide in bytes -> invalid.
    biggest = max(sizes)
    offsets = []
    for b in range(n):
        offsets.append(0 if (b % 2 == 0) else biggest)  # only 2 slots, no liveness check
    return offsets


def custom_kernel(data):
    sizes, alloc_step, free_step, config = data
    return plan_arena(sizes, alloc_step, free_step)

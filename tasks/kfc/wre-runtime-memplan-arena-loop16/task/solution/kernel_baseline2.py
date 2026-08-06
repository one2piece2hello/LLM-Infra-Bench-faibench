# Reviewer-only BASELINE2 (not baked into the image): correct-but-naive BUMP-only allocator. Gives every
# block its own private region (offset = running prefix sum of all sizes) and NEVER reuses the
# bytes of a freed block -- the bump top only grows. Always VALID (no two blocks ever share
# bytes, so co-live blocks trivially do not overlap) but the arena equals the SUM of all block
# sizes, far larger than the max-concurrent-live peak the oracle's free-list achieves. Proves the
# correct-but-wasteful band (0 < vs_oracle < 1): a solver that implements the interface but does
# no free-space reuse.

def plan_arena(sizes, alloc_step, free_step):
    offsets = []
    cur = 0
    for s in sizes:
        offsets.append(cur)
        cur += s          # never reclaim freed space -> arena = sum(sizes)
    return offsets


def custom_kernel(data):
    sizes, alloc_step, free_step, config = data
    return plan_arena(sizes, alloc_step, free_step)

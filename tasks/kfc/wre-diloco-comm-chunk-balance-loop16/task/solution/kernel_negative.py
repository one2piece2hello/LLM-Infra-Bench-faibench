# Reviewer-only NEGATIVE (not baked into the image): looks best on the score but VIOLATES the chunk budget. It
# gives every tensor its OWN chunk (boundaries = [1,2,...,N]), so each chunk holds a single tensor
# and the bottleneck equals the largest single tensor -- the smallest possible bottleneck. But that
# uses N chunks, which exceeds num_chunks whenever N > num_chunks, so the validity gate
# (len(boundaries) <= num_chunks) rejects it -> scores 0. The classic "minimized the bottleneck,
# ignored the ring-slot budget" bug.

def balance_chunks(sizes, num_chunks):
    n = len(sizes)
    if n == 0:
        return []
    return list(range(1, n + 1))       # one chunk per tensor -> min bottleneck but N chunks (over budget)


def custom_kernel(data):
    sizes, num_chunks, config = data
    return balance_chunks(sizes, num_chunks)

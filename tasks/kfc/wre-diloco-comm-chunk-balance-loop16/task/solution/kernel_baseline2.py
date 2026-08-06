# Reviewer-only BASELINE2 (not baked into the image): correct-but-naive EQUAL-COUNT partition. Splits the tensors
# into num_chunks contiguous chunks with (nearly) equal NUMBER of tensors each, ignoring their byte
# sizes. Always a VALID contiguous partition, but because tensor sizes are highly skewed (a few huge
# embeddings among many small tensors), an equal-count chunk that happens to contain a huge tensor
# has a far larger byte total than the others -> the bottleneck (max chunk) is much higher than the
# size-balanced optimum -> 0 < vs_oracle < 1. A solver that implements the interface but cuts by
# count, not by bytes.

def balance_chunks(sizes, num_chunks):
    n = len(sizes)
    if n == 0:
        return []
    P = min(num_chunks, n)
    bounds = []
    for k in range(1, P + 1):
        end = (n * k) // P
        if end <= (bounds[-1] if bounds else 0):
            end = (bounds[-1] if bounds else 0) + 1
        bounds.append(min(end, n))
    bounds[-1] = n
    out = []
    for x in bounds:
        if not out or x > out[-1]:
            out.append(x)
    return out


def custom_kernel(data):
    sizes, num_chunks, config = data
    return balance_chunks(sizes, num_chunks)

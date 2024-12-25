import torch

import zhillm

from zhillm.internals_ import layers

def test_embedding(SIZE, BATCH, SEQLEN, SCALE):
    rtol, atol = (1e-3, 3e-4)

    input = torch.randint(0, SIZE[0], (BATCH, SEQLEN,), dtype=torch.int32, device="cuda",)
    input_subs = torch.randint(0, SEQLEN, (BATCH, SEQLEN,), dtype=torch.int32, device="cuda",)

    # input_subs = torch.tensor([0], dtype=torch.int32, device='cuda')
    ff = layers.Embedding(SIZE[1], SIZE[0], SCALE)
    print(ff)

if __name__ == "__main__":
    test_embedding((2, 8), 4, 4, True)

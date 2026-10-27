import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,vocab", [(1, 151936), (64, 152064), (7, 4097), (3, 100)])
def test_matches_torch_argmax(rows, vocab, dtype):
    from babyvllm.kernels.argmax import greedy_argmax

    torch.manual_seed(rows + vocab)
    x = torch.randn(rows, vocab, device="cuda", dtype=dtype) * 4
    out = torch.empty(rows, dtype=torch.int64, device="cuda")
    greedy_argmax(x, out)
    assert torch.equal(out, x.argmax(-1))


def test_ties_go_to_the_lowest_index_across_chunks():
    from babyvllm.kernels.argmax import greedy_argmax

    vocab, block = 20000, 4096
    x = torch.zeros(5, vocab, device="cuda", dtype=torch.bfloat16)
    x[0, [5, 9000, 19999]] = 3.0            # first chunk wins
    x[1, [block - 1, block]] = 3.0          # straddles a chunk boundary
    x[2, [12000, 8200]] = 3.0               # later chunk listed first
    x[3, 19999] = 3.0                       # last, partial chunk
    # row 4 all equal: index 0
    out = torch.empty(5, dtype=torch.int64, device="cuda")
    greedy_argmax(x, out, block=block)
    assert out.tolist() == [5, block - 1, 8200, 19999, 0]
    assert torch.equal(out, x.argmax(-1))


def test_row_slice_of_a_larger_buffer():
    from babyvllm.kernels.argmax import greedy_argmax

    buf = torch.randn(64, 152064, device="cuda", dtype=torch.bfloat16)
    out = torch.full((64,), -1, dtype=torch.int64, device="cuda")
    greedy_argmax(buf[:5], out[:5])
    assert torch.equal(out[:5], buf[:5].argmax(-1))
    assert (out[5:] == -1).all()

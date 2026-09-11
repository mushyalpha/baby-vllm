from babyvllm.worker.model_runner import BATCH_BUCKETS, CTX_BUCKETS, ceil_to_bucket


def test_ceil_to_bucket():
    assert ceil_to_bucket(1, BATCH_BUCKETS) == 1
    assert ceil_to_bucket(3, BATCH_BUCKETS) == 4
    assert ceil_to_bucket(64, BATCH_BUCKETS) == 64
    assert ceil_to_bucket(65, BATCH_BUCKETS) is None
    assert ceil_to_bucket(128, CTX_BUCKETS) == 256
    assert ceil_to_bucket(256, CTX_BUCKETS) == 256
    assert ceil_to_bucket(257, CTX_BUCKETS) == 2048
    assert ceil_to_bucket(2049, CTX_BUCKETS) is None

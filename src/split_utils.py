import numpy as np
from .config import RNG_SEED, TEST_N

def train_test_split_indices(n):
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(n)
    test_idx = idx[:TEST_N]
    train_idx = idx[TEST_N:]
    return train_idx, test_idx

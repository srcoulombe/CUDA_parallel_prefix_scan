# standard library dependencies

# external dependencies
import numpy as np
from numba import config, cuda

# local dependencies
from main import prefix_sum

config.CUDA_LOW_OCCUPANCY_WARNINGS = False

def setup(shape, seed: int = 42,):
  np.random.seed(seed)
  a = np.random.randint(-10, 11, shape)
  return a

def row_test(shape, seed: int = 42):
    a = setup(shape, seed = seed)
    out = prefix_sum(cuda.to_device(a), 0).copy_to_host()
    expected = np.cumsum(a, axis=1)
    assert np.allclose(expected, out)

def column_test(shape, seed: int = 42):
    a = setup(shape, seed = seed)
    out = prefix_sum(cuda.to_device(a), 1).copy_to_host()
    expected = np.cumsum(a, axis=0)
    assert np.allclose(expected, out)

def row_and_column_test(shape, seed: int = 42):
    a = setup(shape, seed = seed)
    out = prefix_sum(
        prefix_sum(
            cuda.to_device(a),
            0,
        ),
        1,
    ).copy_to_host()
    expected = np.cumsum(
        np.cumsum(a, axis=0),
        axis = 1
    )
    assert np.allclose(expected, out)

if __name__ == '__main__':
    print("running row prefix sums tests...", end = "")
    row_test((4, 32 * 3))
    row_test((4, 32 * 5))
    row_test((11, 32 * 3))
    row_test((32 * 3 - 1, 32 * 3))
    print(" passed!")

    print("running column prefix sums tests...", end = "")
    column_test((32 * 3, 4))
    column_test((32 * 5, 4))
    column_test((32 * 3, 11))
    column_test((32 * 3, 32 * 3 - 1))
    print(" passed!")

    print("running 2D prefix sums tests...", end = "")
    row_and_column_test((32, 32))
    row_and_column_test((32, 32*3))
    row_and_column_test((32*3, 32))
    print(" passed!")
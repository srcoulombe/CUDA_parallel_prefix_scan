# standard library dependencies

# external dependencies
import numpy as np
from numba import cuda

# local dependencies
from main import prefix_sum

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
    print("row prefix sums tests")
    row_test((4, 32 * 3))
    row_test((4, 32 * 5))
    row_test((11, 32 * 3))
    row_test((32 * 3 - 1, 32 * 3))

    print("column prefix sums tests")
    column_test((32 * 3, 4))
    column_test((32 * 5, 4))
    column_test((32 * 3, 11))
    column_test((32 * 3, 32 * 3 - 1))

    print("2D prefix sums tests")
    row_and_column_test((32, 32))
    row_and_column_test((32, 32*3))
    row_and_column_test((32*3, 32))

# standard library dependencies

# external dependencies
import numpy as np
from numba import cuda

# local dependencies
from main import row_prefix_sums, column_prefix_sums

def setup(shape, seed: int = 42,):
  np.random.seed(seed)
  a = np.random.randint(-10, 11, shape)
  return a

def better_row_test(shape, seed: int = 42):
    a = setup(shape, seed = seed)
    out = row_prefix_sums(cuda.to_device(a)).copy_to_host()
    expected = np.cumsum(a, axis=1)
    print(np.allclose(expected, out))

def better_column_test(shape, seed: int = 42):
    a = setup(shape, seed = seed)
    out = column_prefix_sums(cuda.to_device(a)).copy_to_host()
    expected = np.cumsum(a, axis=0)
    print(np.allclose(expected, out))

def better_2d_test(shape, seed: int = 42):
    a = setup(shape, seed = seed)

    out = column_prefix_sums(
        row_prefix_sums(
            cuda.to_device(a)
        )
    ).copy_to_host()
    expected = np.cumsum(
        np.cumsum(a, axis=0),
        axis = 1
    )
    print(np.allclose(expected, out))

if __name__ == '__main__':
    print("row prefix sums tests")
    better_row_test((4, 32 * 3))
    better_row_test((4, 32 * 5))
    better_row_test((11, 32 * 3))
    better_row_test((32 * 3 - 1, 32 * 3))

    print("column prefix sums tests")
    better_column_test((32 * 3, 4))
    better_column_test((32 * 5, 4))
    better_column_test((32 * 3, 11))
    better_column_test((32 * 3, 32 * 3 - 1))

    print("2D prefix sums tests")
    better_2d_test((32, 32))
    better_2d_test((32, 32*3))
    better_2d_test((32*3, 32))

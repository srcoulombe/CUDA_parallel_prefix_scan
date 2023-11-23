# standard library dependencies

# external dependencies
import numpy as np
from numba import cuda

# local dependencies
from kernels import (
  flat_device_scan,
  flat_device_scan_post,
  CUDA_WARPSIZE,
)

def single_block_row_test(seed: int = 42, inclusive: bool = True):
  np.random.seed(seed)
  a = np.random.randint(-10,11,CUDA_WARPSIZE).reshape((1, -1))
  b = cuda.to_device(a)
  threadblock_totals = cuda.to_device(
    np.zeros(a.size // CUDA_WARPSIZE, dtype=int)
  ).reshape((1, -1))
  flat_device_scan[(1,1), (CUDA_WARPSIZE // 2, 1)](
    b,
    threadblock_totals,
    inclusive,
    0,
  )
  got = b.copy_to_host()
  print(a)
  print(np.cumsum(a).reshape((1, -1)))
  print(got)
  print(np.allclose(np.cumsum(a), got))

def multi_block_row_test(seed: int = 42, inclusive: bool = True):
  np.random.seed(seed)
  a = np.random.randint(-10,11,CUDA_WARPSIZE * 3).reshape((1, -1))
  b = cuda.to_device(a)
  num_threadblocks = a.size // CUDA_WARPSIZE
  threadblock_totals = cuda.to_device(
    np.zeros(a.size // CUDA_WARPSIZE, dtype=int)
  ).reshape((1, -1))
  flat_device_scan[(1, num_threadblocks), (num_threadblocks, 1)](
    b,
    threadblock_totals,
    inclusive,
    0,
  )
  flat_device_scan[(1,1), (num_threadblocks, 1)](
      threadblock_totals,
      cuda.to_device(np.zeros_like(threadblock_totals)),
      False,
      1,
  )
  flat_device_scan_post[(num_threadblocks, 1), (CUDA_WARPSIZE,)](
      b,
      threadblock_totals,
  )
  got = b.copy_to_host()
  print(a)
  print(np.cumsum(a).reshape((1, -1)))
  print(got)
  print(np.allclose(np.cumsum(a), got))

def single_block_column_test(seed: int = 42, inclusive: bool = True):
  np.random.seed(seed)
  a = np.random.randint(-10,11,CUDA_WARPSIZE).reshape((-1, 1))
  b = cuda.to_device(a)
  threadblock_totals = cuda.to_device(
    np.zeros(a.size // CUDA_WARPSIZE, dtype=int)
  ).reshape((-1, 1))
  flat_device_scan[(1,1), (CUDA_WARPSIZE // 2, 1)](
    b,
    threadblock_totals,
    inclusive,
    1,
  )
  got = b.copy_to_host()
  print(a)
  print(np.cumsum(a).flatten())
  print(got)
  print(np.allclose(np.cumsum(a).flatten(), got.flatten()))

if __name__ == "__main__":
  single_block_row_test()
  # multi_block_row_test()

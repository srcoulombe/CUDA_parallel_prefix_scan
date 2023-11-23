# standard library dependencies

# external dependencies
import numpy as np
import numba
from numba import cuda

# local dependencies

CUDA_WARPSIZE = 32

@cuda.jit(device=True)
def global_memory_indexer(axis: int):
  block_x_index = cuda.blockIdx.x
  block_y_index = cuda.blockIdx.y
  if axis == 0:
    n = 2 * cuda.blockDim.x
    thread_index = cuda.threadIdx.x
    left_element_global_thread_index = 2 * thread_index + (n * block_x_index)
    right_element_global_thread_index = 2 * thread_index + 1 + (n * block_x_index)
    return (
      (block_y_index, left_element_global_thread_index),
      (block_y_index, right_element_global_thread_index),
    )
  else:
    n = 2 * cuda.blockDim.y
    thread_index = cuda.threadIdx.y
    left_element_global_thread_index = 2 * thread_index + (n * block_y_index)
    right_element_global_thread_index = 2 * thread_index + 1 + (n * block_y_index)
    return (
        (left_element_global_thread_index, block_x_index),
        (right_element_global_thread_index, block_x_index),
    )

@cuda.jit()
def better_flat_device_scan(global_data: np.ndarray,
                            threadblock_totals: np.ndarray,
                            inclusive: bool = False,
                            axis: int = 0,):
  # assumptions:
  # 1. axis is in (0, 1)
  # 2. threadblocks are either (1, D) or (D, 1)
  # 3. blockgrid is 2D
  if axis == 0:
    THREADS_PER_BLOCK = cuda.blockDim.x
    tid = cuda.threadIdx.x
  else:
    THREADS_PER_BLOCK = cuda.blockDim.y
    tid = cuda.threadIdx.y
  n = 2 * THREADS_PER_BLOCK
  bid_x = cuda.blockIdx.x
  bid_y = cuda.blockIdx.y
  global_indices = global_memory_indexer(axis)
  first_element_global_index = global_indices[0]
  second_element_global_index = global_indices[1]
  # populate shared array
  data = cuda.shared.array(shape=(32), dtype=numba.int32)
  data[2 * tid] = global_data[first_element_global_index]
  data[2 * tid + 1] = global_data[second_element_global_index]
  # upsweep
  offset = 1
  d = n // 2
  while d > 0:
    if tid < d:
      left_index = offset * (2 * tid + 1) - 1
      right_index = offset * (2 * tid + 2) - 1
      data[right_index] += data[left_index]
    offset *= 2
    d //= 2
  cuda.syncthreads()
  # some setup work
  total = data[n - 1]
  if tid == 0:
    data[n - 1] = 0
    threadblock_totals[bid_y, bid_x] = total
  # downsweep
  d = 1
  offset = n
  while d < n:
    offset //= 2
    if tid < d:
      left_index = offset * (2 * tid + 1) - 1
      right_index = offset * (2 * tid + 2) - 1
      tmp = data[left_index]
      data[left_index] = data[right_index]
      data[right_index] += tmp
    d *= 2
  cuda.syncthreads()
  # update global data
  if inclusive:
    global_data[first_element_global_index] = data[2 * tid + 1]
    if tid == THREADS_PER_BLOCK - 1:
      global_data[second_element_global_index] = total
    else:
      global_data[second_element_global_index] = data[2 * tid + 2]
  else:
    global_data[first_element_global_index] = data[2 * tid]
    global_data[second_element_global_index] = data[2 * tid + 1]

@cuda.jit
def better_flat_device_scan_post( global_data: np.ndarray,
                                  threadblock_cumulative_sums: np.ndarray,):
  bid_x = cuda.blockIdx.x
  bid_y = cuda.blockIdx.y
  global_idx_x, global_idx_y = cuda.grid(2)
  global_data[global_idx_x, global_idx_y] += threadblock_cumulative_sums[bid_x, bid_y]




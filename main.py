# standard library dependencies
from math import ceil

# external dependencies
import numpy as np
from numba import cuda

# local dependencies
from utils import pad_array, shapes_for_prefix_sum
from kernels import better_flat_device_scan, better_flat_device_scan_post

def prefix_sum(device_data_array, axis: int, warpsize: int = 32):
  shapes = shapes_for_prefix_sum(
      device_data_array.shape,
      axis,
      warpsize,
  )
  threadblock_totals = cuda.to_device(
      pad_array(
        np.zeros(
            shapes["threadblock_totals_shape"],
            dtype = int,
        ),
        axis,
    ),
  )
  better_flat_device_scan[shapes["initial_grid_shape"], shapes["thread_blocks_shape"]](
    device_data_array,
    threadblock_totals,
    True,
    axis,
  )
  better_flat_device_scan[shapes["threadblock_totals_grid_shape"], shapes["thread_blocks_shape"]](
      threadblock_totals,
      cuda.to_device(
          np.zeros(
              threadblock_totals.shape,
              dtype = int,
          )
      ),
      False,
      axis,
  )
  better_flat_device_scan_post[shapes["post_grid_shape"], shapes["post_thread_blocks_shape"]](
      device_data_array,
      threadblock_totals,
  )
  return device_data_array

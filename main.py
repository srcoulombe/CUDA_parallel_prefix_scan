# standard library dependencies
from math import ceil

# external dependencies
import numpy as np
from numba import cuda

# local dependencies
from kernels import better_flat_device_scan, better_flat_device_scan_post
from kernels import CUDA_WARPSIZE as WARPSIZE
from utils import *

def row_prefix_sums(device_data_array):
  threadblock_totals = cuda.to_device(
      pad_array(
        np.zeros(
            (
                device_data_array.shape[0],
                ceil(device_data_array.shape[1] / WARPSIZE)
            ),
            dtype = int,
        ),
        0,
    ),
  )
  grid_shape = (
      ceil(device_data_array.shape[1] / WARPSIZE),
      device_data_array.shape[0],
  )
  thread_blocks_shape = (
      ceil(WARPSIZE / 2),
      1,
  )
  better_flat_device_scan[grid_shape, thread_blocks_shape](
    device_data_array,
    threadblock_totals,
    True,
    0,
  )
  grid_shape = (
      ceil(threadblock_totals.shape[1] / WARPSIZE),
      threadblock_totals.shape[0],
  )
  better_flat_device_scan[grid_shape, thread_blocks_shape](
      threadblock_totals,
      cuda.to_device(
          np.zeros(
              threadblock_totals.shape,
              dtype = int,
          )
      ),
      False,
      0,
  )
  grid_shape = (
      device_data_array.shape[0],
      ceil(device_data_array.shape[1] / WARPSIZE)
  )
  better_flat_device_scan_post[grid_shape, (1, WARPSIZE)](
      device_data_array,
      threadblock_totals,
      0
  )
  return device_data_array

def column_prefix_sums(device_data_array):
  threadblock_totals = cuda.to_device(
      pad_array(
        np.zeros(
            (
                ceil(device_data_array.shape[0] / WARPSIZE),
                device_data_array.shape[1]
            ),
            dtype = int,
        ),
        1,
    ),
  )
  grid_shape = (
      device_data_array.shape[1],
      ceil(device_data_array.shape[0] / WARPSIZE),
  )
  thread_blocks_shape = (
      1,
      ceil(WARPSIZE / 2),
  )
  better_flat_device_scan[grid_shape, thread_blocks_shape](
    device_data_array,
    threadblock_totals,
    True,
    1,
  )

  grid_shape = (
    threadblock_totals.shape[1],
    ceil(threadblock_totals.shape[0] / WARPSIZE),
  )
  better_flat_device_scan[grid_shape, thread_blocks_shape](
      threadblock_totals,
      cuda.to_device(
          np.zeros(
              threadblock_totals.shape,
              dtype = int,
          )
      ),
      False,
      1,
  )
  grid_shape = (
      ceil(device_data_array.shape[0] / WARPSIZE),
      device_data_array.shape[1]
  )
  better_flat_device_scan_post[grid_shape, (WARPSIZE, 1)](
      device_data_array,
      threadblock_totals,
      1
  )
  return device_data_array

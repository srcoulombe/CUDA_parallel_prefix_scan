# standard library dependencies
from math import ceil

# external dependencies
import numpy as np

# local dependencies

def diff_to_pad_to(i: int, to: int):
  pad_width = to * ceil(i / to)
  return abs(i - pad_width)

def pad_array(array: np.ndarray,
              axis: int,
              to_multiple_of: int = 32):
  # note: this should be done BEFORE
  # we do EWST so that we can just pass
  # a reference to the device array
  assert array.ndim == 2
  assert axis in (0, 1)
  padded_lengths = tuple(
    diff_to_pad_to(array.shape[i], to_multiple_of)
    for i in (0, 1)
  )
  if axis == 0:
    padded_dimensions = (
      (0, 0),
      (0, padded_lengths[1]),
    )
  else:
    padded_dimensions = (
      (0, padded_lengths[0]),
      (0, 0)
    )
  return np.pad(
    array,
    padded_dimensions,
    mode = "constant",
    constant_values = 0,
  )

def shapes_for_prefix_sum(array_shape, axis: int, warpsize: int):
  assert axis in (0, 1)
  assert array_shape[1 - axis] % warpsize == 0
  if axis == 0:
    threadblock_totals_shape = (
        array_shape[0],
        ceil(array_shape[1] / warpsize),
    )
    thread_blocks_shape = (
        ceil(warpsize / 2),
        1,
    )
    initial_grid_shape = (
      ceil(array_shape[1] / warpsize),
      array_shape[0],
    )
    threadblock_totals_grid_shape = (
      ceil(threadblock_totals_shape[1] / warpsize),
      threadblock_totals_shape[0],
    )
    post_grid_shape = (
        array_shape[0],
        ceil(array_shape[1] / warpsize),
    )
    post_thread_blocks_shape = (1, warpsize)
  else:
    threadblock_totals_shape = (
        ceil(array_shape[0] / warpsize),
        array_shape[1],
    )
    thread_blocks_shape = (
        1,
        ceil(warpsize / 2),
    )
    initial_grid_shape = (
        array_shape[1],
        ceil(array_shape[0] / warpsize),
    )
    threadblock_totals_grid_shape = (
        threadblock_totals_shape[1],
        ceil(threadblock_totals_shape[0] / warpsize),
    )
    post_grid_shape = (
        ceil(array_shape[0] / warpsize),
        array_shape[1],
    )
    post_thread_blocks_shape = (warpsize, 1)

  return dict(
    threadblock_totals_shape = threadblock_totals_shape,
    thread_blocks_shape = thread_blocks_shape,
    initial_grid_shape = initial_grid_shape,
    threadblock_totals_grid_shape = threadblock_totals_grid_shape,
    post_grid_shape = post_grid_shape,
    post_thread_blocks_shape = post_thread_blocks_shape,
  )

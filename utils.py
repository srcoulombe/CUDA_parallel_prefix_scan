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

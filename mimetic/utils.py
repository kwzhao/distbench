"""Defines common utilities."""

from __future__ import annotations

import attrs
import random
from typing import Sequence, Iterable


def fsecs_to_inanos(secs: float) -> int:
  """Converts float seconds to int nanoseconds.

  This function assumes the decimal places of `secs` are meaningful and tries to
  approximately preserve them.

  Args:
    secs: Seconds as float.

  Returns:
    Integer nanoseconds.
  """
  return int(secs * 1e9)


@attrs.frozen
class Ecdf:
  """An empirical CDF."""

  _points: list[tuple[float, float]]  # a point is (value, quantile)

  @classmethod
  def from_data(cls, data: Sequence[float]) -> Ecdf:
    """Creates eCDF from a sequence of positive values."""
    if not data:
      raise ValueError('cannot build eCDF from no data points')
    data = list(data)
    data.sort()
    points, prev, prev_quantile = [], None, None
    if len(data) == 1:
      points.append((data[0], 0.0))
    else:
      for i, value in enumerate(data):
        quantile = i / (len(data) - 1)
        if (
            prev is None
            or value > prev * 1.1
            or prev_quantile is None
            or quantile > prev_quantile + 0.01
            or i == len(data) - 1
        ):
          points.append((value, quantile))
          prev, prev_quantile = value, quantile
    return Ecdf(points=points)

  def points(self) -> Iterable[tuple[float, float]]:
    return self._points

  def mean(self) -> float:
    """Estimates the mean of the eCDF using weighted midpoints.

    If the eCDF only contains one value, that value is returned directly.

    Returns:
      The approximate mean of the eCDF.
    """
    if len(self._points) == 1:
      return self._points[0][0]
    mean = 0
    for i in range(1, len(self._points)):
      x0, y0 = self._points[i - 1]
      x1, y1 = self._points[i]
      midpoint = (x0 + x1) / 2
      delta_y = y1 - y0
      mean += midpoint * delta_y
    return mean

  def value_at_quantile(self, y: float) -> float:
    """Approximates a value at a given quantile using linear interpolation.

    If the eCDF only contains one value, that value is returned directly.

    Args:
      y: The quantile.

    Returns:
      The approximate value at the given quantile.
    """
    if len(self._points) == 1:
      return self._points[0][0]
    for i in range(1, len(self._points)):
      if y <= self._points[i][1]:
        x0, y0 = self._points[i - 1]
        x1, y1 = self._points[i]
        # avoid sampling a value smaller than the min value in the data or
        # larger than the max value in the data.
        y = min(max(y, y0), y1)
        return x0 + (x1 - x0) / (y1 - y0) * (y - y0)
    raise ValueError(f'quantile must be in [0, 1), got {y}')

  def sample(self) -> float:
    """Samples a value from the distribution represented by the eCDF."""
    return self.value_at_quantile(random.random())

  def sample_int(self) -> int:
    """Samples an integer from the distribution represented by the eCDF."""
    return round(self.sample())

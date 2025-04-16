"""Defines the `arrival.Model` protocol and its implementations.

This module defines how to model and synthesize sequences of root requests.
"""

@attrs.frozen
class NormEcdf:
  """A normalized inter-arrival time distribution."""

  method_name: str
  _inner: utils.Ecdf

  @classmethod
  def from_root_history(cls, history: dataset.RootHistory) -> NormEcdf:
    """Creates a `NormEcdf` arrival model from a `dataset.RootHistory`."""
    length = len(history)
    roots = history.roots
    # Compute the timing deltas between adjacent root calls.
    # `dataset.RootHistory` guarantees the roots are sorted by start time.
    deltas = [
        roots[i + 1].timestamp - roots[i].timestamp for i in range(length - 1)
    ]
    # Normalize the deltas to the maximum observed value.
    max_delta = max(deltas)
    norm_deltas = [delta / max_delta for delta in deltas]
    norm_ecdf = utils.Ecdf.from_data(norm_deltas)
    return NormEcdf(history.method_name, norm_ecdf)

  def inner(self) -> utils.Ecdf:
    """Returns the inner normalized eCDF."""
    return self._inner

  def scaled_for_mean(self, mean: float) -> utils.Ecdf:
    """Returns an eCDF whose inter-arrivals are scaled for a given mean."""
    scaling_factor = mean / self._inner.mean()
    points = [
        (val * scaling_factor, quantile)
        for val, quantile in self._inner.points()
    ]
    return utils.Ecdf(points)

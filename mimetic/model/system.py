"""This module defines a model of a distributed system.

A distributed system is modeled as a collection of method models and root
arrival models. Models can be constructed directly from `dataset.Trace`s:

  ```python
  trace = dataset.Trace.from_data(df)
  system = Model.from_trace(trace)
  ```
"""

from __future__ import annotations

import attrs
import tqdm
from typing import TypeVar, Generic, Type, Mapping, Any

from mimetic import dataset
from mimetic.model import method, arrival, event

T = TypeVar('T', bound=method.Model)


@attrs.frozen
class Model(Generic[T]):
  """A distributed system model.

  This class is generic over the type of the underlying method models, provided
  the type implements the `method.Model` protocol.

  Attributes:
    method_models: A mapping from method name to method model.
    arrival_models: A mapping from method name to root arrival model.
  """

  method_models: dict[str, T]
  arrival_models: dict[str, arrival.NormEcdf]

  @classmethod
  def from_trace(
      cls,
      trace: dataset.Trace,
      method_model: Type[T] = method.DecisionTree,
      **kwargs: Any,
  ) -> Model[T]:
    """Creates a `Model` from a `dataset.Trace` and a `method.Model`."""
    total = trace.nr_methods()
    method_models = {
        history.method_name: method_model.from_method_history(history, **kwargs)
        for history in tqdm.tqdm(trace.method_histories(), total=total)
    }
    arrival_models = {
        history.method_name: arrival.NormEcdf.from_root_history(history)
        for history in trace.root_histories()
        if len(history) > 1  # we can't compute eCDFs without multiple values
    }
    return Model(method_models, arrival_models)

  def replace_method_models(self, replacements: Mapping[str, T]) -> Model[T]:
    """Creates a new `Model` with specified method models replaced."""
    method_models = {}
    for method_name, model in self.method_models.items():
      if method_name in replacements:
        method_models[method_name] = replacements[method_name]
      else:
        method_models[method_name] = model
    return Model(method_models, self.arrival_models)

  def sample_method(self, method_name: str) -> event.EventSequenceTree:
    """Samples an `event.EventSequenceTree` for a given method."""
    if method_name not in self.method_models:
      raise ValueError(f'Unknown method: {method_name}.')
    sequence = self.method_models[method_name].sample()
    children = [
        self.sample_method(call.method_name) for call in sequence.calls()
    ]
    return event.EventSequenceTree(sequence, children)

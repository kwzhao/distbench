"""Defines the `method.Model` protocol and its implementations.

This module wraps different underlying method models in a common interface for
extensible system-level modeling.
"""

from __future__ import annotations

import attrs
from typing import Protocol, Any

from mimetic import dataset
from mimetic.model import event, decision, builder


class Model(Protocol):
  """A protocol for method models."""

  method_name: str

  @classmethod
  def from_method_history(
      cls, history: dataset.MethodExecutionHistory, **kwargs: Any
  ):
    """Creates a method model from a `dataset.MethodExecutionHistory`."""

  def sample(self) -> event.EventSequence:
    """Samples an `event.EventSequence` from the model."""


@attrs.frozen
class DecisionTree:
  """A method built on stochastic decision trees.

  Attributes:
    method_name: The method name.
    decision_tree: The inner stochastic decision tree.
  """

  method_name: str
  decision_tree: decision.Tree

  @classmethod
  def from_method_history(
      cls, history: dataset.MethodExecutionHistory, alpha: float = 0.0
  ) -> DecisionTree:
    """Creates a method model from a `dataset.MethodExecutionHistory`."""
    tree_builder = builder.Tree()
    for execution in history:
      seq = event.EventSequence.from_execution(execution)
      tree_builder.insert(seq)
    decision_tree = tree_builder.build().prune(alpha)
    return DecisionTree(history.method_name, decision_tree)

  @classmethod
  def stub(cls, history: dataset.MethodExecutionHistory) -> DecisionTree:
    """Creates a stub that only models end-to-end delay."""
    tree_builder = builder.Tree()
    for execution in history.without_children():
      seq = event.EventSequence.from_execution(execution)
      tree_builder.insert(seq)
    decision_tree = tree_builder.build()
    return DecisionTree(history.method_name, decision_tree)

  def sample(self) -> event.EventSequence:
    """Samples an `event.EventSequence` from the model."""
    events = self.decision_tree.sample()
    return event.EventSequence.from_raw_sequence(self.method_name, events)

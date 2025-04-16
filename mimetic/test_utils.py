"""Defines testing utilities."""

import attrs
import pandas as pd

from mimetic import dataset
from mimetic.model import decision

@attrs.define
class TraceBuilder:
  """A builder for test distributed traces."""

  _df: pd.DataFrame = attrs.field(
      init=False,
      factory=lambda: pd.DataFrame(columns=dataset.COLUMNS),
  )

  def add_span(self, **kwargs) -> None:
    """Adds a span to the trace with data in `kwargs`."""
    given_columns = set(kwargs.keys())
    known_columns = set(dataset.COLUMNS)
    if not set(given_columns).issubset(known_columns):
      raise ValueError(f'unknown column(s): {given_columns - known_columns}')
    if 'parent_rpc_span_id' not in kwargs:
      kwargs['parent_rpc_span_id'] = 0
    if 'method_name' not in kwargs:
      kwargs['method_name'] = 'root'
    for col in dataset.COLUMNS:
      if col not in kwargs and 'timestamp' in col:
        kwargs[col] = 0
    self._df = pd.concat(
        [self._df, pd.DataFrame(kwargs, index=[0])], ignore_index=True
    )

  def build(self) -> dataset.Trace:
    """Builds the distributed trace."""
    return dataset.Trace.from_data(self._df)


@attrs.define
class MethodExecutionBuilder:
  """A builder for method executions."""

  _unsorted_child_events: list[dataset.ChildEvent] = attrs.field(
      init=False, factory=list
  )
  _cur_span_id: int = attrs.field(init=False, default=0)

  def add_child(
      self,
      method_name: str,
      start_offset: int,
      end_offset: int,
      request_size: int = 0,
  ) -> None:
    """Adds a child call with a unique span ID to the execution."""
    call_start = dataset.CallStart(
        method_name, self._cur_span_id, request_size, start_offset
    )
    call_finish = dataset.CallFinish(method_name, self._cur_span_id, end_offset)
    self._unsorted_child_events.append(call_start)
    self._unsorted_child_events.append(call_finish)
    self._cur_span_id += 1

  def add_return_and_build(
      self, return_offset: int, method_name: str, response_size: int = 0
  ) -> dataset.MethodExecution:
    """Adds a return event and builds a `dataset.MethodExecution`.

    This method clears the state of the builder so that subsequent calls to
    `add_child` and `add_return_and_build` will create a new execution.

    Args:
      return_offset: The offset of the `Return` event, in nanoseconds.
      method_name: The method name of the returned execution.
      response_size: The response size of the returned execution.

    Returns:
      A `dataset.MethodExecution`.
    """
    child_events = sorted(
        self._unsorted_child_events, key=lambda e: e.offset_ns
    )
    return_event = dataset.Return(response_size, return_offset)
    self._unsorted_child_events.clear()
    return dataset.MethodExecution(
        method_name, dataset.ServerStart(), child_events, return_event
    )


def tree_structure(tree: decision.Tree) -> ...:
  """Turns a decision tree into nested dicts and lists.

  This allows decision trees to be tested for equality using
  `assertSameStructure`. All weights `w` are transformed to `int(w * 100)`.

  Args:
    tree: a decision tree

  Returns:
    a decision tree as nested dicts and lists.
  """

  def tree_node_structure(node: decision.TreeNode):
    match node.model:
      case decision.Sentinel():
        model = 'sentinel'
      case decision.StartModel():
        model = 'start'
      case decision.CallGroupModel(members=members):
        members = sorted(members)
        members = '_'.join(map(str, members))
        model = f'call_group_{members}'
      case decision.ReturnModel():
        model = 'return'
      case _:
        raise TypeError('invalid model type')
    # Convert weights to int for equality testing.
    weights = list(map(lambda x: int(x * 100), node.weights))
    children = list(map(tree_node_structure, node.children))
    return {'model': model, 'weights': weights, 'children': children}

  return tree_node_structure(tree.root)

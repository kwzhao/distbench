"""Defines classes for working with distributed traces.

The main entry point is the `Trace` class, which wraps well-formed trace data
and hides data frame internals, providing methods to extract useful information
for downstream modeling.

Example usage:

  ```python
  trace = dataset.Trace.from_data(df)
  _ = trace.method_history(method_name)
  ```
"""

pd.options.mode.chained_assignment = None

COLUMNS = [
    'trace_id',
    'span_id',
    'parent_rpc_span_id',
    'method_name',
    'request_size',
    'response_size',
    'client_start_timestamp',
    'client_finish_timestamp',
    'server_start_timestamp',
    'server_finish_timestamp',
]


@attrs.frozen
class ServerStart:
  """Server start event."""

  offset_ns: int = 0


@attrs.frozen
class CallStart:
  """Call start event."""

  method_name: str
  span_id: int
  request_size: int
  offset_ns: int


@attrs.frozen
class CallFinish:
  """Call finish event."""

  method_name: str
  span_id: int
  offset_ns: int


ChildEvent = CallStart | CallFinish


@attrs.frozen
class Return:
  """Return event."""

  response_size: int
  offset_ns: int


Event = ServerStart | ChildEvent | Return


def _mk_child_events(child: pd.Series, parent: pd.Series) -> list[ChildEvent]:
  return [
      CallStart(
          method_name=child.method_name,
          span_id=child.span_id,
          request_size=child.request_size,
          offset_ns=utils.fsecs_to_inanos(
              child.client_start_timestamp - parent['server_start_timestamp']
          ),
      ),
      CallFinish(
          method_name=child.method_name,
          span_id=child.span_id,
          offset_ns=utils.fsecs_to_inanos(
              child.client_finish_timestamp - parent['server_start_timestamp']
          ),
      ),
  ]


@attrs.frozen
class MethodExecution:
  """A single method execution.

  We guarantee that child events are sorted by start time.

  Attributes:
    start_event: The start of server-side processing.
    child_events: Backend RPCs.
    return_event: The end of server-side processing.
  """

  method_name: str
  start_event: ServerStart
  child_events: list[ChildEvent]
  return_event: Return

  @classmethod
  def from_data(
      cls, parent: pd.Series, children: pd.DataFrame
  ) -> MethodExecution:
    """Creates a method execution from parent and child data."""
    child_events = children.apply(
        lambda child: _mk_child_events(child, parent),
        axis=1,
        result_type='reduce',
    )
    child_events = sum(child_events.to_list(), [])
    child_events = sorted(child_events, key=lambda e: e.offset_ns)
    return_event = Return(
        response_size=parent['response_size'],
        offset_ns=utils.fsecs_to_inanos(
            parent['server_finish_timestamp'] - parent['server_start_timestamp']
        ),
    )
    return MethodExecution(
        parent['method_name'], ServerStart(), child_events, return_event
    )

  def events(self) -> Iterable[Event]:
    """Yields all events for this RPC."""
    yield self.start_event
    yield from self.child_events
    yield self.return_event

  def nr_children(self) -> int:
    """Returns the total number of children of this execution."""
    return len(self.child_events) // 2

  def has_children(self) -> bool:
    """Returns True if this execution has children."""
    return self.nr_children() > 0

  def child_spans(self) -> Iterable[int]:
    """Yields all child span IDs."""
    return (ev.span_id for ev in self.child_events if isinstance(ev, CallStart))

  def without_children(self) -> MethodExecution:
    """Returns a copy of this execution without child calls."""
    return MethodExecution(
        self.method_name, self.start_event, [], self.return_event
    )


@attrs.frozen
class MethodExecutionHistory:
  """A collection of method executions for a particular method.

  Conceptually, a method execution history captures what a method "did" over
  some observation period.

  Attributes;
    method_name: The method name.
    calls: The list of RPCs.
  """

  method_name: str
  executions: list[MethodExecution]

  def has_children(self) -> bool:
    """Returns True if any execution in the history has children."""
    return any(rpc.has_children() for rpc in self.executions)

  def without_children(self) -> MethodExecutionHistory:
    """Returns a copy of this history without child calls."""
    return MethodExecutionHistory(
        self.method_name, [e.without_children() for e in self.executions]
    )

  def __len__(self) -> int:
    return len(self.executions)

  def __iter__(self):
    return iter(self.executions)


@attrs.frozen
class CallTree:
  """A `CallTree` is tree of method names, each corresponding to a call.

  Attributes:
    method_name: The method rooted at this node.
    children: The child `CallTree`s.
  """

  method_name: str
  children: list[CallTree]

  def size(self) -> int:
    """Returns the number of calls in the tree."""
    return 1 + sum(child.size() for child in self.children)


@attrs.frozen()
class MethodExecutionTree:
  """A tree of method executions.

  Parent-child relationships in the tree correspond to parent-child
  relationships in the executions.

  Attributes:
    execution: The execution rooted at this node.
    children: The child execution trees.
  """

  execution: MethodExecution
  children: list[MethodExecutionTree]

  def size(self) -> int:
    """Returns the total number of RPCs in the tree."""
    return 1 + sum(child.size() for child in self.children)

  def call_tree(self) -> CallTree:
    """Returns a `CallTree` corresponding to this tree."""
    return CallTree(
        self.execution.method_name,
        [child.call_tree() for child in self.children],
    )


@attrs.frozen()
class MethodExecutionTreeHistory:
  """A collection of method execution trees for a particular method.

  Attributes:
    method_name: The method name.
    trees: The method execution trees.
  """

  method_name: str
  trees: list[MethodExecutionTree]

  def call_trees(self) -> Iterable[CallTree]:
    return (t.call_tree() for t in self.trees)

  def __len__(self) -> int:
    return len(self.trees)

  def __iter__(self):
    return iter(self.trees)


@attrs.frozen
class ParentChildGraph:
  """A graph representing all parent-child relationships."""

  _graph: nx.DiGraph

  @classmethod
  def from_data(cls, df: pd.DataFrame) -> ParentChildGraph:
    """Creates a `ParentChildGraph` from data.

    A pre-condition is that `df` has already been validated--in particular, that
    it contains columns `span_id` and `parent_rpc_span_id`. We guarantee that a
    `span_id` is in the graph iff it is found in the `span_id` column of `df`.

    Args:
      df: A trace data frame.

    Returns:
      A `ParentChildGraph`

    Raises:
      ValueError: If unexpected span_id 0 found.
    """
    all_spans = set(df['span_id'])
    if 0 in all_spans:
      raise ValueError('DataFrame unexpected span_id: 0 is not a valid span_id')
    df_nonroot = df[df['parent_rpc_span_id'].isin(all_spans)]
    graph = nx.from_pandas_edgelist(
        df_nonroot,
        source='parent_rpc_span_id',
        target='span_id',
        create_using=nx.DiGraph(),
    )
    # At this point, `graph` contains all child spans and their parents.
    graph_spans = set(graph.nodes)
    # Add all parents without children.
    graph.add_nodes_from(all_spans - graph_spans)
    # Remove parent spans that aren't in `df`, but were referenced by a child.
    graph.remove_nodes_from(graph_spans - all_spans)
    return ParentChildGraph(graph)

  def children_of(self, span_id: int) -> Iterable[int]:
    """Returns the child span IDs of a given span ID."""
    return self._graph.neighbors(span_id)

  def nodes(self) -> Iterable[int]:
    """Returns all spans in the graph."""
    return nx.nodes(self._graph)

  def edges(self) -> Iterable[tuple[int, int]]:
    """Returns all edges in the graph."""
    return nx.edges(self._graph)

  def copy(self) -> ParentChildGraph:
    """Returns a copy of the graph."""
    return ParentChildGraph(self._graph.copy())

  def roots(self) -> Iterable[int]:
    """Returns all root spans in the graph."""
    return [x for x in self._graph.nodes() if self._graph.in_degree(x) == 0]


@attrs.frozen
class Root:
  """A root request in a trace.

  A root request is derived from any span for which `parent_rpc_span_id == 0`.
  """

  method_name: str
  timestamp: float


@attrs.frozen
class RootHistory:
  """A collection of root requests, sorted by increasing timestamp.

  Attributes:
    method_name: The method name.
    roots: The trace roots.
  """

  method_name: str
  roots: Sequence[Root]

  def __len__(self) -> int:
    return len(self.roots)

  def __iter__(self):
    return iter(self.roots)


@attrs.frozen
class Trace:
  """A distributed trace.

  This class wraps a data frame and provides methods to collect method execution
  histories.
  """

  _df: pd.DataFrame
  _graph: ParentChildGraph

  @classmethod
  def from_data(cls, df: pd.DataFrame) -> Trace:
    """Creates a `Trace` from a dataframe.

    Well-formed data frames have at least the following columns:

    - trace_id
    - span_id
    - parent_rpc_span_id
    - method_name
    - request_size
    - response_size
    - client_start_timestamp
    - client_finish_timestamp
    - server_start_timestamp
    - server_finish_timestamp

    The data is assumed to have been collected from [go/yangtze-river] with
    the same underlying types.

    Args:
      df: The trace data frame.

    Returns:
      A `Trace` object.

    Raises:
      ValueError: If the dataframe does not contain the required columns.
    """
    if not set(COLUMNS).issubset(df.columns):
      raise ValueError(
          f'DataFrame missing columns: expected {COLUMNS}, got {df.columns}'
      )
    df = df[COLUMNS].set_index('span_id', drop=False)
    graph = ParentChildGraph.from_data(df)
    return Trace(df, graph)

  def data(self) -> pd.DataFrame:
    """Returns the data frame backing the trace."""
    return self._df.copy()

  def parent_child_graph(self) -> ParentChildGraph:
    """Returns the parent-child relationship graph."""
    return self._graph.copy()

  def _children(self, span_id: int) -> pd.DataFrame:
    children = list(self._graph.children_of(span_id))
    children = np.array(children, dtype=self._df.index.dtype)
    return self._df.loc[children]

  def _method_history(
      self, method_name: str, method_df: pd.DataFrame
  ) -> MethodExecutionHistory:
    calls = []
    for _, span in method_df.iterrows():
      children = self._children(span['span_id'])
      calls.append(MethodExecution.from_data(span, children))
    return MethodExecutionHistory(method_name, calls)

  def method_history(self, method_name: str) -> MethodExecutionHistory:
    """Returns the history of a method."""
    return self._method_history(
        method_name, self._df[self._df['method_name'] == method_name]
    )

  def method_histories(self) -> Iterable[MethodExecutionHistory]:
    """Yields all method execution histories."""
    for method_name, group in self._df.groupby('method_name'):
      yield self._method_history(str(method_name), group)

  def nr_spans(self, method_name: str) -> int:
    """Returns the total number of spans for a method."""
    return len(self._df[self._df['method_name'] == method_name])

  def method_tree(self, span_id: int) -> MethodExecutionTree:
    """Returns the tree of method executions for a given span."""
    span = self._df.loc[span_id]
    rpc = MethodExecution.from_data(span, self._children(span_id))
    children = [self.method_tree(span_id) for span_id in rpc.child_spans()]
    return MethodExecutionTree(rpc, children)

  def _method_tree_history(
      self, method_name: str, method_df: pd.DataFrame
  ) -> MethodExecutionTreeHistory:
    trees = []
    for _, span in method_df.iterrows():
      trees.append(self.method_tree(span['span_id']))
    return MethodExecutionTreeHistory(method_name, trees)

  def method_tree_history(self, method_name: str) -> MethodExecutionTreeHistory:
    """Returns the method execution tree history of a given method."""
    return self._method_tree_history(
        method_name, self._df[self._df['method_name'] == method_name]
    )

  def method_tree_histories(self) -> Iterable[MethodExecutionTreeHistory]:
    """Yields all method execution tree histories for all methods."""
    for method_name, group in self._df.groupby('method_name'):
      yield self._method_tree_history(str(method_name), group)

  def nr_methods(self) -> int:
    """Returns the total number of methods in the trace."""
    return self._df['method_name'].nunique()

  def roots(self) -> Iterable[Root]:
    """Yields all trace roots sorted by timestamp."""
    root_span_ids = set(self._graph.roots())
    for span in (
        self._df[self._df['span_id'].isin(root_span_ids)]
        .sort_values(by='client_start_timestamp')
        .itertuples()
    ):
      yield Root(span.method_name, span.client_start_timestamp)

  def root_histories(self) -> Iterable[RootHistory]:
    """Yields all trace root histories."""
    root_span_ids = set(self._graph.roots())
    root_df = self._df[self._df['span_id'].isin(root_span_ids)]
    for method_name, group in root_df.groupby('method_name'):
      group = group.sort_values(by='client_start_timestamp')
      roots = [
          Root(span.method_name, span.client_start_timestamp)
          for span in group.itertuples()
      ]
      yield RootHistory(method_name, roots)

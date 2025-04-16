"""This module defines a stochastic decision tree for modeling RPCs.

Each node contains a sampleable model for a particular kind of event. For
example, a node might contain a `ReturnModel` from which we can sample response
sizes.

Moreover, each child node also has an associated weight. The sum of the child
weights at any level of the tree is always one. This allows us to sample paths
through the tree from root to leaf, where every path is a plausible RPC event
sequence.

The types in this module are not intended to be constructed directly, but rather
through the builders in `builder.py`.

Example usage:

  ```python
  builder = builder.Tree()
  for event_sequence in ...:
    builder.insert(event_sequence)
  decision_tree = builder.build()
  # Now we can sample event sequences back
  sampled = decision_tree.sample()
  ```
"""

@attrs.frozen
class Sentinel:
  """A sentinel model, for conveniently implementing trees."""

  def sample(self, _) -> event.Event:
    raise NotImplementedError


@attrs.frozen
class StartModel:
  """A model for `event.Start` events."""

  def sample(self, event_id: int) -> event.Start:
    """Samples an `event.Start`."""
    return event.Start(event_id)


@attrs.frozen
class DependencyOnStart:
  """A model for dependencies on `event.Start` events.

  This model captures the notion that the timing of some modeled event can
  depend on when an RPC begins server-side processing (`event.Start`), with some
  distribution of time between them.

  Attributes:
    deltas_ns: A continuous distribution of deltas in nanoseconds, relative to
      the `event.Start`.
  """

  deltas_ns: utils.Ecdf = attrs.field(eq=False, repr=False)

  def sample(self) -> event.DependencyOnStart:
    """Samples an `event.DependencyOnStart`."""
    delta_ns = self.deltas_ns.sample_int()
    return event.DependencyOnStart(delta_ns)


@attrs.frozen
class DependencyOnCall:
  """A model for dependencies on `event.Call`.

  This model captures the notion that the timing of some modeled event can
  depend on when some other RPC finishes, with some distribution of time between
  them.

  Attributes:
    level: The tree level of the dependency (excluding the sentinel).
    method_name: The method_name of the dependency call.
    deltas_ns: A continuous distribution of deltas in nanoseconds, relative to
      the end of the `event.Call`.
  """

  level: int
  method_name: str
  deltas_ns: utils.Ecdf = attrs.field(eq=False, repr=False)

  def sample(self) -> event.DependencyOnCall:
    """Samples an `event.DependencyOnCall`."""
    delta_ns = self.deltas_ns.sample_int()
    return event.DependencyOnCall(self.level, self.method_name, delta_ns)


Dependency = DependencyOnStart | DependencyOnCall


@attrs.frozen
class CallModel:
  """A model for `event.Call`.

  Each `CallModel` models calls to a particular backend during the execution of
  a particular RPC.

  Attributes:
    method_name: The method name.
    request_sizes: A discrete distribution of request sizes.
    dependency: The timing dependency.
  """

  method_name: str
  request_sizes: utils.Ecdf = attrs.field(repr=False)
  dependency: Dependency

  def sample(self) -> event.Call:
    """Samples an `event.Call`."""
    request_size = self.request_sizes.sample_int()
    dependency = self.dependency.sample()
    return event.Call(self.method_name, request_size, dependency)


@attrs.frozen
class CallGroupModel:
  """A model for `event.CallGroup` events.

  A call group is a group of concurrent RPCs, and it is uniquely identified by
  its member method names.

  Attributes:
    members: The method names in this call group, with duplicates allowed.
    calls: A mapping from method name to `CallModel`.
  """

  members: tuple[str, ...]
  calls: dict[str, CallModel]

  def uniqueified_members(self) -> Iterable[tuple[str, int]]:
    """Gets the members of this call group.

    Each member method is made unique by associating it with its index. The
    indices are only unique for members with the same method.

    Yields:
      `(method_name, index)` for each call in the call group.
    """
    counts = collections.defaultdict(int)
    for member in self.members:
      yield (member, counts[member])
      counts[member] += 1

  def sample(self, event_id: int) -> event.CallGroup:
    """Samples an `event.CallGroup`."""
    calls = [self.calls[method].sample() for method in self.members]
    return event.CallGroup(event_id, calls)


@attrs.frozen
class ReturnModel:
  """A model for `event.Return` events.

  Attributes:
    response_sizes: A discrete distribution of response sizes.
    deltas_ns: A continuous distribution of deltas in nanoseconds.
  """

  response_sizes: utils.Ecdf = attrs.field(repr=False)
  deltas_ns: utils.Ecdf = attrs.field(repr=False)

  def sample(self, event_id: int) -> event.Return:
    """Samples an `event.Return`."""
    response_size = self.response_sizes.sample_int()
    delta_ns = self.deltas_ns.sample_int()
    return event.Return(event_id, response_size, delta_ns)


Model = Sentinel | StartModel | CallGroupModel | ReturnModel


@attrs.frozen
class TreeNode:
  """A node in the decision tree.

  Each contains a sampleable model for a particular kind of event.

  Attributes:
    model: An event model.
    count: The number of samples from which the model was built.
    weights: The weights of all child nodes. All weights sum to one.
    children: The child nodes.
  """

  model: Model
  count: int
  weights: list[float]
  children: list[TreeNode]

  def sample(self, acc: list[event.Event], level: int) -> None:
    """Samples `event.Event`s from the tree rooted at this node.

    The sampled events are appended to the accumulator `acc`. This method
    samples the model at this node then recursively samples a model from a child
    node. The child node is chosen randomly according to the child weights.

    Args:
      acc: An accumulator for sampled events.
      level: The level of the tree, excluding the `Sentinel`.
    """
    if not isinstance(self.model, Sentinel):
      acc.append(self.model.sample(level))
    if self.children:
      child = random.choices(self.children, weights=self.weights)[0]
      child.sample(acc, level + 1)

  def weights_and_children(self) -> Iterable[tuple[float, TreeNode]]:
    """Gets the weights and children of this node."""
    return zip(self.weights, self.children)

  def prune(self, alpha: float) -> TreeNode:
    """Prunes the tree rooted at this node.

    We say a stochastic decision tree is pruned at level `alpha` if all branches
    with weights less than `alpha` are removed.

    Args:
      alpha: The pruning level.

    Returns:
      A new `TreeNode` which has been alpha-pruned.
    """
    if not self.children:
      return copy.deepcopy(self)
    # Filter for children that occur more frequently than `alpha`.
    is_large_enough = lambda w: w >= alpha
    weights, children = zip(*[
        (w, c.prune(alpha))
        for w, c in self.weights_and_children()
        if is_large_enough(w)
    ])
    # Renormalize weights and build a new `TreeNode`.
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]
    return TreeNode(self.model, self.count, list(weights), list(children))

  def size(self) -> int:
    """Returns the number of nodes in the tree."""
    return 1 + sum(child.size() for child in self.children)


@attrs.frozen
class Tree:
  """A stochastic decision tree.

  Intuitively, this tree captures what happens during an RPC. Each node in the
  tree represents some kind of event, and from that event, there is some
  probability of other events occurring.

  As an example, consider a model for cache retrieval. From the start of
  server-side processing, there is some probability of a cache hit, and some
  probability of a cache miss. On cache hit, the RPC will return immediately,
  but on cache miss, the RPC will issue backend calls to 1) open a file and 2)
  read it before returning. This logic can be represented as a stochastic
  decision tree.

  Attributes:
    root: The root `TreeNode`, which always contains a `Sentinel`.
  """

  root: TreeNode

  def sample(self) -> list[event.Event]:
    """Samples an sequence of `event.Event`s from the tree."""
    acc = []
    self.root.sample(acc, -1)
    return acc

  def prune(self, alpha: float) -> Tree:
    root = self.root.prune(alpha)
    return Tree(root)

  def size(self) -> int:
    """Returns the number of nodes in the tree."""
    return self.root.size()

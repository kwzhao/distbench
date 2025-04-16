"""Defines builders for decision trees.

Intuitively, decision trees are built from event sequences. For any two event
sequences `(a_1, a_2, ..., a_n)` and `(b_1, b_2, ..., b_n)`, the first `i` at
which `a_i != b_i` will create a corresponding branch in the decision tree.

This module can be seen as the glue between the event types in in `event.py` and
the model types in `decision.py`. Event types correspond to builder types, which
in turn correspond to model types. For example, the `ReturnModel` builder in
this module takes as input a collection of `event.Return` events and uses them
to construct a `decision.ReturnModel`.

In most cases, users will interface with the `Tree` builder:

  ```python
  # Given an `event_sequence: event.EventSequence`...
  builder = builder.Tree()
  builder.insert(event_sequence)
  decision_tree = builder.build()
  ```
"""

@attrs.define
class Sentinel:
  """A sentinel for conveniently implementing trees."""

  def add_sample(self, _) -> None:
    raise NotImplementedError

  def build(self) -> decision.Sentinel:
    return decision.Sentinel()


@attrs.define
class StartModel:
  """A builder for `decision.StartModel`."""

  def add_sample(self, _) -> None:
    ...

  def build(self) -> decision.StartModel:
    return decision.StartModel()


@attrs.define
class CallModel:
  """A builder for `decision.CallModel`.

  Attributes:
    method_name: The method name of the modeled call.
    request_sizes: A list of observed request sizes.
    dependency_candidates: A list of dependency candidates, from which one will
      be chosen.
  """

  method_name: str
  request_sizes: list[int]
  dependency_candidates: list[event.Dependency]

  @classmethod
  def with_sample(cls, call: event.Call) -> CallModel:
    """Creates a `CallModel` with a single sample."""
    return CallModel(call.method_name, [call.request_size], [call.dependency])

  def add_sample(self, call: event.Call) -> None:
    """Adds a sample to the model."""
    if self.method_name != call.method_name:
      raise ValueError("method name mismatch")
    self.request_sizes.append(call.request_size)
    self.dependency_candidates.append(call.dependency)

  def pick_and_build_dependency(self) -> decision.Dependency:
    """Picks a dependency from the current list of candidates.

    This method picks the dependency with the largest number of occurrences,
    i.e. the most popular candidate.

    Returns:
      A `decision.Dependency` for this call.
    """
    # Partition the dependency candidates into equivalence classes. Refer to
    # `event.py` for equivalency definitions.
    dependency_classes = collections.defaultdict(list)
    for dep in self.dependency_candidates:
      dependency_classes[dep].append(dep)

    # Now select the dependency with the maximum number of occurrences.
    dependency_instances = max(dependency_classes.values(), key=len)
    assert dependency_instances, "call without dependency"
    deltas_ns = [dep.delta_ns for dep in dependency_instances]
    deltas_ns = utils.Ecdf.from_data(deltas_ns)
    match dependency_instances[0]:
      case event.DependencyOnStart():
        dependency = decision.DependencyOnStart(deltas_ns)
      case event.DependencyOnCall(event_id=event_id, method_name=method_name):
        dependency = decision.DependencyOnCall(event_id, method_name, deltas_ns)
      case _:
        raise TypeError(
            "invalid Dependency type"
            f" {dependency_instances[0].__class__.__name__}"
        )
    return dependency

  def build(self) -> decision.CallModel:
    """Builds a `decision.CallModel`."""
    request_sizes = utils.Ecdf.from_data(self.request_sizes)
    dependency = self.pick_and_build_dependency()
    return decision.CallModel(self.method_name, request_sizes, dependency)


@attrs.define
class CallGroupModel:
  """A builder for `decision.CallGroupModel`.

  Attributes:
    members: The members of this call group, with duplicates allowed.
    calls: A mapping from each member to its `CallModel` builder.
  """

  members: tuple[str, ...]
  calls: dict[str, CallModel]

  def add_sample(self, ev: event.CallGroup) -> None:
    """Adds an `event.CallGroup` sample to the model."""
    for call in ev.calls:
      if call.method_name in self.calls:
        self.calls[call.method_name].add_sample(call)
      else:
        self.calls[call.method_name] = CallModel.with_sample(call)

  def build(self) -> decision.CallGroupModel:
    """Builds a `decision.CallGroupModel`."""
    calls = {method: call.build() for method, call in self.calls.items()}
    return decision.CallGroupModel(self.members, calls)


@attrs.define
class ReturnModel:
  """A builder for `decision.ReturnModel`.

  Attributes:
    response_sizes: A list of response size samples.
    deltas_ns: A list of delta samples in nanoseconds.
  """

  response_sizes: list[int]
  deltas_ns: list[int]

  def add_sample(self, ev: event.Return) -> None:
    """Adds an `event.Return` sample to the model."""
    self.response_sizes.append(ev.response_size)
    self.deltas_ns.append(ev.delta_ns)

  def build(self) -> decision.ReturnModel:
    """Builds a `decision.ReturnModel`."""
    response_sizes = utils.Ecdf.from_data(self.response_sizes)
    deltas_ns = utils.Ecdf.from_data(self.deltas_ns)
    return decision.ReturnModel(response_sizes, deltas_ns)


Builder = Sentinel | StartModel | CallGroupModel | ReturnModel


@attrs.define
class TreeNode:
  """A builder for `decision.TreeNode`.

  Each `TreeNode` builder contains a builder for a particular model.

  Attributes:
    builder: A model builder.
    count: The number of samples submitted to the builder.
    children: A mapping from `event.Key` to `TreeChild`.
  """

  builder: Builder
  count: int
  children: dict[event.Key, TreeNode]

  @classmethod
  def with_sample(cls, ev: event.Event) -> TreeNode:
    """Creates a `TreeNode` with a single sample."""
    match ev:
      case event.Start():
        builder = StartModel()
      case event.CallGroup(calls=calls):
        members = tuple(call.method_name for call in calls)
        call_builders = {}
        for call in calls:
          if call.method_name in call_builders:
            call_builders[call.method_name].add_sample(call)
          else:
            call_builders[call.method_name] = CallModel.with_sample(call)
        builder = CallGroupModel(members, call_builders)
      case event.Return(response_size=response_size, delta_ns=delta_ns):
        builder = ReturnModel([response_size], [delta_ns])
      case _:
        raise TypeError(f"invalid Event type {ev.__class__.__name__}")
    return TreeNode(builder, 1, {})

  def add_sample(self, ev: event.Event) -> None:
    """Adds an `event.Event` sample to the model."""
    self.count += 1
    self.builder.add_sample(ev)

  def build(self) -> decision.TreeNode:
    """Builds a `decision.TreeNode`."""
    model = self.builder.build()
    total_weight = 0
    weights = []
    children = []
    for child in self.children.values():
      total_weight += child.count
      weights.append(child.count)
      children.append(child.build())
    weights = [weight / total_weight for weight in weights]
    return decision.TreeNode(model, self.count, weights, children)


@attrs.define
class Tree:
  """The builder for a `decision.Tree`.

  Attributes:
    root: The root of the builder tree, which always a `Sentinel` builder.
  """

  root: TreeNode = attrs.field(
      init=False, factory=lambda: TreeNode(Sentinel(), 1, {})
  )

  def insert(self, seq: event.EventSequence) -> None:
    """Inserts an `event.EventSequence` into the tree."""
    cur = self.root
    for ev in seq:
      key = ev.key()
      if key in cur.children:
        cur.children[key].add_sample(ev)
      else:
        cur.children[key] = TreeNode.with_sample(ev)
      cur = cur.children[key]

  def build(self) -> decision.Tree:
    """Builds a `decision.Tree`."""
    return decision.Tree(self.root.build())

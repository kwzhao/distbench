"""This module defines an event-level view of RPCs.

An RPC is modeled as a sequence of events, beginning with a `Start` event and
terminating with a `Return` event. Optionally an RPC can issue other RPCs to
backend services, which are modeled with a `Call` class. Concurrent `Call`s are
grouped into `CallGroup` events.

Each event has an associated key, which allows similar events (events with the
same key) to be grouped together for aggregate downstream modeling.
"""

@attrs.frozen
class StartKey:
  """The key for `Start` events."""

  ...


@attrs.frozen
class CallGroupKey:
  """The key for `CallGroup` events.

  A `CallGroup` is a group of concurrent `Call`s, and it's identified by the
  method names of its constituents. Duplicates are allowed.

  Attributes:
    methods: The method names of the `Call`s in the `CallGroup`.
  """

  methods: tuple[str, ...]


@attrs.frozen
class ReturnKey:
  """The key for `Return` events."""

  ...


Key = StartKey | CallGroupKey | ReturnKey


@attrs.frozen
class Anchor:
  """An `Anchor` is the target of a dependency.

  Conceptually, an event `A` can be "anchored" to some another event `B`, which
  means that `A` cannot begin until after `B` finishes. Certain events in this
  module can be only be created by anchoring them to other events.

  Attributes:
    event_id: The event ID of the dependency.
    dataset_event: The dependency.
  """

  event_id: int
  dataset_event: dataset.Event


@attrs.frozen
class DependencyOnStart:
  """A dependency on a `Start` event.

  This expresses that an event depends on the start of server-side processing.

  Attributes:
    delta_ns: The elapsed time after the `Start`, in nanoseconds.
  """

  delta_ns: int = attrs.field(eq=False)


@attrs.frozen
class DependencyOnCall:
  """A dependency on a `Call` event.

  This expresses that an event depends on the completion of some backend call.

  Attributes:
    event_id: The event ID of the backend call.
    method_name: The method name of the backend call.
    delta_ns: The elapsed time after the backend call completed, in nanoseconds.
  """

  event_id: int
  method_name: str
  delta_ns: int = attrs.field(eq=False)


Dependency = DependencyOnStart | DependencyOnCall


@attrs.frozen
class Start:
  """A `Start` event.

  The `Start` event represents the beginning of server-side processing. A
  `Start` may be followed by some `CallGroup` events before the RPC terminates
  with a `Return` event.

  Attributes:
    event_id: The event ID.
  """

  event_id: int

  def key(self) -> Key:
    """Returns the key for this event."""
    return StartKey()


# TODO(kwzhao): priority level?
@attrs.frozen
class Call:
  """A backend RPC.

  `Call`s are not events themselves, but can be grouped together into concurrent
  `CallGroup` events. A `Call` always has a dependency, so it is created by
  anchoring a `dataset.CallStart` to some other `dataset.Event`.

  Attributes:
    method_name: The method name of the call.
    request_size: The request size.
    dependency: The timing dependency of the call.
  """

  method_name: str
  request_size: int
  dependency: Dependency

  @classmethod
  def anchored(cls, dataset_call: dataset.CallStart, anchor: Anchor) -> Call:
    """Creates a `Call` from a `dataset.CallStart` and an `Anchor`."""
    match anchor.dataset_event:
      case dataset.ServerStart():
        dependency = DependencyOnStart(dataset_call.offset_ns)
      case dataset.CallFinish(
          method_name=method_name, offset_ns=anchor_offset_ns
      ):
        delta_ns = dataset_call.offset_ns - anchor_offset_ns
        dependency = DependencyOnCall(anchor.event_id, method_name, delta_ns)
      case _:
        raise TypeError(
            f'invalid anchor event {anchor.dataset_event.__class__.__name__}'
        )
    return Call(dataset_call.method_name, dataset_call.request_size, dependency)


@attrs.define
class _CallGroupBuilder:
  """A builder for `CallGroup` events."""

  # Conceptually, a call group is a collection of RPCs that mutually overlap in
  # time. We build call groups as follows: keep track of all RPCs that have
  # begun but not yet terminated. For convenience, we call these "active calls".
  # All of the active calls clearly overlap. Once an RPC `r` terminates, `r` and
  # all currently active calls immediately form a call group. Any RPC `r'` that
  # starts after `r` completes cannot be in the same call group as `r`, since
  # they won't overlap.

  _active_calls: dict[int, Call] = attrs.field(init=False, factory=dict)

  def call_start(self, span_id: int, call: Call) -> None:
    """Signal that a call has started."""
    self._active_calls[span_id] = call

  def call_finish(self, span_id: int) -> Optional[list[Call]]:
    """Signal that a call has finished, optionally returning a call group.

    Args:
      span_id: The span ID of the completed call.

    Returns:
      A list of concurrent calls if the end of this call results in a new call
      group, or `None`.
    """
    if span_id in self._active_calls:
      concurrent_calls = list(self._active_calls.values())
      self._active_calls.clear()
      return concurrent_calls
    else:
      return None


@attrs.frozen
class CallGroup:
  """A `CallGroup` is a group of mutually concurrent `Call`s.

  Two calls `a` and `b` are concurrent iff their time intervals `[a.start,
  a.finish]` and `[b.start, b.finish]` have a nonempty intersection. WLOG,
  suppose `a.finish < b.start`. Then, we say `a` "happens before" `b`, or `a ->
  b`. Similarly, we say that a call group `A` happens before a call group `B`
  iff for all `a` in `A`, there exists some `b` in `B` such that `a -> b`.

  Attributes:
    event_id: The event ID of the call group.
    calls: The constituent calls.
  """

  event_id: int
  calls: list[Call]

  def key(self) -> Key:
    """Returns the key for this event."""
    methods = tuple(sorted(call.method_name for call in self.calls))
    return CallGroupKey(methods)

  def __iter__(self):
    return iter(self.calls)

  def __len__(self):
    return len(self.calls)


# TODO(kwzhao): Return dependencies are not well-defined.
@attrs.frozen
class Return:
  """A `Return` event.

  `Return` marks the end of server-side RPC processing. A `Return` can be
  created by anchoring a `dataset.Return` to some other `dataset.Event`.

  Attributes:
    event_id: The event ID.
    response_size: The RPC response size.
    delta_ns: The elapsed time after dependent events, in nanoseconds.
  """

  event_id: int
  response_size: int
  delta_ns: int

  @classmethod
  def anchored(
      cls, return_event: dataset.Return, anchor: Anchor, event_id: int
  ) -> Return:
    """Creates a `Return` from a `dataset.Return` and an `Anchor`."""
    match anchor.dataset_event:
      case dataset.ServerStart():
        delta_ns = return_event.offset_ns
      case dataset.CallFinish(offset_ns=anchor_offset_ns):
        delta_ns = return_event.offset_ns - anchor_offset_ns
      case _:
        raise TypeError(
            f'invalid anchor event {anchor.dataset_event.__class__.__name__}'
        )
    return Return(event_id, return_event.response_size, delta_ns)

  def key(self) -> Key:
    """Returns the key for this event."""
    return ReturnKey()


Event = Start | CallGroup | Return


@attrs.frozen
class EventSequence:
  """A sequence of events which models a `dataset.Rpc`.

  Attributes:
    method_name: The method name.
    start_event: The start of server-side processing.
    call_group_events: Groups of mutually concurrent backend RPCs.
    return_event: The end of server-side processing.
  """

  method_name: str
  start_event: Start
  call_group_events: list[CallGroup]
  return_event: Return

  # TODO(kwzhao): allow RPCs that return before children finish
  @classmethod
  def from_execution(cls, execution: dataset.MethodExecution) -> EventSequence:
    """Creates an `EventSequence` from a `dataset.MethodExecution`.

    Mutually concurrent backend RPCs are grouped into `CallGroup`s, and
    dependencies are inferred using a simple heuristic: if an RPC `b` starts
    immediately after another RPC `a` completes, then we assume `b` depends on
    `a`.

    Args:
      execution: A `dataset.MethodExecution`.

    Returns:
      An `EventSequence` modeling the RPC.
    """
    start_event = Start(event_id=0)
    call_group_events = []
    builder = _CallGroupBuilder()
    anchor = Anchor(0, execution.start_event)
    event_id = 1
    span2event = {}
    for child_event in execution.child_events:
      match child_event:
        case dataset.CallStart(span_id=span_id):
          call = Call.anchored(child_event, anchor)
          builder.call_start(span_id, call)
          span2event[span_id] = event_id
        case dataset.CallFinish(span_id=span_id):
          anchor = Anchor(span2event[span_id], child_event)
          maybe_calls = builder.call_finish(span_id)
          if maybe_calls is not None:
            calls = maybe_calls
            call_group_events.append(CallGroup(event_id, calls))
            event_id += 1
        case _:
          raise TypeError(
              f'invalid ChildEvent {child_event.__class__.__name__}'
          )
    return_event = Return.anchored(execution.return_event, anchor, event_id)
    return EventSequence(
        execution.method_name, start_event, call_group_events, return_event
    )

  @classmethod
  def from_raw_sequence(
      cls, method_name: str, seq: Sequence[Event]
  ) -> EventSequence:
    """Creates an `EventSequence` from a raw sequence of `Event`s.

    Valid `EventSequence`s must begin with a `Start` and terminate with a
    `Return`. Optionally, there can be intervening `CallGroup` events.

    Args:
      method_name: The method name of the event sequence.
      seq: A sequence of `Event`s.

    Returns:
      An `EventSequence`.

    Raises:
      A `ValueError` if the raw sequence doesn't form a valid event sequence.
    """
    # Pattern match against the given sequence. A valid sequence must have a
    # `Start` event as its first element and a `Return` event as its last. The
    # intervening elements, if any, should be `CallGroup` events. Anything else
    # is considered invalid.
    match seq:
      case [Start(), *call_group_events, Return()] if all(
          isinstance(e, CallGroup) for e in call_group_events
      ):
        return EventSequence(
            method_name, seq[0], list(call_group_events), seq[-1]
        )
      case _:
        raise ValueError('invalid raw event sequence')

  def keys(self) -> tuple[Key, ...]:
    """Returns all event keys for this sequence."""
    return tuple(ev.key() for ev in self)

  def call_groups(self) -> Iterable[CallGroup]:
    """Returns all call groups in this sequence."""
    return iter(self.call_group_events)

  def calls(self) -> Iterable[Call]:
    """Returns all calls in this sequence."""
    for call_group in self.call_groups():
      yield from call_group

  def __len__(self):
    return len(self.call_group_events) + 2  # start and return

  def __iter__(self):
    yield self.start_event
    yield from self.call_group_events
    yield self.return_event


@attrs.frozen
class EventSequenceTree:
  """A tree of `EventSequence`s.

  Attributes:
    sequence: The `EventSequence` of this node.
    children: The child `EventSequenceTree`s.
  """

  sequence: EventSequence
  children: list[EventSequenceTree]

  @classmethod
  def from_execution_tree(
      cls, rpc_tree: dataset.MethodExecutionTree
  ) -> EventSequenceTree:
    """Creates an `EventSequenceTree` from a `dataset.RpcTree`."""
    sequence = EventSequence.from_execution(rpc_tree.execution)
    children = [
        EventSequenceTree.from_execution_tree(child)
        for child in rpc_tree.children
    ]
    return EventSequenceTree(sequence, children)

  def keys(self) -> ...:
    """Returns all event keys in the tree as a tuple."""
    return (
        self.sequence.keys(),
        tuple(child.keys() for child in self.children),
    )

  def size(self) -> int:
    """Returns the number of event sequences in the tree."""
    return 1 + sum(child.size() for child in self.children)

  def call_tree(self) -> dataset.CallTree:
    """Returns a `CallTree` corresponding to this tree."""
    return dataset.CallTree(
        self.sequence.method_name,
        [child.call_tree() for child in self.children],
    )

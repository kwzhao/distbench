"""Translates stochastic decision trees into Distbench specifications.

This module defines a `Builder` class that, given a `system.Model`, generates
Distbench specifications for any method as well as the transitive closure of its
dependencies:

  ```python
  system = system.Model.from_trace(trace, rp.DecisionTree)
  builder = distbench.Builder(system)
  builder.add_method(method_name)
  builder.build()
  ```
"""

from __future__ import annotations

import attrs
import collections
from typing import Iterable, Mapping, Optional

from mimetic import system, method, utils
from mimetic.model import decision

# Assuming these are proto imports that would be available in the environment
# If these don't exist, they would need to be added to the project
import traffic_config_pb2
import joint_distribution_pb2

_DUMMY_PAYLOAD_NAME = 'dummy_payload'
_ROOT_REQUEST_PAYLOAD_NAME = 'root_request_payload'
_ROOT_REQUEST_PAYLOAD_SIZE = 1024


def _mk_dummy_payload() -> traffic_config_pb2.PayloadSpec:
  return traffic_config_pb2.PayloadSpec(name=_DUMMY_PAYLOAD_NAME, size=0)


def _mk_root_request_payload() -> traffic_config_pb2.PayloadSpec:
  return traffic_config_pb2.PayloadSpec(
      name=_ROOT_REQUEST_PAYLOAD_NAME, size=_ROOT_REQUEST_PAYLOAD_SIZE
  )


def _service_name(method_name: str) -> str:
  match method_name.split('.'):
    case [service_name, _]:
      return service_name
    case _:
      return method_name


def _client_name(method_name: str) -> str:
  return f'client({method_name})'


def _poisson_arrival_name(method_names: Iterable[str]) -> str:
  return f'poisson({"__".join(method_names)})'


def _mixed_rpc_name(method_names: Iterable[str]) -> str:
  return f'mixed_rpc({"__".join(method_names)})'


def _path_name(path: list[int]) -> str:
  return ':'.join(str(x) for x in path)


def _call_action_name(src: str, path: list[int], dst: str, index: int) -> str:
  return f'{src}|{_path_name(path)}|{dst}|{index}'


def _return_action_name(src: str, path: list[int]) -> str:
  return f'return({src}|{_path_name(path)})'


def _predicate_name(path: list[int]) -> str:
  return f'{_path_name(path)}'


def _request_payload_name(
    src: str, path: list[int], dst: str, index: int
) -> str:
  return f'request({_call_action_name(src, path, dst, index)})'


def _request_size_distribution_name(
    src: str, path: list[int], dst: str, index: int
) -> str:
  return _request_payload_name(src, path, dst, index)


def _response_payload_name(src: str, path: list[int]) -> str:
  return f'response({_return_action_name(src, path)})'


def _response_size_distribution_name(src: str, path: list[int]) -> str:
  return _response_payload_name(src, path)


def _delay_distribution_name(action_name: str) -> str:
  return f'delay({action_name})'


def _cdf_points(
    dist: utils.Ecdf,
) -> list[joint_distribution_pb2.CdfPoint]:
  """Translates distributions into Distbench's `CdfPoint`s."""
  points = {}
  for x, y in dist.points():
    points[int(x)] = y
  cdf_points = [
      joint_distribution_pb2.CdfPoint(cdf=y, value=x)
      for x, y in sorted(points.items())
  ]
  return cdf_points


@attrs.frozen
class _TraversalCtx:
  method_name: str
  path: list[int] = attrs.field(factory=list)
  predicates: list[str] = attrs.field(factory=list)


@attrs.define
class _TraversalAcc:
  """The accumulated state of a tree traversal.

  Attributes:
    actions: A mapping from tree path to a list of actions. Each integer in the
      tree path is a child index.
    rpc_caller2method: A set of caller/method pairs for backend calls.
    predicate_probabilities: Predicate probabilities.
    payload_descriptions: Payload descriptions.
    size_distribution_configs: Size distributions.
    delay_distribution_configs: Delay distributions.
    new_method_names: A set of method names that were encountered during
      traversal.
  """

  actions: collections.defaultdict[
      tuple[int, ...], list[traffic_config_pb2.Action]
  ] = attrs.field(init=False, factory=lambda: collections.defaultdict(list))
  rpc_caller2method: set[tuple[str, str]] = attrs.field(init=False, factory=set)
  predicate_probabilities: dict[str, float] = attrs.field(
      init=False, factory=dict
  )
  payload_descriptions: list[traffic_config_pb2.PayloadSpec] = attrs.field(
      init=False, factory=list
  )
  size_distribution_configs: list[joint_distribution_pb2.DistributionConfig] = (
      attrs.field(init=False, factory=list)
  )
  delay_distribution_configs: list[
      joint_distribution_pb2.DistributionConfig
  ] = attrs.field(init=False, factory=list)
  new_method_names: set[str] = attrs.field(init=False, factory=set)

  def iter_actions(self) -> Iterable[traffic_config_pb2.Action]:
    for lst in self.actions.values():
      yield from lst


def _process_call_group(
    call_group: decision.CallGroupModel, ctx: _TraversalCtx, acc: _TraversalAcc
) -> None:
  """Processes a `CallGroupModel`, emitting and accumulating protobufs."""
  for dst, ix in call_group.uniqueified_members():
    call = call_group.calls[dst]
    # Record this RPC src/dst pair.
    acc.rpc_caller2method.add((_service_name(ctx.method_name), dst))

    # Generate a request size distribution.
    size_distribution = joint_distribution_pb2.DistributionConfig(
        name=_request_size_distribution_name(
            ctx.method_name, ctx.path, dst, ix
        ),
        cdf_points=_cdf_points(call.request_sizes),
        field_names=['payload_size'],
    )
    acc.size_distribution_configs.append(size_distribution)

    # Generate a payload description for the request payload override.
    payload_description = traffic_config_pb2.PayloadSpec(
        name=_request_payload_name(ctx.method_name, ctx.path, dst, ix),
        size_distribution_name=size_distribution.name,
    )
    acc.payload_descriptions.append(payload_description)

    # Generate the dependency.
    deps = []
    if isinstance(call.dependency, decision.DependencyOnCall):
      dep_lvl = call.dependency.level
      dep_dst = call.dependency.method_name
      dep_path = ctx.path[: dep_lvl + 1]
      deps.append(_call_action_name(ctx.method_name, dep_path, dep_dst, 0))

    # Generate the delay distribution
    action_name = _call_action_name(ctx.method_name, ctx.path, dst, ix)
    delay_distribution = joint_distribution_pb2.DistributionConfig(
        name=_delay_distribution_name(action_name),
        cdf_points=_cdf_points(call.dependency.deltas_ns),
        field_names=['action_delay_ns'],
    )
    acc.delay_distribution_configs.append(delay_distribution)

    # Generate the action for this call.
    action = traffic_config_pb2.Action(
        name=action_name,
        dependencies=deps,
        rpc_name=dst,
        predicates=ctx.predicates,
        request_payload_override=payload_description.name,
        delay_distribution_name=delay_distribution.name,
    )
    acc.actions[tuple(ctx.path)].append(action)

    # `call_method_name` will need to be visited later.
    acc.new_method_names.add(dst)


def _process_return(
    ret: decision.ReturnModel, ctx: _TraversalCtx, acc: _TraversalAcc
) -> None:
  """Processes a `ReturnModel`, emitting and accumulating protobufs."""
  # Generate a response size distribution.
  size_distribution = joint_distribution_pb2.DistributionConfig(
      name=_response_size_distribution_name(ctx.method_name, ctx.path),
      cdf_points=_cdf_points(ret.response_sizes),
      field_names=['payload_size'],
  )
  acc.size_distribution_configs.append(size_distribution)

  # Generate a payload description for the response payload override.
  payload_description = traffic_config_pb2.PayloadSpec(
      name=_response_payload_name(ctx.method_name, ctx.path),
      size_distribution_name=size_distribution.name,
  )
  acc.payload_descriptions.append(payload_description)

  # Generate dependencies, omitting transitive ones.
  transitive_deps = set()
  deps = []
  for i in range(len(ctx.path) - 1, 0, -1):  # iterating backward
    path = tuple(ctx.path[:i])
    for action in acc.actions[path]:
      if action.name not in transitive_deps:
        deps.append(action.name)
      transitive_deps.update(action.dependencies)

  # Generate the delay distribution
  action_name = _return_action_name(ctx.method_name, ctx.path)
  delay_distribution = joint_distribution_pb2.DistributionConfig(
      name=_delay_distribution_name(action_name),
      cdf_points=_cdf_points(ret.deltas_ns),
      field_names=['action_delay_ns'],
  )
  acc.delay_distribution_configs.append(delay_distribution)

  # Generate the action.
  action = traffic_config_pb2.Action(
      name=action_name,
      dependencies=deps,
      predicates=ctx.predicates,
      send_response_when_done=True,
      response_payload_override=payload_description.name,
      delay_distribution_name=delay_distribution.name,
  )
  acc.actions[tuple(ctx.path)].append(action)


def _traverse_node(
    node: decision.TreeNode, ctx: _TraversalCtx, acc: _TraversalAcc
) -> None:
  """Traverses a `decision.TreeNode`, accumulating results."""

  # Process the model for this level.
  match node.model:
    case decision.CallGroupModel():
      _process_call_group(node.model, ctx, acc)
    case decision.ReturnModel():
      _process_return(node.model, ctx, acc)
    case _:
      pass

  # Emit predicates for the next level and recurse.
  cum_weights = 0.0
  cum_predicates = []
  for i, (weight, child) in enumerate(node.weights_and_children()):
    child_path = ctx.path + [i]
    predicate = _predicate_name(child_path)
    probability = weight / (1 - cum_weights)
    predicates = cum_predicates.copy()
    if probability < 1.0:
      predicates.append(predicate)
    child_ctx = _TraversalCtx(
        method_name=ctx.method_name,
        path=child_path,
        predicates=predicates,
    )
    if probability < 1.0:
      acc.predicate_probabilities[predicate] = probability
      cum_weights += weight
      cum_predicates.append(f'!{predicate}')
    _traverse_node(child, child_ctx, acc)


def _traverse(tree: method.DecisionTree) -> _TraversalAcc:
  """Traverses a `rp.DecisionTree` and returns accumulated results."""
  ctx = _TraversalCtx(
      method_name=tree.method_name,
  )
  acc = _TraversalAcc()
  _traverse_node(tree.decision_tree.root, ctx, acc)
  return acc


@attrs.define
class MethodParams:
  """Method parameters for mixed Poisson arrivals."""

  weight: float
  request_sizes: utils.Ecdf


@attrs.define
class Builder:
  """Builds a `DistributedSystemDescription`.

  See the module-level documentation for example usage.
  """

  _model: system.Model[method.DecisionTree]
  _visited_method_names: set[str] = attrs.field(init=False, factory=set)

  _service_names: set[str] = attrs.field(init=False, factory=set)
  _action_lists: list[traffic_config_pb2.ActionList] = attrs.field(
      init=False, factory=list
  )
  _actions: list[traffic_config_pb2.Action] = attrs.field(
      init=False, factory=list
  )
  _rpc_descriptions: list[traffic_config_pb2.RpcSpec] = attrs.field(
      init=False, factory=list
  )
  _rpc_method2callers: collections.defaultdict[str, set[str]] = attrs.field(
      init=False, factory=lambda: collections.defaultdict(set)
  )
  _payload_descriptions: list[traffic_config_pb2.PayloadSpec] = attrs.field(
      init=False,
      factory=lambda: [_mk_dummy_payload(), _mk_root_request_payload()],
  )
  _distribution_config: list[joint_distribution_pb2.DistributionConfig] = (
      attrs.field(init=False, factory=list)
  )
  _size_distribution_configs: list[
      joint_distribution_pb2.DistributionConfig
  ] = attrs.field(init=False, factory=list)
  _delay_distribution_configs: list[
      joint_distribution_pb2.DistributionConfig
  ] = attrs.field(init=False, factory=list)

  @classmethod
  def with_model(cls, model: system.Model[method.DecisionTree]) -> Builder:
    """Creates a new `Builder` with the given `system.Model`.

    Because of how Distbench interprets forward slashes, we prohibit method
    names from containing them. Users are expected to sanitize their traces
    before constructing models from them if their intention is to generate
    Distbench specs.

    Args:
      model: The system model.

    Returns:
      A new `distbench.Builder`.
    """
    for method_name in model.method_models:
      if '/' in method_name:
        raise ValueError(
            f'Method name {method_name} contains a forward slash. Please'
            ' sanitize your traces first.'
        )
    return Builder(model)

  def add_method(self, method_name: str) -> None:
    """Adds a method to the builder.

    This will cause the builder to emit specifications for the specified remote
    procedure as well as the transitive closure of all of its backend
    dependencies.

    Args:
      method_name: The method name to add.
    """
    if method_name not in self._model.method_models:
      raise ValueError(f'Unknown method name: {method_name}.')
    if method_name in self._visited_method_names:
      return

    # Traverse the stochastic decision tree and accumulate intermediate results.
    acc = _traverse(self._model.method_models[method_name])
    action_list = traffic_config_pb2.ActionList(
        name=method_name,
        action_names=[action.name for action in acc.iter_actions()],
        predicate_probabilities=acc.predicate_probabilities,
    )

    # Update the build state with the accumulated traversal results.
    self._service_names.add(_service_name(method_name))
    self._action_lists.append(action_list)
    self._actions.extend(acc.iter_actions())
    for src, dst in acc.rpc_caller2method:
      self._rpc_method2callers[dst].add(src)
    self._payload_descriptions.extend(acc.payload_descriptions)
    self._size_distribution_configs.extend(acc.size_distribution_configs)
    self._delay_distribution_configs.extend(acc.delay_distribution_configs)
    self._visited_method_names.add(method_name)

    # Recursively add specifications for backend dependencies.
    for child_method_name in acc.new_method_names:
      self.add_method(child_method_name)

  def add_arrival(
      self,
      method_name: str,
      qps: int,
      request_sizes: utils.Ecdf,
      duration_secs: int,
  ) -> None:
    """Adds a root arrival model to the builder.

    This will cause the builder to emit root requests for the specified remote
    procedure. Currently, this method only emits root requests with Poisson
    arrivals.

    Args:
      method_name: The method name.
      qps: The QPS of the root arrivals.
      request_sizes: The distribution of request sizes.
      duration_secs: The total duration of the arrival process in seconds.

    Raises:
      ValueError: If the specified method does not have an arrival model.
    """
    if method_name not in self._model.arrival_models:
      raise ValueError(f'Method {method_name} does not have an arrival model.')
    rpc = traffic_config_pb2.RpcSpec(
        name=method_name,
        client=[_client_name(method_name)],
        server=_service_name(method_name),
        request_payload_name=_DUMMY_PAYLOAD_NAME,
        response_payload_name=_DUMMY_PAYLOAD_NAME,
        fanout_filter='round_robin',
    )
    iterations = traffic_config_pb2.Iterations(
        max_duration_us=duration_secs * 1000000,
        open_loop_interval_ns=int(1e9 / qps),
        open_loop_interval_distribution='exponential',
    )
    size_distribution = joint_distribution_pb2.DistributionConfig(
        name=_request_size_distribution_name('root', [0], method_name, 0),
        cdf_points=_cdf_points(request_sizes),
        field_names=['payload_size'],
    )
    payload_description = traffic_config_pb2.PayloadSpec(
        name=_request_payload_name('root', [0], method_name, 0),
        size_distribution_name=size_distribution.name,
    )
    action = traffic_config_pb2.Action(
        name=_client_name(method_name),
        iterations=iterations,
        rpc_name=method_name,
        request_payload_override=payload_description.name,
    )
    action_list = traffic_config_pb2.ActionList(
        name=_client_name(method_name),
        action_names=[action.name],
    )

    self._service_names.add(_client_name(method_name))
    self._action_lists.append(action_list)
    self._actions.append(action)
    self._payload_descriptions.append(payload_description)
    self._size_distribution_configs.append(size_distribution)
    self._rpc_descriptions.append(rpc)

  def add_mixed_poisson_arrivals(
      self, method_params: dict[str, MethodParams], qps: int, duration_secs: int
  ):
    """Adds a mixed Poisson arrival model to the builder.

    This causes the builder to emit root requests for the specified methods at a
    given QPS, where the frequency of each method is determined by its weight.

    Args:
      method_params: A mapping from method name to its parameters.
      qps: The QPS of the root arrivals.
      duration_secs: The total duration of the arrival process in seconds.
    """
    method_names = list(method_params.keys())
    rpcs = []
    mixed_rpc_actions = []
    cum_weights = 0.0
    cum_predicates = []
    predicate_probabilities = {}
    size_distributions = []
    payload_descriptions = []
    for method_name, params in method_params.items():
      rpc = traffic_config_pb2.RpcSpec(
          name=method_name,
          client=[_poisson_arrival_name(method_names)],
          server=_service_name(method_name),
          request_payload_name=_DUMMY_PAYLOAD_NAME,
          response_payload_name=_DUMMY_PAYLOAD_NAME,
          fanout_filter='round_robin',
      )
      rpcs.append(rpc)

      # Compute predicates.
      predicate = _client_name(method_name)
      probability = params.weight / (1 - cum_weights)
      predicates = cum_predicates.copy()
      if probability < 1.0:
        predicates.append(predicate)
        predicate_probabilities[predicate] = probability
        cum_weights += params.weight
        cum_predicates.append(f'!{predicate}')

      # Add request size distribution.
      size_distribution = joint_distribution_pb2.DistributionConfig(
          name=_request_size_distribution_name('root', [0], method_name, 0),
          cdf_points=_cdf_points(params.request_sizes),
          field_names=['payload_size'],
      )
      size_distributions.append(size_distribution)
      payload_description = traffic_config_pb2.PayloadSpec(
          name=_request_payload_name('root', [0], method_name, 0),
          size_distribution_name=size_distribution.name,
      )
      payload_descriptions.append(payload_description)

      action = traffic_config_pb2.Action(
          name=_client_name(method_name),
          rpc_name=method_name,
          predicates=predicates,
          request_payload_override=payload_description.name,
      )
      mixed_rpc_actions.append(action)

    mixed_rpc_action_list = traffic_config_pb2.ActionList(
        name=_mixed_rpc_name(method_names),
        action_names=[action.name for action in mixed_rpc_actions],
        predicate_probabilities=predicate_probabilities,
    )
    iterations = traffic_config_pb2.Iterations(
        max_duration_us=duration_secs * 1000000,
        open_loop_interval_ns=int(1e9 / qps),
        open_loop_interval_distribution='exponential',
    )
    arrival_action = traffic_config_pb2.Action(
        name=_poisson_arrival_name(method_names),
        iterations=iterations,
        action_list_name=mixed_rpc_action_list.name,
    )
    arrival_action_list = traffic_config_pb2.ActionList(
        name=_poisson_arrival_name(method_names),
        action_names=[arrival_action.name],
    )
    self._service_names.add(_poisson_arrival_name(method_names))
    self._action_lists.extend([mixed_rpc_action_list, arrival_action_list])
    self._actions.extend(mixed_rpc_actions + [arrival_action])
    self._payload_descriptions.extend(payload_descriptions)
    self._size_distribution_configs.extend(size_distributions)
    self._rpc_descriptions.extend(rpcs)

  def build(
      self, replicas: Optional[Mapping[str, int]] = None
  ) -> traffic_config_pb2.DistributedSystemDescription:
    """Builds a `DistributedSystemDescription`.

    By default, all services have a single replica, where a service is defined
    as a group of methods with the same prefix (e.g. `cache.get` and
    `cache.put`, where `cache` is the service name). If `replicas` is specified,
    its entries will override the default ones.

    Args:
      replicas: A mapping from service name to the number of replicas.

    Returns:
      A `DistributedSystemDescription`.
    """
    service2replicas = collections.defaultdict(lambda: 1)
    if replicas is not None:
      for service_name, count in replicas.items():
        service2replicas[service_name] = count
    for dst, srcs in self._rpc_method2callers.items():
      rpc = traffic_config_pb2.RpcSpec(
          name=dst,
          client=srcs,
          server=_service_name(dst),
          request_payload_name=_DUMMY_PAYLOAD_NAME,
          response_payload_name=_DUMMY_PAYLOAD_NAME,
          fanout_filter='round_robin',
      )
      self._rpc_descriptions.append(rpc)
    services = []
    for service_name in self._service_names:
      services.append(
          traffic_config_pb2.ServiceSpec(
              name=service_name,
              count=service2replicas[service_name],
          )
      )
    return traffic_config_pb2.DistributedSystemDescription(
        name='system',
        services=services,
        action_lists=self._action_lists,
        actions=self._actions,
        rpc_descriptions=self._rpc_descriptions,
        payload_descriptions=self._payload_descriptions,
        size_distribution_configs=self._size_distribution_configs,
        delay_distribution_configs=self._delay_distribution_configs,
    )

  def build_all(
      self, replicas: Optional[Mapping[str, int]] = None
  ) -> traffic_config_pb2.DistributedSystemDescription:
    """Builds a `DistributedSystemDescription` for all methods.

    By default, all services have a single replica, where a service is defined
    as a group of methods with the same prefix (e.g. `cache.get` and
    `cache.put`, where `cache` is the service name). If `replicas` is specified,
    its entries will override the default ones.

    Args:
      replicas: A mapping from service name to the number of replicas.

    Returns:
      A `DistributedSystemDescription`.
    """
    for method_name in self._model.method_models:
      self.add_method(method_name)
    return self.build(replicas)

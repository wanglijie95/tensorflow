# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

# Picked a long key value to minimize the chance of collision with user defined
# collection keys.
GLOBAL_STEP_READ_KEY = 'global_step_read_op_cache'


# TODO(drpng): remove this after legacy uses are resolved.
write_graph = graph_io.write_graph


@tf_export('train.global_step')
def global_step(sess, global_step_tensor):
  """Small helper to get the global step.

  ```python
  # Creates a variable to hold the global_step.
  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
  # Creates a session.
  sess = tf.Session()
  # Initializes the variable.
  print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))

  global_step: 10
  ```

  Args:
    sess: A TensorFlow `Session` object.
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  """
  if context.executing_eagerly():
    return int(global_step_tensor.numpy())
  return int(sess.run(global_step_tensor))


@tf_export('train.get_global_step')
def get_global_step(graph=None):
  """Get the global step tensor.

  The global step tensor must be an integer variable. We first try to find it
  in the collection `GLOBAL_STEP`, or by name `global_step:0`.

  Args:
    graph: The graph to find the global step in. If missing, use default graph.

  Returns:
    The global step variable, or `None` if none was found.

  Raises:
    TypeError: If the global step tensor has a non-integer type, or if it is not
      a `Variable`.
  """
  graph = graph or ops.get_default_graph()
  global_step_tensor = None
  global_step_tensors = graph.get_collection(ops.GraphKeys.GLOBAL_STEP)
  if len(global_step_tensors) == 1:
    global_step_tensor = global_step_tensors[0]
  elif not global_step_tensors:
    try:
      global_step_tensor = graph.get_tensor_by_name('global_step:0')
    except KeyError:
      return None
  else:
    logging.error('Multiple tensors in global_step collection.')
    return None

  assert_global_step(global_step_tensor)
  return global_step_tensor


@tf_export('train.create_global_step')
def create_global_step(graph=None):
  """Create global step tensor in graph.

  Args:
    graph: The graph in which to create the global step tensor. If missing,
      use default graph.

  Returns:
    Global step tensor.

  Raises:
    ValueError: if global step tensor is already defined.
  """
  graph = graph or ops.get_default_graph()
  if get_global_step(graph) is not None:
    raise ValueError('"global_step" already exists.')
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    if context.executing_eagerly():
      with ops.device('cpu:0'):
        return variable_scope.get_variable(
            ops.GraphKeys.GLOBAL_STEP,
            shape=[],
            dtype=dtypes.int64,
            initializer=init_ops.zeros_initializer(),
            trainable=False,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                         ops.GraphKeys.GLOBAL_STEP])
    return variable_scope.get_variable(
        ops.GraphKeys.GLOBAL_STEP,
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                     ops.GraphKeys.GLOBAL_STEP])


@tf_export('train.get_or_create_global_step')
def get_or_create_global_step(graph=None):
  """Returns and create (if necessary) the global step tensor.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    The global step tensor.
  """
  graph = graph or ops.get_default_graph()
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    global_step_tensor = create_global_step(graph)
  return global_step_tensor


@tf_export('train.assert_global_step')
def assert_global_step(global_step_tensor):
  """Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.

  Args:
    global_step_tensor: `Tensor` to test.
  """
  if not (isinstance(global_step_tensor, variables.Variable) or
          isinstance(global_step_tensor, ops.Tensor) or
          resource_variable_ops.is_resource_variable(global_step_tensor)):
    raise TypeError(
        'Existing "global_step" must be a Variable or Tensor: %s.' %
        global_step_tensor)

  if not global_step_tensor.dtype.base_dtype.is_integer:
    raise TypeError('Existing "global_step" does not have integer type: %s' %
                    global_step_tensor.dtype)

  if (global_step_tensor.get_shape().ndims != 0 and
      global_step_tensor.get_shape().is_fully_defined()):
    raise TypeError('Existing "global_step" is not scalar: %s' %
                    global_step_tensor.get_shape())


def _get_global_step_read(graph=None):
  """Gets global step read tensor in graph.

  Args:
    graph: The graph in which to create the global step read tensor. If missing,
      use default graph.

  Returns:
    Global step read tensor.

  Raises:
    RuntimeError: if multiple items found in collection GLOBAL_STEP_READ_KEY.
  """
  graph = graph or ops.get_default_graph()
  global_step_read_tensors = graph.get_collection(GLOBAL_STEP_READ_KEY)
  if len(global_step_read_tensors) > 1:
    raise RuntimeError('There are multiple items in collection {}. '
                       'There should be only one.'.format(GLOBAL_STEP_READ_KEY))

  if len(global_step_read_tensors) == 1:
    return global_step_read_tensors[0]
  return None


def _get_or_create_global_step_read(graph=None):
  """Gets or creates global step read tensor in graph.

  Args:
    graph: The graph in which to create the global step read tensor. If missing,
      use default graph.

  Returns:
    Global step read tensor if there is global_step_tensor else return None.
  """
  graph = graph or ops.get_default_graph()
  global_step_read_tensor = _get_global_step_read(graph)
  if global_step_read_tensor is not None:
    return global_step_read_tensor
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    return None
  # add 'zero' so that it will create a copy of variable as Tensor.
  with graph.as_default() as g, g.name_scope(None):
    with g.name_scope(global_step_tensor.op.name + '/'):
      # using initialized_value to ensure that global_step is initialized before
      # this run. This is needed for example Estimator makes all model_fn build
      # under global_step_read_tensor dependency.
      global_step_value = global_step_tensor.initialized_value() if isinstance(
          global_step_tensor, variables.Variable) else global_step_tensor
      global_step_read_tensor = global_step_value + 0
      ops.add_to_collection(GLOBAL_STEP_READ_KEY, global_step_read_tensor)
  return _get_global_step_read(graph)


def _increment_global_step(increment, graph=None):
  graph = graph or ops.get_default_graph()
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    raise ValueError(
        'Global step tensor should be created by '
        'tf.train.get_or_create_global_step before calling increment.')
  global_step_read_tensor = _get_or_create_global_step_read(graph)
  with graph.as_default() as g, g.name_scope(None):
    with g.name_scope(global_step_tensor.op.name + '/'):
      with ops.control_dependencies([global_step_read_tensor]):
        return state_ops.assign_add(global_step_tensor, increment)

@tf_export('train.get_recovery_clock')
def get_recovery_clock(graph=None):
  """Get the recovery_clock.

  We first try to find recovery_clock in the collection `RECOVERY_CLOCK`.

  Args:
    graph: The graph to find the recovery_clock in. If missing, use default graph.

  Returns:
    The Ref of the recovery_clock.

  Raises:
    TypeError: If recovery_clock is not type `RecoveryClock`, raise TypeError
  """
  graph = graph or ops.get_default_graph()
  refs = graph.get_collection(ops.GraphKeys.RECOVERY_CLOCK)

  if len(refs) == 1:
    recovery_clock = refs[0]
    if not isinstance(recovery_clock, data_flow_ops.RecoveryClock):
      raise TypeError('Existing "recovery_clock" does not have RecoveryClock type: %s' %
                      type(recovery_clock))
    return recovery_clock
  elif not refs:
    return None
  else:
    logging.error('Multiple recovery clocks in recovery_clock collection.')
    return None

@tf_export('train.create_recovery_clock')
def create_recovery_clock(total_num_replicas, graph=None):
  """Create RecoveryClock in graph.

  Args:
    graph: The graph in which to create the RecoveryClock. If missing,
      use default graph.

  Returns:
    The Ref of `RecoveryClock`.

  Raises:
    ValueError: If RecoveryClock is already defined.
  """
  graph = graph or ops.get_default_graph()
  if get_recovery_clock(graph) is not None:
    raise ValueError('"RecoveryClock" already exists.')

  # if server_lib.get_cluster_spec() is None:
  #   raise ValueError(
  #     "The ClusterSpec is not be specified now, so can't create RecoveryClock")
  # if (server_lib.get_job_name() is None) or (server_lib.get_task_index() is None):
  #   raise ValueError(
  #     "The Server is not be created or specified, so can't create RecoveryClock")
  # if server_lib.get_job_name() != "worker":
  #   raise ValueError("This task is not worker, Maybe some thing wrong")
  # total_num_replicas = server_lib.get_num_tasks("worker")

  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    recovery_clock = data_flow_ops.RecoveryClock(
                                    total_num_replicas=total_num_replicas,
                                    shared_name="recovery_clock")
    graph.add_to_collection(ops.GraphKeys.RECOVERY_CLOCK, recovery_clock)
    return recovery_clock

@tf_export('train.get_or_create_recovery_clock')
def get_or_create_recovery_clock(total_num_replicas, graph=None):
  """Returns and create (if necessary) the recovery clock.

  Args:
    graph: The graph in which to create the recovery clock. If missing, use
      default graph.

  Returns:
    The local step tensor.
  """
  graph = graph or ops.get_default_graph()
  recovery_clock = get_recovery_clock(graph)
  if recovery_clock is None:
    recovery_clock = create_recovery_clock(total_num_replicas, graph)
  return recovery_clock

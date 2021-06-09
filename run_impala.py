# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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

"""Runs IMPALA on bsuite locally."""
from typing import Optional

import acme
import bsuite
import haiku as hk
import jax.numpy as jnp
import numpy as np
from absl import app
from absl import flags
from acme import specs
from acme import wrappers
from acme.agents.jax.impala.agent import IMPALAConfig, IMPALAFromConfig
from acme.jax import networks
from acme.tf import networks

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')
FLAGS = flags.FLAGS


class Network(hk.RNNCore):
  """A simple recurrent network for testing."""

  def __init__(self, num_actions: int):
    super().__init__(name='my_network')
    self._torso = hk.Sequential([
        lambda x: jnp.reshape(x, [np.prod(x.shape)]),
        hk.nets.MLP([50, 50]),
    ])
    self._core = hk.IdentityCore()
    self._head = networks.PolicyValueHead(num_actions)

  def __call__(self, inputs, state):
    embeddings = self._torso(inputs)
    embeddings, new_state = self._core(embeddings, state)
    logits, value = self._head(embeddings)
    return (logits, value), new_state

  def initial_state(self, batch_size: int):
    return self._core.initial_state(batch_size)



def main(_):
  # Create an environment and grab the spec.
  raw_environment = bsuite.load_and_record_to_csv(
      bsuite_id=(FLAGS.bsuite_id),
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )
  environment = wrappers.SinglePrecisionWrapper(raw_environment)
  spec = specs.make_environment_spec(environment)

  # Create the networks to optimize.
  def forward_fn(x, s):
      model = Network(spec.actions.num_values)
      return model(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
      model = Network(spec.actions.num_values)
      return model.initial_state(batch_size)

  def unroll_fn(inputs, state):
      model = Network(spec.actions.num_values)
      return hk.static_unroll(model, inputs, state)

  # We pass pure, Haiku-agnostic functions to the agent.
  forward_fn_transformed = hk.without_apply_rng(hk.transform(
      forward_fn,
      apply_rng=True))
  unroll_fn_transformed = hk.without_apply_rng(hk.transform(
      unroll_fn,
      apply_rng=True))
  initial_state_fn_transformed = hk.without_apply_rng(hk.transform(
      initial_state_fn,
      apply_rng=True))

  # Construct the agent.
  config = IMPALAConfig(
      sequence_length=3,
      sequence_period=3,
      batch_size=6,
  )

  agent = IMPALAFromConfig(
      environment_spec=spec,
      forward_fn=forward_fn_transformed.apply,
      initial_state_init_fn=initial_state_fn_transformed.init,
      initial_state_fn=initial_state_fn_transformed.apply,
      unroll_init_fn=unroll_fn_transformed.init,
      unroll_fn=unroll_fn_transformed.apply,
      config=config,
  )

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=environment.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)

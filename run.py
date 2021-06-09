# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Single-process IMPALA wiring."""

import threading
from typing import List

import jax
import optax
from bsuite import bsuite
from examples.impala import actor as actor_lib
from examples.impala import agent as agent_lib
from examples.impala import haiku_nets
from examples.impala import learner as learner_lib
from examples.impala import util
from tap import Tap


class Args(Tap):
    bsuite_id: str = "cartpole/0"
    results_dir: str = "/tmp/bsuite"  # CSV results directory.
    overwrite: bool = False  # Whether to overwrite csv results.
    action_repeat: int = 1
    batch_size: int = 2
    discount_factor: float = 0.99
    max_env_frames: int = 20000
    num_actors: int = 2
    unroll_length: int = 20


def run_actor(actor: actor_lib.Actor, stop_signal: List[bool]):
    """Runs an actor to produce num_trajectories trajectories."""
    while not stop_signal[0]:
        frame_count, params = actor.pull_params()
        actor.unroll_and_push(frame_count, params)


def run(args: Args):
    frames_per_iter = args.action_repeat * args.batch_size * args.unroll_length

    # Create an environment and grab the spec.
    def build_env():
        return bsuite.load_and_record_to_csv(
            bsuite_id=args.bsuite_id,
            results_dir=args.results_dir,
            overwrite=args.overwrite,
        )

    env_for_spec = build_env()
    # Construct the agent. We need a sample environment for its spec.
    num_actions = env_for_spec.action_spec().num_values
    agent = agent_lib.Agent(
        num_actions, env_for_spec.observation_spec(), haiku_nets.CatchNet
    )
    # Construct the optimizer.
    max_updates = args.max_env_frames / frames_per_iter
    opt = optax.rmsprop(1e-1, decay=0.99, eps=0.1)
    # Construct the learner.
    learner = learner_lib.Learner(
        agent,
        jax.random.PRNGKey(428),
        opt,
        args.batch_size,
        args.discount_factor,
        frames_per_iter,
        max_abs_reward=1.0,
        logger=util.AbslLogger(),  # Provide your own logger here.
    )
    # Construct the actors on different threads.
    # stop_signal in a list so the reference is shared.
    actor_threads = []
    stop_signal = [False]
    for i in range(args.num_actors):
        actor = actor_lib.Actor(
            agent,
            build_env(),
            args.unroll_length,
            learner,
            rng_seed=i,
            logger=util.AbslLogger(),  # Provide your own logger here.
        )
        actor_threads.append(
            threading.Thread(target=run_actor, args=(actor, stop_signal))
        )
    # Start the actors and learner.
    for t in actor_threads:
        t.start()
    learner.run(int(max_updates))
    # Stop.
    stop_signal[0] = True
    for t in actor_threads:
        t.join()

    return args.bsuite_id


if __name__ == "__main__":
    run(Args().parse_args())

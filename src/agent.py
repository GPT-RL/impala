import functools
from typing import Tuple

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
from examples.impala import agent
from examples.impala.agent import AgentOutput, Nest, NetFactory


class Agent(agent.Agent):
    def __init__(self, num_actions: int, obs_spec: Nest, net_factory: NetFactory):
        """Constructs an Agent object.

        Args:
          num_actions: Number of possible actions for the agent. Assumes a flat,
            discrete, 0-indexed action space.
          obs_spec: The observation spec of the environment.
          net_factory: A function from num_actions to a Haiku module representing
            the agent. This module should have an initial_state() function and an
            unroll function.
        """
        self._obs_spec = obs_spec
        net_factory = functools.partial(net_factory, num_actions)
        # Instantiate two hk.transforms() - one for getting the initial state of the
        # agent, another for actually initializing and running the agent.
        _, self._initial_state_apply_fn = hk.without_apply_rng(
            hk.transform(lambda batch_size: net_factory().initial_state(batch_size))
        )

        self._init_fn, self._apply_fn = hk.transform(
            lambda obs, state: net_factory().unroll(obs, state)
        )
        self._rng = hk.PRNGSequence(42)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng_key,
        params: hk.Params,
        timestep: dm_env.TimeStep,
        state: Nest,
    ) -> Tuple[AgentOutput, Nest]:
        """For a given single-step, unbatched timestep, output the chosen action."""
        # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.
        timestep = jax.tree_map(lambda t: t[None, None, ...], timestep)
        state = jax.tree_map(lambda t: t[None, ...], state)

        net_out, next_state = self._apply_fn(params, rng_key, timestep, state)
        # Remove the padding from above.
        net_out = jax.tree_map(lambda t: jnp.squeeze(t, axis=(0, 1)), net_out)
        next_state = jax.tree_map(lambda t: jnp.squeeze(t, axis=0), next_state)
        # Sample an action and return.
        action = hk.multinomial(rng_key, net_out.policy_logits, num_samples=1)
        action = jnp.squeeze(action, axis=-1)
        return AgentOutput(net_out.policy_logits, net_out.value, action), next_state

    def unroll(
        self,
        params: hk.Params,
        trajectory: dm_env.TimeStep,
        state: Nest,
    ) -> AgentOutput:
        """Unroll the agent along trajectory."""
        net_out, _ = self._apply_fn(params, next(self._rng), trajectory, state)
        return AgentOutput(net_out.policy_logits, net_out.value, action=[])

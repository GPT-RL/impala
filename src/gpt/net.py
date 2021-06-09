import dm_env
import haiku as hk
import jax.numpy as jnp
from examples.impala.haiku_nets import CatchNet, NetOutput

from gpt import GPT2
from gpt.pretrained import ModelSize, pretained_params


class Net(CatchNet):
    """A simple neural network for catch."""

    def __init__(self, num_actions, gpt_size: ModelSize):
        super().__init__(num_actions)
        self.gpt_size = gpt_size

    def __call__(self, x: dm_env.TimeStep, state):
        with pretained_params(self.gpt_size):
            gpt = GPT2(**self.gpt_size.config)
            torso_net = hk.Sequential(
                [
                    hk.Flatten(),
                    hk.Linear(gpt.embedding_size),
                    gpt,
                ]
            )

        torso_output = torso_net(x.observation)
        policy_logits = hk.Linear(self._num_actions)(torso_output)
        value = hk.Linear(1)(torso_output)
        value = jnp.squeeze(value, axis=-1)
        return NetOutput(policy_logits=policy_logits, value=value), state

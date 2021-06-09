import functools
from typing import Dict, Text, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from examples.impala import learner
from examples.impala import util
from jax.experimental import optimizers


def partition(params):
    return hk.data_structures.partition(lambda m, n, p: "/gpt2/" not in m, params)


class Learner(learner.Learner):
    def _loss(
        self,
        theta: hk.Params,
        fixed: hk.Params,
        trajectories: util.Transition,
    ) -> Tuple[jnp.ndarray, Dict[Text, jnp.ndarray]]:
        theta = hk.data_structures.merge(theta, fixed)
        return super()._loss(theta, trajectories)

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, params, opt_state, batch: util.Transition):
        """The actual update function."""
        theta, fixed = partition(params)
        (_, logs), grads = jax.value_and_grad(self._loss, has_aux=True)(
            theta, fixed, batch
        )

        grad_norm_unclipped = optimizers.l2_norm(grads)
        updates, updated_opt_state = self._opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        weight_norm = optimizers.l2_norm(params)
        logs.update(
            {
                "grad_norm_unclipped": grad_norm_unclipped,
                "weight_norm": weight_norm,
            }
        )
        return params, updated_opt_state, logs

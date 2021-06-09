from collections import defaultdict
from typing import DefaultDict, Dict

import chex
import haiku as hk
from jax import numpy as jnp

__all__ = ["multi_roll", "get_pretrained_initializers", "default_initializers"]


def multi_roll(a: chex.ArrayDevice, shifts: chex.ArrayDevice, axis=0):
    """Rolls each row or column of an array independently.
    Adapted from: https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    """
    # `axis` is the axis being rolled
    axis_size = a.shape[axis]

    inds = jnp.indices(a.shape)[axis]
    inds = (inds - jnp.expand_dims(shifts, axis)) % axis_size
    return jnp.take_along_axis(a, inds, axis=axis)


def get_pretrained_initializers(
    params: dict,
) -> Dict[str, Dict[str, hk.initializers.Initializer]]:
    return {
        k: (
            get_pretrained_initializers(v)
            if isinstance(v, dict)
            else hk.initializers.Constant(jnp.array(v))
        )
        for k, v in params.items()
    }


def default_initializers(
    w_init_scale: float,
) -> DefaultDict[str, Dict[str, hk.initializers.Initializer]]:
    return defaultdict(lambda: {"w": hk.initializers.VarianceScaling(w_init_scale)})

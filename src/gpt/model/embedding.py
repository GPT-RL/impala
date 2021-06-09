from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
from functools import partial


import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .util import get_pretrained_initializers

__all__ = [
    "shift_embeddings",
    "PositionalEmbedding",
    "sequence_lengths",
    "pad_sequences",
]


shift_embeddings = jax.vmap(partial(jnp.roll, axis=0), (None, 0))


@dataclass
class PositionalEmbedding(hk.Module):
    embedding_size: int
    context_size: int
    name: Optional[str] = "positional"

    def __call__(self, seq_lengths: chex.ArrayDevice) -> chex.ArrayDevice:
        positional_embeddings = hk.get_parameter(
            "embeddings",
            (self.context_size, self.embedding_size),
            init=hk.initializers.TruncatedNormal(stddev=0.02),
        )
        # Account for padding & cached key/values by shifting the positional embedding for
        # each sequence in the batch.
        return shift_embeddings(positional_embeddings, -seq_lengths)


def sequence_lengths(
    token_ids: chex.ArrayDevice, pad_token_id: int, padding_side: str = "left"
) -> jnp.array:
    """Given a padded matrix of token IDs `token_ids`, finds the length of each sequence.

    Args:
        token_ids (chex.ArrayDevice): A batch of token IDs. Shape: (batch, max_sequence_length)
        pad_token_id (int): The token ID used for padding

    Returns:
        jnp.array: The length of each sequence in the batch (as integers). Shape: (batch,)
    """
    batch_size, maxlen = token_ids.shape
    # Determine which rows have a padding token
    has_pad = jnp.array(
        [pad_token_id in token_ids[i] for i in range(batch_size)], dtype=jnp.int32
    )
    # If no rows contain a pad token, then all the sequences are maxlen.
    if not jnp.any(has_pad):
        return jnp.full((batch_size,), maxlen)
    if padding_side == "left":
        pad_token_positions = maxlen - np.argmax(token_ids != pad_token_id, axis=1)
    else:
        pad_token_positions = jnp.argmax(token_ids == pad_token_id, axis=1)
    # If a row contains padding tokens, then the seq length is the index of the first padding token.
    # Otherwise, the seq length == maxlen.
    return jnp.where(has_pad, pad_token_positions, maxlen)


def pad_sequences(
    x: chex.ArrayDevice,
    max_seq_length: int,
    seq_axis=1,
    padding_side: str = "left",
    **kwargs
) -> chex.ArrayDevice:
    pad_amount = max_seq_length - x.shape[seq_axis]
    if padding_side == "left":
        seq_axis_padding = (pad_amount, 0)
    else:
        seq_axis_padding = (0, pad_amount)
    pads = [seq_axis_padding if i == seq_axis else (0, 0) for i in range(len(x.shape))]
    return jnp.pad(x, pads, **kwargs)

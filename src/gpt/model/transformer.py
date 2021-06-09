from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku.initializers import VarianceScaling as VarianceScalingInit

from .embedding import pad_sequences


__all__ = [
    "causal_mask",
    "CausalSelfAttention",
    "DenseBlock",
    "layer_norm",
    "Transformer",
]


def causal_mask(source_len: int, dest_len: int) -> chex.ArrayDevice:
    i = jnp.arange(dest_len, dtype=jnp.int32)[:, None]
    j = jnp.arange(source_len, dtype=jnp.int32)
    mask = i >= j - source_len + dest_len
    return jnp.reshape(mask, (dest_len, source_len))


@dataclass
class CausalSelfAttention(hk.Module):
    """Multi-headed attention mechanism with a causal attention mask applied."""

    num_heads: int
    key_size: int
    w_init_scale: float
    query_size: int
    value_size: int
    model_size: int
    name: Optional[str] = None

    def __call__(
        self,
        query: chex.ArrayDevice,
        key: Optional[chex.ArrayDevice] = None,
        value: Optional[chex.ArrayDevice] = None,
        past_key_values: Optional[Tuple[chex.ArrayDevice, chex.ArrayDevice]] = None,
        mask: Optional[chex.ArrayDevice] = None,
        use_cache=False,
    ) -> chex.ArrayDevice:
        """Compute Multi-Head Attention.

        Args:
            query (chex.ArrayDevice): Query values for this timestep. Shape: (batch, sequence, head, head_feature).
            key (chex.ArrayDevice): Key values for this timestep. Shape: (batch, sequence, head, head_feature).
            value (chex.ArrayDevice): Value values for this timestep. Shape: (batch, sequence, head, head_feature).
            past_key_values (Optional[Tuple[chex.ArrayDevice, chex.ArrayDevice]], optional): Cached keys/values from previous timesteps. Shape: (batch, sequence, head, head_feature). Defaults to None.
            mask (Optional[chex.ArrayDevice], optional): Attention mask. Shape: (batch, 1, sequence, sequence) or (sequence, sequence). Defaults to None.
            use_cache (bool, optional): Whether or not to return the keys/values from the current timestep (so they can be re-used via `past_key_values`). Defaults to False.

        Returns:
            chex.ArrayDevice: the hidden state after computing multi-head attention.
            Optional[Tuple[chex.ArrayDevice, chex.ArrayDevice]]: if `use_cache` is `True`, this will be the `key` and `value` tensors
                (the `key` and `value` inputs concatenated with any cached keys/values passed to `past_key_values`). Otherwise, this is `None`.
        """
        key = key if key is not None else query
        value = value if value is not None else query

        query_heads = self._linear_projection(query, self.query_size, name="query")
        key_heads = self._linear_projection(key, self.key_size, name="key")
        value_heads = self._linear_projection(value, self.value_size, name="value")

        # Allow re-use of previously computed keys/values.
        if past_key_values is not None:
            past_keys, past_values = past_key_values
            # !!! Assumes past keys/values are left-padded
            # !!! to maximum sequence length.
            key_seq_len = key_heads.shape[1]
            key_heads = (
                jnp.roll(past_keys, -key_seq_len, axis=1)
                .at[:, -key_seq_len:]
                .set(key_heads)
            )
            val_seq_len = value_heads.shape[1]
            value_heads = (
                jnp.roll(past_values, -val_seq_len, axis=1)
                .at[:, -val_seq_len:]
                .set(value_heads)
            )

        present = None
        if use_cache:
            present = (key_heads, value_heads)

        attention_logits = jnp.einsum("bthd,bThd->bhtT", query_heads, key_heads)
        attention_logits = attention_logits * jax.lax.rsqrt(jnp.float32(self.key_size))

        # Get the attention-head's built-in attention mask.
        # (e.g. causal mask)
        _, _, destination_len, source_len = attention_logits.shape
        builtin_mask = causal_mask(source_len, destination_len)
        mask = mask * builtin_mask if mask is not None else builtin_mask
        if mask is not None:
            attention_logits -= 1e10 * (1.0 - mask)

        attention_weights = jax.nn.softmax(attention_logits)
        attention = jnp.einsum("bhtT,bThd->bthd", attention_weights, value_heads)

        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*attention.shape[:2], -1))
        output = hk.Linear(
            self.model_size, w_init=VarianceScalingInit(self.w_init_scale)
        )(attention_vec)
        return (output, present)

    @hk.transparent
    def _linear_projection(
        self, x: chex.ArrayDevice, head_size: int, name: Optional[str] = None
    ) -> chex.ArrayDevice:
        y = hk.Linear(
            self.num_heads * head_size,
            w_init=VarianceScalingInit(self.w_init_scale),
            name=name,
        )(x)
        return y.reshape((*x.shape[:2], self.num_heads, head_size))


@dataclass
class DenseBlock(hk.Module):
    """A 2-layer MLP which widens then narrows the input."""

    widening_factor: int = 4
    w_init_scale: Optional[float] = None
    name: Optional[str] = None

    def __call__(self, x: chex.ArrayDevice) -> chex.ArrayDevice:
        hiddens = x.shape[-1]
        x = hk.Linear(
            self.widening_factor * hiddens,
            w_init=VarianceScalingInit(self.w_init_scale),
            name="proj_in",
        )(x)
        x = jax.nn.gelu(x, approximate=False)
        return hk.Linear(
            hiddens, w_init=VarianceScalingInit(self.w_init_scale), name="proj_out"
        )(x)


def layer_norm(
    x: chex.ArrayDevice,
    pretrained: Optional[Dict[str, chex.ArrayDevice]] = None,
    name: Optional[str] = None,
) -> chex.ArrayDevice:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


@dataclass
class Transformer(hk.Module):
    """A transformer stack."""

    num_heads: int
    num_layers: int
    key_size: int
    dropout_rate: float
    max_seq_length: int
    init_scale: float
    name: Optional[str] = None

    def __call__(
        self,
        h: chex.ArrayDevice,
        past_key_values: Optional[
            List[Tuple[chex.ArrayDevice, chex.ArrayDevice]]
        ] = None,
        mask: Optional[chex.ArrayDevice] = None,
        is_training: bool = False,
        use_cache: bool = False,
    ) -> chex.ArrayDevice:
        """Connects the transformer.

        Args:
          h: Inputs. Shape: (batch, sequence, hidden_size).
          mask: Padding mask. Shape: (batch, sequence).
          is_training: Whether we're training or not.

        Returns:
          chex.ArrayDevice: The final hidden state. Shape: (batch, sequence, hidden_size)
        """
        dropout_rate = self.dropout_rate if is_training else 0.0
        if past_key_values is None:
            past_key_values = [None] * self.num_layers

        if mask is not None:
            mask = mask[:, None, None, :]

        # Note: names chosen to approximately match those used in the GPT-2 code;
        # see https://github.com/openai/gpt-2/blob/master/src/model.py.
        present_kv = []
        for i in range(self.num_layers):
            name = f"h{i}_ln_1"
            h_norm = layer_norm(h, name=name)
            past = past_key_values[i]
            name = f"h{i}_attn"
            h_attn, key_values = CausalSelfAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                query_size=self.key_size,
                value_size=self.key_size,
                model_size=self.key_size * self.num_heads,
                w_init_scale=self.init_scale,
                name=name,
            )(h_norm, past_key_values=past, mask=mask, use_cache=use_cache)
            if use_cache:
                if past is None:
                    # Jax's JIT compilation is shape-specific, so
                    # we left-pad the cached values along the
                    # sequence dimension to the max seq length.
                    # (That way the cached tensors have the same
                    # shape, regardless of the actual sequence length).
                    key_values = tuple(
                        pad_sequences(x, self.max_seq_length) for x in key_values
                    )
                # Store the computed keys/values so they can be reused.
                present_kv.append(key_values)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn

            name = f"h{i}_ln_2"
            h_norm = layer_norm(h, name=name)

            name = f"h{i}_mlp"
            h_dense = DenseBlock(w_init_scale=self.init_scale, name=name)(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense

        name = "ln_f"
        h = layer_norm(h, name=name)

        return h, (present_kv or None)

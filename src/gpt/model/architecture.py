# Architecture code (heavily) modified from: https://github.com/deepmind/dm-haiku/blob/master/examples/transformer/model.py
from dataclasses import dataclass
from typing import Optional, Tuple

import chex
import haiku as hk
import jax.numpy as jnp

from .embedding import PositionalEmbedding
from .transformer import Transformer

__all__ = ["GPT2"]


@dataclass
class GPT2(hk.Module):
    """A GPT2 architecture with ."""

    num_layers: int = 12
    num_heads: int = 12
    key_size: int = 64
    dropout_rate: float = 0.1
    embedding_size: int = 768
    vocabulary_size: int = 50257
    context_size: int = 1024
    eos_token_id: int = 50256
    name: str = "gpt2"
    init_scale: Optional[float] = None

    @hk.transparent
    def _init_embeddings(self) -> Tuple[hk.Module, hk.Module]:
        token_embeddings = hk.Embed(
            self.vocabulary_size,
            self.embedding_size,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="token",
        )
        positional_embeddings = PositionalEmbedding(
            self.embedding_size, self.context_size
        )
        return token_embeddings, positional_embeddings

    @hk.transparent
    def _init_transformer(self) -> hk.Module:
        return Transformer(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            max_seq_length=self.context_size,
            key_size=self.key_size,
            init_scale=self.init_scale or (2.0 / self.num_layers),
        )

    def token_embeddings(self) -> chex.ArrayDevice:
        return hk.get_parameter(
            f"token/embeddings",
            (self.vocabulary_size, self.embedding_size),
            init=jnp.zeros,
        )

    def __call__(self, obs: chex.ArrayDevice):
        transformer = self._init_transformer()
        obs = obs.reshape(obs.shape[0], -1, self.embedding_size)
        h, _ = transformer(obs, is_training=False, use_cache=False)
        return h[:, -1]

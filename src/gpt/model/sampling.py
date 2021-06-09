from dataclasses import field
from functools import partial
from typing import Callable, List, Optional, Tuple
from jax import numpy as jnp
import jax
import chex
import numpy as np
import haiku as hk

from .embedding import sequence_lengths, pad_sequences

__all__ = [
    "TokenSequence",
    "logits",
    "greedy",
    "next_token_logits",
    "top_p_filter",
    "top_k_filter",
]


@chex.dataclass
class TokenSequence:
    token_ids: chex.ArrayDevice = field(compare=False)
    # Log-probabilities of the subsequences of tokens in the sequence
    subsequence_lps: chex.ArrayDevice = field(
        default_factory=lambda: jnp.array([], dtype=jnp.float32), compare=False
    )
    log_prob: float = field(default=0.0, compare=True)

    def __post_init__(self):
        if self.subsequence_lps is not None:
            self.log_prob = jnp.mean(self.subsequence_lps)

    def __len__(self):
        return len(self.token_ids)

    def with_next_token(self, token_id: int, log_prob: float):
        return type(self)(
            token_ids=jnp.append(self.token_ids, token_id),
            subsequence_lps=jnp.append(self.subsequence_lps, log_prob),
        )


def greedy(
    n_steps: int,
    model_params: hk.Params,
    model_fn: Callable,
    embeddings: chex.ArrayDevice,
    start_token_ids: chex.ArrayDevice,
    start_seq_lengths: chex.ArrayDevice,
    mask: Optional[chex.ArrayDevice] = None,
    padding_side: str = "left",
    past_seq_axis: int = 1,
    seed: int = 1234,
):

    batch_size = start_token_ids.shape[0]
    sequences: List[List[Tuple[int, float]]] = [[] for _ in range(batch_size)]
    hidden_states = []
    rng_seq = hk.PRNGSequence(seed)
    cur_inputs = start_token_ids
    seq_lengths = start_seq_lengths
    past = None
    mask_shift = -1 if padding_side == "left" else 1
    for _ in range(n_steps):
        log_probs, h, past = decode(
            model_params,
            rng_seq,
            model_fn,
            embeddings,
            cur_inputs,
            seq_lengths,
            past,
            mask,
        )
        hidden_states.append(h)
        next_tokens = jnp.argmax(log_probs, axis=1)
        # Add the new tokens to their sequences.
        for s, t in zip(sequences, next_tokens):
            s.append((t, log_probs[t]))

        # Prepare for the next interation.
        seq_lengths += 1
        cur_inputs = next_tokens[:, None]

        # Adjust the mask to match the past if needed.
        if mask is not None:
            if past is not None:
                max_seq_len = past[0][0].shape[past_seq_axis]
                if mask.shape[1] < max_seq_len:
                    # Pad the mask along the sequence dimension to cover the
                    # maximum sequence length.
                    mask = pad_sequences(mask, max_seq_len, padding_side=padding_side)
            # Update the attention mask for the next iteration.
            mask = jnp.roll(mask, mask_shift, axis=1).at[:, -1].set(1)

    return sequences, hidden_states


def decode(
    params: hk.Params,
    rng_seq: hk.PRNGSequence,
    model_fn: Callable,
    embeddings: chex.ArrayDevice,
    token_ids: chex.ArrayDevice,
    seq_lengths: chex.ArrayDevice,
    past_key_values: Optional[chex.ArrayDevice],
    mask: Optional[chex.ArrayDevice],
    k: int = 0,
    p: float = 1.0,
):
    h, past_key_values = model_fn(
        params,
        next(rng_seq),
        token_ids,
        seq_lengths,
        past_key_values=past_key_values,
        mask=mask,
    )
    logits_ = logits(h, embeddings)
    logits_ = next_token_logits(logits_, seq_lengths)
    if k < 0:
        logits_ = top_k_filter(logits_, k)
    if p < 1.0:
        logits_ = top_p_filter(logits_, p)

    log_probs = jax.nn.log_softmax(logits_, axis=1)
    return log_probs, h, past_key_values


def logits(h: chex.ArrayDevice, embeddings: chex.ArrayDevice):
    """Given a hidden-state `h` (from a language model) and token embeddings `embeddings`,
    returns the logits for the next tokens.
    Args:
        h (chex.ArrayDevice): A hidden state from a language model. Shape: (batch, sequence, embedding_size)
        embeddings (chex.ArrayDevice): Token embeddings. Shape: (vocab_size, embedding_size)
    Returns:
        (chex.ArrayDevice): Logits for the next tokens. Shape: (batch, sequence, vocab_size)
    """
    batch_size, seq_len = h.shape[:2]
    n_vocab, n_emb = embeddings.shape
    h_flat = jnp.reshape(h, [-1, n_emb])
    flat_logits = jnp.einsum("ij,kj", h_flat, embeddings)
    return jnp.reshape(flat_logits, [batch_size, seq_len, n_vocab])


def next_token_logits(
    logits: chex.ArrayDevice, seq_lengths: chex.ArrayDevice
) -> chex.ArrayDevice:
    """Extract only the `logits` for the next token of each sequence in the batch.

    Args:
        logits (chex.ArrayDevice): Logits for the next tokens across all tokens in the sequence. Shape: (batch, sequence, vocab_size).
        seq_lengths (chex.ArrayDevice): An array of integers containing the length of each sequence in the batch. Shape: (batch,).

    Returns:
        chex.ArrayDevice: Logits for only the next tokens in each sequence. Shape: (batch, vocab_size)
    """
    return jnp.stack([logits[i, j] for i, j in enumerate(seq_lengths - 1)], axis=0)


def top_p_filter(
    logits: jnp.array, p: float, filter_value=-float("Inf"), min_tokens_to_keep=1
) -> jnp.array:
    """
    Filter a distribution of logits using nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    Adapted from: https://github.com/huggingface/transformers/blob/9856c9213dfe9f8355fe00dd6cd0fa1ceae4fa5a/src/transformers/generation_logits_process.py#L171
    """
    sorted_indices = jnp.argsort(logits, axis=1)[:, ::-1]
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=1)
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # Remove tokens with cumulative probability above the threshold, keeping at least `min_tokens_to_keep`
    inds_to_remove = jax.ops.index_update(
        cumulative_probs > p, jax.ops.index[:, :min_tokens_to_keep], False
    )
    batch_i, token_i = jnp.where(inds_to_remove)
    token_i = sorted_indices[batch_i, token_i]
    return jax.ops.index_update(logits, jax.ops.index[batch_i, token_i], filter_value)


def top_k_filter(logits, k: int, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    Filter a distribution of logits using top-k filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    Adapted from: https://github.com/huggingface/transformers/blob/9856c9213dfe9f8355fe00dd6cd0fa1ceae4fa5a/src/transformers/generation_logits_process.py#L213
    """
    k = min(max(k, min_tokens_to_keep), logits.shape[1])  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < jax.lax.top_k(logits, k=k)[0][..., -1, None]
    logits = jax.ops.index_update(logits, indices_to_remove, filter_value)

    return logits

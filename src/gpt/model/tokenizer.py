from ..pretrained import ModelSize
from transformers.tokenization_utils_fast import PaddingStrategy
from transformers import GPT2TokenizerFast
from typing import Tuple, Union
import os

# Disable tokenizer parallelism for HuggingFace Tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


__all__ = ["get_tokenizer"]


def get_tokenizer(
    size: ModelSize, padding_side: str = "left"
) -> Tuple[GPT2TokenizerFast, dict]:
    max_len = GPT2TokenizerFast.max_model_input_sizes[size.huggingface_name]
    config = {
        "padding": PaddingStrategy.LONGEST,
        "truncation": True,
        "max_length": max_len,
    }
    tokenizer = GPT2TokenizerFast.from_pretrained(
        size.huggingface_name,
        model_max_length=max_len,
        padding_side=padding_side,
        pad_token="<|endoftext|>",
    )
    return tokenizer, config

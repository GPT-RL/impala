from typing import Mapping, Optional
from dataclasses import dataclass, asdict

import json
import pickle
import os
import shutil
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from functools import lru_cache

import haiku as hk
import chex
import requests
from halo import Halo
from jax import numpy as jnp

from .convert import extract_tf_parameters, tf_to_haiku

__all__ = [
    "ModelSizes",
    "download_pretrained",
    "download_and_convert",
    "load_params",
    "pretrained_creator",
    "pretrained_params",
]

PRETRAINED_CACHE = Path(os.getenv("GPT_PRETRAINED_CACHE", "~/.cache/GPT"))


@dataclass
class GPTConfig:
    num_layers: int = 12
    num_heads: int = 12
    key_size: int = 64
    dropout_rate: float = 0.1
    embedding_size: int = 768
    vocabulary_size: int = 50257
    context_size: int = 1024
    eos_token_id: int = 50256
    init_scale: Optional[float] = None


class ModelSize(Enum):
    SMALL = "124M"
    MEDIUM = "355M"
    LARGE = "774M"
    XL = "1558M"

    @property
    def huggingface_name(self):
        return {
            "124M": "gpt2",
            "355M": "gpt2-medium",
            "774M": "gpt2-large",
            "1558M": "gpt2-xl",
        }[self.value]

    @property
    def config(self) -> dict:
        return asdict(
            {
                "124M": GPTConfig(num_layers=12, num_heads=12, embedding_size=768),
                "355M": GPTConfig(num_layers=24, num_heads=16, embedding_size=1024),
                "774M": GPTConfig(num_layers=36, num_heads=20, embedding_size=1280),
                "1558M": GPTConfig(num_layers=48, num_heads=25, embedding_size=1600),
            }[self.value]
        )


@lru_cache()
def download_pretrained(size: ModelSize, path: Path, chunk_size: int = 1000):
    """Download a pretrained model from OpenAI"""
    # Adapted from: https://github.com/openai/gpt-2/blob/master/download_model.py
    # Prepare the download path
    path = path.expanduser().resolve()
    if not path.exists():
        path.mkdir(parents=True)

    # Download model files from OpenAI
    files = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]
    spinner_message = f"Downloading GPT-2 {size.value}"
    spinner = Halo(text=spinner_message, spinner="dots")
    spinner.start()
    for filename in files:
        spinner.text = f"{spinner_message}: {filename}"
        r = requests.get(
            f"https://openaipublic.blob.core.windows.net/gpt-2/models/{size.value}/{filename}",
            stream=True,
        )
        with (path / filename).open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    spinner.stop_and_persist(
        symbol="⬇️".encode("utf-8"), text=f"GPT-2 {size.value} Downloaded"
    )


def download_and_convert(
    size: ModelSize, model_dir: Path, params_path: Path, hyperparams_path: Path
) -> dict:
    tf_dir = model_dir / "tf"
    if not tf_dir.exists():
        tf_dir.mkdir(parents=True)
        download_pretrained(size, tf_dir)

    # Convert downloaded parameters to a Haiku-compatible format.
    spinner = Halo(text="Converting TF Model to Haiku", spinner="dots")
    spinner.start()
    shutil.move(tf_dir / "hparams.json", hyperparams_path)
    params = tf_to_haiku(extract_tf_parameters(tf_dir))
    with params_path.open("wb") as f:
        pickle.dump(params, f)
    # Clean up
    shutil.rmtree(tf_dir)
    spinner.stop_and_persist(
        symbol="✨".encode("utf-8"), text=f"GPT-2 {size.value} Converted to Haiku"
    )
    return params


def load_params(size: ModelSize, directory: Path = PRETRAINED_CACHE, download=True):
    """Load a pretrained model from `directory`.
    If the model is not found and `download` is `True`, it
    will be downloaded from OpenAI and saved to the directory."""
    model_dir: Path = directory.expanduser().resolve() / size.value
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    params_path = model_dir / "params.pkl"
    hyperparams_path = model_dir / "hyperparams.json"
    if download and not (params_path.exists() and hyperparams_path.exists()):
        params = download_and_convert(size, model_dir, params_path, hyperparams_path)
    else:
        with params_path.open("rb") as f:
            params = pickle.load(f)

    with hyperparams_path.open("r") as f:
        hyperparams = json.load(f)

    return params, hyperparams


def pretrained_creator(params: hk.Params):
    def _creator(next_creator, shape, dtype, init, context):
        # Handle the case where GPT module is nested inside another
        # module.
        if context.module_name.startswith("gpt2/"):
            start_index = 0
            offset = 0
        else:
            start_index = context.module_name.find("/gpt2/")
            offset = 1
        if start_index >= 0:
            # Replace the default initializer with a constant initializer
            # containing the pretrained weights.
            module_name = context.module_name[start_index + offset :]
            init = hk.initializers.Constant(params[module_name][context.name])

        return next_creator(shape, dtype, init)

    return _creator


def pretained_params(size: ModelSize, **kwargs):
    params, _ = load_params(size, **kwargs)
    # Convert all parameters to Jax arrays.
    params = hk.data_structures.map(
        lambda module_name, name, value: jnp.array(value), params
    )
    return hk.experimental.custom_creator(pretrained_creator(params))

"""Utilities for converting pre-trained Tensorflow weights to a Haiku compatible format"""
import re
import os

import chex
import haiku as hk
import numpy as np

from collections import defaultdict
from pathlib import Path
from typing import Dict
from haiku._src.typing import Params


def extract_tf_parameters(path: Path) -> Dict[str, chex.ArrayNumpy]:
    """Extract parameters from a Tensorflow checkpoint."""
    # Silence TF warnings.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow.compat.v1 as tf

    with tf.Session() as sess:
        meta_path = (path / "model.ckpt.meta").resolve().as_posix()
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, (path / "model.ckpt").resolve().as_posix())

        names = []
        variables = []
        for v in [x for x in tf.global_variables() if x.name.startswith("model/")]:
            names.append(v.name)
            variables.append(v)
        values = sess.run(variables)
    return dict(zip(names, values))


def convert_attention(
    weights: chex.ArrayNumpy, biases: chex.ArrayNumpy
) -> Dict[str, Dict[str, chex.ArrayNumpy]]:
    """The TF version of GPT stores the query, key, and value weights/biases as a single tensor.
    This function extracts them as separate tensors."""
    split_w = np.split(weights, 3, axis=-1)
    split_b = np.split(biases, 3, axis=-1)
    result = {}
    for submodule, w, b in zip(["query", "key", "value"], split_w, split_b):
        # Squeeze the weights to remove the batch dimension.
        result[f"attn/{submodule}"] = {"w": w.squeeze(), "b": b.squeeze()}
    return result


def tf_to_haiku(tf_params: Dict[str, chex.ArrayNumpy]) -> dict:
    """Convert a pretrained model in Tensorflow to Haiku"""
    # Extract the embeddings and group the remaining parameters by module
    token_embeddings = None
    pos_embeddings = None
    grouped_params = defaultdict(dict)
    for name, value in tf_params.items():
        _, module = re.sub(r":[0-9]+$", "", name).split("/", 1)
        if module == "wte":
            token_embeddings = value
        elif module == "wpe":
            pos_embeddings = value
        else:
            module, remainder = module.split("/", 1)
            grouped_params[module][remainder] = value

    # Rename and reshape the TF params to match the Haiku architecture.
    converted_params = {
        "gpt2/positional": {
            "embeddings": pos_embeddings,
        },
        "gpt2/token": {"embeddings": token_embeddings},
    }
    for module in list(grouped_params.keys()):
        params = grouped_params[module]
        # Hidden layers
        if re.match(r"h[0-9]+", module):
            module_params = {
                "ln_1": {"scale": params["ln_1/g"], "offset": params["ln_1/b"]},
                "ln_2": {"scale": params["ln_2/g"], "offset": params["ln_2/b"]},
                "mlp/proj_in": {
                    "w": params["mlp/c_fc/w"].squeeze(),
                    "b": params["mlp/c_fc/b"],
                },
                "mlp/proj_out": {
                    "w": params["mlp/c_proj/w"].squeeze(),
                    "b": params["mlp/c_proj/b"],
                },
                "attn/linear": {
                    "w": params["attn/c_proj/w"].squeeze(),
                    "b": params["attn/c_proj/b"],
                },
            }
            module_params.update(
                convert_attention(params["attn/c_attn/w"], params["attn/c_attn/b"])
            )
            converted_params.update(
                {f"gpt2/transformer/{module}_{k}": v for k, v in module_params.items()}
            )

        # Final layer norm
        elif module == "ln_f":
            converted_params["gpt2/transformer/ln_f"] = {
                "scale": params["g"],
                "offset": params["b"],
            }
        else:
            raise ValueError(f"Unexpected module name: {module}")
        # Free up some RAM.
        del grouped_params[module]
        del params

    return hk.data_structures.to_immutable_dict(converted_params)

import functools
import logging
import os
import pathlib
import tarfile
import typing

import torch
import yaml
from transformers import GPT2Config
from utils.convert import cpu_map_location, gpu_map_location

# Initialize the logger for this module
LOGGER = logging.getLogger(__name__)

# Define a function to convert Nemo model configuration to GPT2Config
def nemo_to_gpt_config(nemo_model_config, vocab_size, eos_id, bos_id):
    # Define a dictionary to map Nemo configuration keys to GPT2Config keys
    convertion_dict = {
        "activation_function": "activation",
        "layer_norm_epsilon": "layernorm_epsilon",
        "n_embd": "hidden_size",
        "n_head": "num_attention_heads",
        "n_layer": "num_layers",
        "n_positions": "max_position_embeddings",
        "rotary_pct": "rotary_percentage",
        "bias": "bias",
        "intermediate_size": "ffn_hidden_size",
    }

    # Initialize an empty dictionary to store the GPT2Config arguments
    kwargs = {}

    # Iterate over the Nemo configuration keys and values
    for key, value in convertion_dict.items():
        # If the Nemo configuration key is present, add it to the GPT2Config arguments
        if value in nemo_model_config:
            kwargs[key] = nemo_model_config[value]

    # Add the vocab_size, eos_token_id, and bos_token_id arguments to the GPT2Config
    kwargs["vocab_size"] = vocab_size
    kwargs["eos_token_id"] = eos_id
    kwargs["bos_token_id"] = bos_id

    # Return a new GPT2Config instance with the specified arguments
    return GPT2Config(**kwargs)


# Define a function to add special tokens to a tokenizer
def add_special_tokens_to_tokenizer(tokenizer):

    # Define a dictionary to map special token names to their corresponding values
    special_tokens = {
        "cls_token": "<cls>",
        "sep_id": "<sep>",
        "mask_id": "<mask>",
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }

    # Iterate over the special tokens
    for token_name, token_value in special_tokens.items():
        # If the token is not already present in the tokenizer, add it
        if not hasattr(tokenizer, token_name):
            tokenizer.add_special_tokens({token_name: token_value})

        # If the token ID is not already present in the tokenizer, add it
        if not hasattr(tokenizer.tokenizer, f"{token_name}_id"):
            tokenizer.add_special_tokens({token_name: token_value})


# Define a function to unpack a Nemo checkpoint archive
def unpack_nemo_ckpt(nemo_archive_path: typing.Union[str, pathlib.Path],
                     out_dir_path: typing.Union[str, pathlib.Path]):
    # Convert the input paths to pathlib.Path objects
    nemo_archive_path = pathlib.Path(nemo_archive_path)
    out_dir_path = pathlib.Path(out_dir_path)

    # Check if the archive exists
    if not nemo_archive_path.exists():
        raise FileNotFoundError(f"{nemo_archive_path} does not exist")

    # Try to extract the archive using different modes
    for tar_mode in ["r:", "r:gz"]:
        try:
            # Open the archive in read mode
            with tarfile.open(nemo_archive_path, mode=tar_mode) as tar_file:

                # Define a function to check if a file path is within a directory
                def is_within_directory(directory, target):

                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)

                    prefix = os.path.commonprefix([abs_directory, abs_target])

                    return prefix == abs_directory

                # Define a function to get safe members from the archive
                def safe_members(tar_file):
                    members = []
                    for member in tar_file.getmembers():
                        member_path = os.path.join(out_dir_path, member.name)
                        if not is_within_directory(out_dir_path, member_path):
                            raise Exception(
                                "Attempted Path Traversal in Tar File")
                        members.append(member)
                    return members

                # Extract the archive to the output directory
                tar_file.extractall(out_dir_path,
                                    members=safe_members(tar_file),
                                    numeric_owner=False)

            # Return the output directory path
            return out_dir_path
        except tarfile.ReadError:
            pass

    # Raise an error if the archive could not be extracted
    raise RuntimeError(f"Could not unpack {nemo_archive_path}")


# Define a function to load a Nemo model checkpoint
def load_nemo_ckpt(nemo_ckpt_path: typing.Union[str, pathlib.Path],
                   device: torch.device):
    # Convert the input path to a pathlib.Path object
    nemo_ckpt_path = pathlib.Path(nemo_ckpt_path)

    # Check if the checkpoint exists
    if not nemo_ckpt_path.exists():
        raise FileNotFoundError(f"{nemo_ckpt_path} does not exist")

    # Load the Nemo model checkpoint
    nemo_model = torch.load(nemo_ckpt_path, map

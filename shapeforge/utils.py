import json
from pathlib import Path

import yaml


def load_yaml_config(file_path: str, show: bool = False) -> dict:
    """
    Load a YAML configuration file and optionally print its contents.

    Parameters
    ----------
    file_path : str
        The path to the YAML configuration file.
    show : bool, optional
        If True, print the loaded configuration. Default is False.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.
    """
    fp = Path(file_path).resolve()
    if not fp.exists():
        raise FileNotFoundError(
            f"Configuration file '{file_path}' does not exist."
        )
    if not fp.suffix == ".yaml":
        raise ValueError(
            f"Configuration file '{file_path}' must have a .yaml extension."
        )

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    if show:
        json_str = json.dumps(config, indent=4)
        print(f"Loaded configuration from {file_path}:\n{json_str}")

    return config


def _validate_dict(d: dict, keys: list, val_types: list = None) -> None:
    """
    Validate that a dictionary contains specific keys.

    Parameters
    ----------
    d : dict
        The dictionary to validate.
    keys : list
        The list of keys that must be present in the dictionary.
    val_types : list, optional
        A list of types corresponding to each key in `keys`. If provided,
        the function will also check that the values associated with each key
        are of the specified type.

    Raises
    ------
    ValueError
        If any of the specified keys are missing from the dictionary.
    """
    if not isinstance(d, dict):
        raise TypeError("Input must be a dictionary.")
    if not isinstance(keys, list):
        raise TypeError("Expected keys must be provided as a list.")

    missing_keys = [key for key in keys if key not in d]
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
    if val_types is None:
        val_types = [None] * len(keys)
    if len(keys) != len(val_types):
        raise ValueError("Length of keys and val_types must match.")
    for key, val_type in zip(keys, val_types):
        if val_type is None:
            continue
        if not isinstance(d[key], val_type):
            raise TypeError(
                f"Value for key '{key}' must be of type {val_type.__name__}, "
                f"but got {type(d[key]).__name__}."
            )

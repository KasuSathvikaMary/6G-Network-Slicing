import yaml
import os
import random
import numpy as np
import torch
import copy

def merge_configs(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        elif k == 'slices' and 'slices' in result:
            # Simple list merge: override slices by name
            base_slices = {s['name']: s for s in result['slices']}
            for override_slice in v:
                name = override_slice.get('name')
                if name in base_slices:
                    base_slices[name].update(override_slice)
                else:
                    base_slices[name] = override_slice
            result['slices'] = list(base_slices.values())
        else:
            result[k] = v
    return result

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    if 'base_config' in config:
        base_path = os.path.join(os.path.dirname(path), config['base_config'])
        base_config = load_config(base_path)
        config = merge_configs(base_config, config)
    return config

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_feature_names(n_slices: int, slice_names: list) -> list:
    features = []
    for f_name in ["demand", "latency", "violations", "alloc_prb", "alloc_cu", "priority", "tod"]:
        for s_name in slice_names:
            features.append(f"{f_name}_{s_name}")
    return features

import random
import torch
import os
import pickle
import json

import numpy as np
import pandas as pd

def set_seed(seed):
    """ Set seed for reproducible experiments."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_pkl(dir, name, dict):
	""" Save dict to pickle file. """
	os.makedirs(dir, exist_ok=True)
	filename = os.path.join(dir, name)
	with open(filename,'wb') as f:
	    pickle.dump(dict, f)

def load_pkl(dir, name):
	""" Load pickle file to dict. """
	filename = os.path.join(dir, name)
	with open(filename, 'rb') as f:
		file = pickle.load(f)
	return file

def save_json(dir, name, dict):
	""" Save dict to json file. """
	os.makedirs(dir, exist_ok=True)
	filename = os.path.join(dir, name)
	with open(filename, 'w') as json_file:
		json.dump(dict, json_file, indent=4)


def list_to_device(list, device):
    """ Put all items in a list to the device (cpu or gpu). """
    return [item.to(device) for item in list]


def input_to_device(input, device):
    """ Put input data to the device (cpu or gpu). """
    # Check if input is a dict, list, or tensor.
    if isinstance(input, dict):
        return {k: v.to(device) for k, v in input.items()}
    elif isinstance(input, list):
        return list_to_device(input, device)
    elif isinstance(input, torch.Tensor):
        return input.to(device)
    else:
        raise TypeError(f"Unsupported input type: {type(input)}")


def overlap_col_df(df_1, df_2, col):
    """ Returns overlap of rows in a specific column. """
    return np.intersect1d(np.unique(df_1[col].values), np.unique(df_2[col].values))

def _series_intersection(s1, s2):
    """ Return intersection of two sets. """
    return pd.Series(list(set(s1) & set(s2)))
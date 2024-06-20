import torch
import numpy as np
import random

def collate_fn_train(batch):
    """
    select a random caption from each image
    """
    images = [item['x'] for item in batch]
    # get a random caption from each image
    random_index = [np.random.randint(0, len(item['captions'])) for item in batch]
    captions = [item['captions'][random_index[i]]
                for i, item in enumerate(batch)]
    return torch.stack(images), captions

def collate_fn_val(batch):
    """
    select a random caption from each image
    """
    images = [item['x'] for item in batch]
    # get a random caption from each image
    captions = [ item['captions'] for item in batch]
    return torch.stack(images), captions

def enforce_determinism(seed:int):
    np.random.seed(seed)
    random.seed(0)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True)
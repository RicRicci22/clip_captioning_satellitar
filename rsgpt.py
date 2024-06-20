import os
import json
from typing import List, Dict

import torch
import torchvision.transforms as T
from PIL import Image


class RSGPT(torch.utils.data.Dataset):
    """ 
    TODO: Add description
    https://arxiv.org/pdf/2307.15266
    """

    def __init__(
        self,
        root: str = ".data/rsgpt_dataset",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        
        self.root = root
        self.transform = transform
        self.captions = self.load_captions(os.path.join(self.root, "captions.json"))
        self.image_root = "images"

    @staticmethod
    def load_captions(path: str) -> List[Dict]:
        print(path)
        with open(path) as f:
            captions = json.load(f)["annotations"]
        return captions

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Dict:
        captions = self.captions[idx]
        path = os.path.join(self.root, self.image_root, captions["filename"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        sentence = [captions["caption"]+"<|endoftext|>"]
        return dict(x=x, captions=sentence)

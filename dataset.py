
from torchrs.datasets import RSICD, UCMCaptions
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset
import json
import os
from typing import List, Dict
from PIL import Image

# datasets: https://github.com/201528014227051/RSICD_optimal?tab=readme-ov-file


class NWPUCaptions(Dataset):
    splits = ["train", "val", "test"]

    def __init__(self, root, split, transform=None):
        assert split in self.splits

        self.image_root = "images"
        self.root = root
        self.split = split
        self.transform = transform
        self.captions = self.load_captions(os.path.join(root, "dataset_nwpu.json"), split)

    @staticmethod
    def load_captions(path: str, split: str) -> List[Dict]:
        with open(path) as f:
            file = json.load(f)
            captions = []
            for category in file.keys():
                for sample in file[category]:
                    if sample["split"] == split:
                        sample["category"] = category
                        captions.append(sample)
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        sample = self.captions[idx]
        path = os.path.join(self.root, self.image_root, sample["category"], sample["filename"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        sentences = []
        for key in sample.keys():
            # if key starts with "raw" then it is a caption
            if key.startswith("raw"):      
                sentences.append(sample[key])
        return dict(x=x, captions=sentences)

class SidneyCaptions(Dataset):
    splits = ["train", "val", "test"]

    def __init__(self, root, split, transform=None):
        assert split in self.splits

        self.image_root = "images"
        self.root = root
        self.split = split
        self.transform = transform
        captions_file = os.path.join(root, 'filenames', "descriptions_SIDNEY.txt")
        split_file = os.path.join(root, 'filenames', f"filenames_{split}.txt")
        self.captions = self.load_captions(captions_file, split_file)

    @staticmethod
    def load_captions(captions_file: str, split_file: str) -> List[Dict]:
        captions = open(captions_file)
        split = open(split_file)
        samples = []
        split = split.read().splitlines()
        captions = captions.read().splitlines()
        captions = [caption.split() for caption in captions]
        captions = [ {'id': caption[0], 'caption': caption[1:]} for caption in captions ]

        for s in split:
            id = s.split('.')[0]
            c = []
            for caption in captions:
                if caption['id'] == id:
                   c.append(' '.join(caption['caption']).strip())
            samples.append({'filename': s, 'captions': c})
            
        
        return samples

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        sample = self.captions[idx]
        path = os.path.join(self.root, self.image_root, sample["filename"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        return dict(x=x, captions=sample["captions"])

def get_datasets(transform):

    # RSICD datasets
    rscid_dataset_train = RSICD(
        root="data/rsicd/",
        split="train", 
        transform=transform
    )
    rscid_dataset_val = RSICD(
        root="data/rsicd/",
        split="val", 
        transform=transform
    )

    # UCM Captions datasets
    ucm_dataset_train = UCMCaptions(
        root="data/ucm/",
        split="train", 
        transform=transform
    )
    ucm_dataset_val = UCMCaptions(
        root="data/ucm/",
        split="val", 
        transform=transform
    )

    # NWPUCaptions datasets
    nwpucaptions_dataset_train = NWPUCaptions(
        root="data/nwpu/",
        split="train", 
        transform=transform
    )
    nwpucaptions_dataset_val = NWPUCaptions(
        root="data/nwpu/",
        split="val", 
        transform=transform
    )

    sydney_dataset_train = SidneyCaptions(
        root="data/sidney/",
        split="train", 
        transform=transform
    )

    sydney_dataset_val = SidneyCaptions(
        root="data/sidney/",
        split="val", 
        transform=transform
    )


    dataset_train = ConcatDataset([rscid_dataset_train, ucm_dataset_train, nwpucaptions_dataset_train, sydney_dataset_train])
    dataset_val = ConcatDataset([rscid_dataset_val, ucm_dataset_val, nwpucaptions_dataset_val, sydney_dataset_val])
    return dataset_train, dataset_val
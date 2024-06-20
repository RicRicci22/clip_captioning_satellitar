# load model
import torch
from model import VisGPT, REMOTE_CLIP, CLIP, VGG
from dataset import NWPUCaptions, SidneyCaptions
from torch.utils.data import DataLoader
from torchrs.datasets import RSICD, UCMCaptions
from torchvision import transforms as T
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.meteor.meteor import Meteor

import numpy as np
from dataset import get_test_datasets, get_test_gpt_dataset

import nltk
import argparse
nltk.download('wordnet')

parser = argparse.ArgumentParser(description='Evaluation script for captioning model')
parser.add_argument('--encoder', type=str, default='REMOTE_CLIP', help='Encoder type')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device type')
parser.add_argument('--path', type=str, default='data/models/full.pth', help='Path to the model checkpoint')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')

args = parser.parse_args()

assert args.encoder in [REMOTE_CLIP, CLIP, VGG]
assert args.device in ['cuda', 'cpu', 'mps']

print(f'Using device: {args.device}')

net = VisGPT(device=args.device, encoder=args.encoder).to(args.device)
net.load_state_dict(torch.load(args.path, map_location=args.device))

def collate_fn(batch):
    """
    select a random caption from each image
    """
    images = [item['x'] for item in batch]
    # get a random caption from each image
    captions = [ item['captions'] for item in batch]
    return torch.stack(images), captions

test_datasets = get_test_gpt_dataset(net.encoder.preprocess)

bleu_scorer = Bleu(n=4)

cider_scorer = Cider()
rouge_scorer = Rouge()
spice_scorer = Spice()
meteor_scorer = Meteor()

def test(dataloader):
    net.eval()

    refs = {}
    res = {}

    count = 0
    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images = images.to(args.device)
            results = net.get_caption(images)
            print(results)
            for b in range(len(captions)):
                # Clean captions to remove anomalies
                if results[b].strip() != "":
                    refs[count + b] = captions[b]
                    res[count + b] = [results[b].strip()]

            count += len(captions)

        bleu_score = bleu_scorer.compute_score(refs, res, verbose=False)[0]
        rouge_score, _ = rouge_scorer.compute_score(refs, res)
        cider_score, _ = cider_scorer.compute_score(refs, res)
        spice_score, _ = spice_scorer.compute_score(refs, res)
        meteor_score, _ = meteor_scorer.compute_score(refs, res)

    return {
        'bleu1': bleu_score[0],
        'bleu2': bleu_score[1],
        'bleu3': bleu_score[2],
        'bleu4': bleu_score[3],
        'rouge': rouge_score,
        'cider': cider_score,
        'meteor': meteor_score,
        'spice': spice_score
    }

if 'sidney' in test_datasets.keys():
    sidneycaptions_dataset = test_datasets['sidney']
    sidneycaptions_dataloader = DataLoader(sidneycaptions_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    res = test(sidneycaptions_dataloader)

    print("-------------- SidneyCaptions results --------------")
    print(f'SidneyCaptions BLEU1: {res["bleu1"]}')
    print(f'SidneyCaptions BLEU2: {res["bleu2"]}')
    print(f'SidneyCaptions BLEU3: {res["bleu3"]}')
    print(f'SidneyCaptions BLEU4: {res["bleu4"]}')
    print(f'SidneyCaptions ROUGE: {res["rouge"]}')
    print(f'SidneyCaptions CIDER: {res["cider"]}')
    print(f'SidneyCaptions METEOR: {res["meteor"]}')
    print(f'SidneyCaptions SPICE: {res["spice"]}')
    print('\n\n')

if 'rsicd' in test_datasets.keys():
    rsicd_dataset = test_datasets['rsicd']
    rsicd_dataloader = DataLoader(rsicd_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    res = test(rsicd_dataloader)

    print("-------------- RSICD results --------------")
    print(f'RSICD BLEU1: {res["bleu1"]}')
    print(f'RSICD BLEU2: {res["bleu2"]}')
    print(f'RSICD BLEU3: {res["bleu3"]}')
    print(f'RSICD BLEU4: {res["bleu4"]}')
    print(f'RSICD ROUGE: {res["rouge"]}')
    print(f'RSICD CIDER: {res["cider"]}')
    print(f'RSICD METEOR: {res["meteor"]}')
    print(f'RSICD SPICE: {res["spice"]}')
    print('\n\n')

if 'ucm' in test_datasets.keys():
    ucm_dataset = test_datasets['ucm']
    ucm_dataloader = DataLoader(ucm_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    res = test(ucm_dataloader)

    print("-------------- UCM results --------------")
    print(f'UCM BLEU1: {res["bleu1"]}')
    print(f'UCM BLEU2: {res["bleu2"]}')
    print(f'UCM BLEU3: {res["bleu3"]}')
    print(f'UCM BLEU4: {res["bleu4"]}')
    print(f'UCM ROUGE: {res["rouge"]}')
    print(f'UCM CIDER: {res["cider"]}')
    print(f'UCM METEOR: {res["meteor"]}')
    print(f'UCM SPICE: {res["spice"]}')
    print('\n\n')

if 'nwpu' in test_datasets.keys():
    nwpucaptions_dataset = test_datasets['nwpu']
    nwpucaptions_dataloader = DataLoader(nwpucaptions_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    res = test(nwpucaptions_dataloader)

    print("-------------- NWPUCaptions results --------------")
    print(f'NWPUCaptions BLEU1: {res["bleu1"]}')
    print(f'NWPUCaptions BLEU2: {res["bleu2"]}')
    print(f'NWPUCaptions BLEU3: {res["bleu3"]}')
    print(f'NWPUCaptions BLEU4: {res["bleu4"]}')
    print(f'NWPUCaptions ROUGE: {res["rouge"]}')
    print(f'NWPUCaptions CIDER: {res["cider"]}')
    print(f'NWPUCaptions METEOR: {res["meteor"]}')
    print(f'NWPUCaptions SPICE: {res["spice"]}')
    print('\n\n')

if 'rsgpt' in test_datasets.keys():
    rsgptcaptions_dataset = test_datasets['rsgpt']
    rsgptcaptions_dataloader = DataLoader(rsgptcaptions_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    res = test(rsgptcaptions_dataloader)

    print("-------------- RSGPTCaptions results --------------")
    print(f'RSGPTCaptions BLEU1: {res["bleu1"]}')
    print(f'RSGPTCaptions BLEU2: {res["bleu2"]}')
    print(f'RSGPTCaptions BLEU3: {res["bleu3"]}')
    print(f'RSGPTCaptions BLEU4: {res["bleu4"]}')
    print(f'RSGPTCaptions ROUGE: {res["rouge"]}')
    print(f'RSGPTCaptions CIDER: {res["cider"]}')
    print(f'RSGPTCaptions METEOR: {res["meteor"]}')
    print(f'RSGPTCaptions SPICE: {res["spice"]}')
    print('\n\n')

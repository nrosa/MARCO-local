# from efficientnet_pytorch import EfficientNet
import torchvision
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import argparse
import misc
import data
import tqdm 
import os
import torch.nn.functional as F

import timm

import modified_resnet

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.distributions import Categorical

import json

import torchvision




def main(args):
    # Load the data
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.precrop_img_size),
        torchvision.transforms.CenterCrop(args.img_size),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    dataset = torchvision.datasets.ImageFolder(
        args.data_loc,
        transform=test_transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )
    dl_iter = iter(dataloader)

    ckpt_path = os.path.join(
        args.results_dir,
        args.slurmid,
        'marco-2',
        'last.pth.tar'
    )

    model = timm.models.create_model(
        args.model,
        num_classes=4,
        checkpoint_path=ckpt_path
    )

    model.cuda()

    # ckpt = torch.load(os.path.join(
    #     '/scratch1/ros282/results/marco_retrain',
    #     args.model,
    #     'best-weights.pth' if args.best else 'checkpoint.pth'
    # ))
    # model.load_state_dict(ckpt['model_state_dict'])

    entropy_lst = []
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataloader))):
            imgs, _ = next(dl_iter)
            

            preds = F.softmax(model(imgs.cuda()), dim=-1)

            entropy = Categorical(probs = preds).entropy()
            entropy_lst.append(entropy)
            
    entropy_lst = torch.cat(entropy_lst)
    entropy_lst = [x.item() for x in entropy_lst]
    file_paths = [x[0] for x in dataset.samples]


     
    with open('entropy.json', 'w') as fp:
        json.dump(
            list(zip(entropy_lst,file_paths)),
            fp,
            indent=4
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DDP usage example')

    # Flags
    parser.add_argument('--convert', action='store_true')
    parser.add_argument('--best', action='store_true')

    # Ints
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--precrop_img_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)

    # Strings
    parser.add_argument('--slurmid', type=str, default='')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--data_loc', type=str, default=r'/scratch1/ros282/')
    parser.add_argument('--results_dir', type=str, default=r'/scratch1/ros282/results/marco_retrain')

    args = parser.parse_args()

    main(args)


    

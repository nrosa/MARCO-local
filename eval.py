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

import modified_resnet

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler




def print_conf_matrix(conf_matrix):
    print_matrix = copy.deepcopy(conf_matrix)
    print_matrix[0] = ['Pred Clear'] + print_matrix[0]
    print_matrix[1] = ['Pred Crystal'] + print_matrix[1]
    print_matrix[2] = ['Pred Other'] + print_matrix[2]
    print_matrix[3] = ['Pred Precipitate'] + print_matrix[3]
    print_matrix = [[
        ' ',
        'True Clear',
        'True Crystal',
        'True Other',
        'True Precipitate'
    ]] + print_matrix
    s = [[str(e) for e in row] for row in print_matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def main(args):
    # Load the data
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.precrop_img_size),
        torchvision.transforms.CenterCrop(args.img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    marco_test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(args.marco_data_loc, 'test'),
        transform=test_transforms,
    )
    marco_test_dataloader = torch.utils.data.DataLoader(
        marco_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    c3_test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(args.c3_data_loc, 'vis'),
        transform=test_transforms,
    )
    c3_test_dataloader = torch.utils.data.DataLoader(
        c3_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    class Convertor(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module = misc.EMA(model, decay=0.99)

        def forward(self, x):
            return self.module(x)

    # #model = misc.EMA(torchvision.models.inception_v3(num_classes=4), decay=0.9999)
    # model = Convertor()
    # model.cuda()

    if args.arch =='resnet50':
        model = torchvision.models.resnet50(num_classes=4)
    elif args.arch == 'modresnet50':
        model = modified_resnet.resnet(num_classes=4)
    elif args.arch == 'densenet201':
        model = torchvision.models.densenet201(num_classes=4)
    elif args.arch == 'densenet161':
        model = torchvision.models.densenet161(num_classes=4)
    elif args.arch == 'wide_resnet50': 
        model = torchvision.models.wide_resnet50_2(num_classes=4)
    elif args.arch == 'resnet152':    
        model = torchvision.models.resnet152(num_classes=4)
    else:
        model = torchvision.models.resnet50(num_classes=4)

    if args.convert:
        model = Convertor(model)
    model.cuda()

    ckpt = torch.load(os.path.join(
        '/scratch1/ros282/results/marco_retrain',
        args.model,
        'best-weights.pth'
    ))
    model.load_state_dict(ckpt['model_state_dict'])

    
    model.eval()
    with torch.no_grad():
        eval_losses = 0
        eval_acc = 0
        conf_matrix = [[0 for _ in range(4)] for _ in range(4)]
        for data_batch in tqdm.tqdm(c3_test_dataloader):
            imgs = data_batch[0].cuda()
            target = data_batch[1].cuda()
            preds = model(imgs)
            loss = torch.nn.functional.cross_entropy(preds, target, reduction='sum')
            preds = torch.argmax(preds, dim=1)
            acc = (preds == target).float().sum()

            for j in range(preds.shape[0]):
                conf_matrix[preds[j]][target[j]] += 1

            eval_losses += loss.item()
            eval_acc += acc.item()

        c3_eval_loss = eval_losses/len(c3_test_dataset)
        c3_eval_acc = eval_acc/len(c3_test_dataset)
        print(f'C3 Evalutation: Loss {round(c3_eval_loss,4)}, Acc {round(c3_eval_acc,4)}')
        print_conf_matrix(conf_matrix)


        eval_losses = 0
        eval_acc = 0
        conf_matrix = [[0 for _ in range(4)] for _ in range(4)]
        for data_batch in tqdm.tqdm(marco_test_dataloader):
            imgs = data_batch[0].cuda()
            target = data_batch[1].cuda()
            preds = model(imgs)
            loss = torch.nn.functional.cross_entropy(preds, target, reduction='sum')
            preds = torch.argmax(preds, dim=1)
            acc = (preds == target).float().sum()

            for j in range(preds.shape[0]):
                conf_matrix[preds[j]][target[j]] += 1

            eval_losses += loss.item()
            eval_acc += acc.item()

        marco_eval_loss = eval_losses/len(marco_test_dataset)
        marco_eval_acc = eval_acc/len(marco_test_dataset)
        print(f'Marco Evalutation: Loss {round(marco_eval_loss,4)}, Acc {round(marco_eval_acc,4)}')
        print_conf_matrix(conf_matrix)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DDP usage example')

    # Flags
    parser.add_argument('--convert', action='store_true')

    # Ints
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--precrop_img_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)

    # Strings
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--marco_data_loc', type=str, default=r'/scratch1/ros282/marco_full')
    parser.add_argument('--c3_data_loc', type=str, default=r'/scratch1/ros282/c3_testset')

    args = parser.parse_args()

    main(args)


    

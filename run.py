# from efficientnet_pytorch import EfficientNet
import torchvision
import torchvision.transforms as T
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import argparse
import misc
import data
import os
import datetime
import sys

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import modified_resnet


OUT_FORMAT = '[{:3d}/{:3d}][{:4d}/{:4d}]  Loss: {:.4f} | Acc: {:.4f} | ({})'


class Normalize_Tensor(torch.nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.view(x.shape[-3], -1).mean(dim=-1)
        std = x.view(x.shape[-3], -1).std(dim=-1).clamp(min=self.eps)
        return T.functional.normalize(x, mean, std)


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
    args.is_master = args.world_rank == 0
    args.device = torch.cuda.device(args.local_rank)

    args.out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.is_master:
        print(f'Out directory is :{args.out_dir}')
        print(sys.argv[1:])

    # initialize PyTorch distributed using environment variables
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=args.world_rank)
    torch.cuda.set_device(args.local_rank)

    batch_size = args.batch_size // torch.distributed.get_world_size()

    # Load the data
    train_transforms = T.Compose([
        T.RandomChoice([
            T.RandomResizedCrop(args.img_size, interpolation=0),
            T.RandomResizedCrop(args.img_size, interpolation=2),
            T.RandomResizedCrop(args.img_size, interpolation=3),
        ]),
        T.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.2),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transforms = T.Compose([
        T.Resize(args.precrop_img_size),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # train_data_set =  data.CombinedImageFolderDataset(
    #     '/storage/nrosa/datasets/marco_256/train',
    #     '/storage/nrosa/datasets/c3_testset_256/vis',
    #     100000,
    #     transform=train_transforms,
    # )
    train_dataset = torchvision.datasets.ImageFolder(
        '/scratch1/ros282/marco_full/train',
        # '/storage/nrosa/datasets/marco_256/train',
        transform=train_transforms,
    )
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=train_sampler
    )
    marco_test_dataset = torchvision.datasets.ImageFolder(
        '/scratch1/ros282/marco_full/test',
        # '/storage/nrosa/datasets/marco_256/test',
        transform=test_transforms,
    )
    marco_test_sampler = DistributedSampler(marco_test_dataset)
    marco_test_dataloader = torch.utils.data.DataLoader(
        marco_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=marco_test_sampler
    )

    c3_test_dataset = torchvision.datasets.ImageFolder(
        '/scratch1/ros282/c3_testset/vis',
        transform=test_transforms,
    )
    c3_test_sampler = DistributedSampler(c3_test_dataset)
    c3_test_dataloader = torch.utils.data.DataLoader(
        c3_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=c3_test_sampler
    )

    # Create the model
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

    model = misc.EMA(model, decay = args.ema_decay)
    model.cuda()

    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        #find_unused_parameters=True,
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    opt_steps = len(train_dataloader) * args.epochs
    if args.is_master:
        print(f'Will run for {opt_steps} iterations ({args.epochs} epochs)')
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.94)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    warmup = misc.Warmup(optimizer)

    marco_eval_losses = []
    marco_eval_accs = []
    marco_train_losses = []
    marco_train_accs = []
    c3_eval_losses = []
    c3_eval_accs = []

    best_val_loss = np.inf

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        marco_test_sampler.set_epoch(epoch)
        # c3_test_sampler.set_epoch(epoch)

        # Start the timer
        start = time.time()

        epoch_losses = []
        epoch_acc = []

        epoch_len = len(train_dataloader)
        report_freq = epoch_len // 4 + 1

        for i, data_batch in enumerate(train_dataloader):
            imgs = data_batch[0].cuda()
            target = data_batch[1].cuda()
            model.zero_grad()
            preds = model(imgs)
            loss = torch.nn.functional.cross_entropy(preds, target)
            acc = (torch.argmax(preds, dim=1) == target).float().mean()
            loss.backward()

            optimizer.step()
            warmup.step()
            model.module.update()

            torch.distributed.all_reduce(loss)
            loss /= torch.distributed.get_world_size()
            torch.distributed.all_reduce(acc)
            acc /= torch.distributed.get_world_size()
            epoch_losses.append(loss.item())
            epoch_acc.append(acc.item())

            # Output training stats
            if i == 0 or i + 1 == epoch_len or (i + 1) % report_freq == 0:
                out_str = OUT_FORMAT.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    epoch_len,
                    np.mean(epoch_losses),
                    np.mean(epoch_acc),
                    time.strftime('%H:%M:%S')
                )
                if args.is_master:
                    print(out_str, flush=True)

        marco_train_losses.append(np.mean(epoch_losses))
        marco_train_accs.append(np.mean(epoch_acc))

        lr_sched.step()

        # Eval
        model.eval()
        with torch.no_grad():
            eval_loss = torch.tensor(0.).cuda()
            eval_acc = torch.tensor(0.).cuda()
            conf_matrix = [[0 for _ in range(4)] for _ in range(4)]
            for i, data_batch in enumerate(marco_test_dataloader):
                imgs = data_batch[0].cuda()
                target = data_batch[1].cuda()
                preds = model(imgs)
                loss = torch.nn.functional.cross_entropy(preds, target, reduction='sum')
                preds = torch.argmax(preds, dim=1)
                acc = (preds == target).float().sum()

                for j in range(preds.shape[0]):
                    conf_matrix[preds[j]][target[j]] += 1

                eval_loss += loss               
                eval_acc += acc

            torch.distributed.all_reduce(eval_loss)
            torch.distributed.all_reduce(eval_acc)
            marco_eval_loss = eval_loss.item() / len(marco_test_dataset)
            marco_eval_acc = eval_acc.item() / len(marco_test_dataset)
            
            if args.is_master:
                print(f'Marco Evalutation: Loss {round(marco_eval_loss, 4)}, Acc {round(marco_eval_acc, 4)}')
                

            best = False
            if marco_eval_loss < best_val_loss:
                best_val_loss = marco_eval_loss
                best = True
                if args.is_master:
                    print("Best validation loss")

            # print_conf_matrix(conf_matrix)

            eval_loss = torch.tensor(0.).cuda()
            eval_acc = torch.tensor(0.).cuda()
            conf_matrix = [[0 for _ in range(4)] for _ in range(4)]
            for i, data_batch in enumerate(c3_test_dataloader):
                imgs = data_batch[0].cuda()
                target = data_batch[1].cuda()
                preds = model(imgs)
                loss = torch.nn.functional.cross_entropy(preds, target, reduction='sum')
                preds = torch.argmax(preds, dim=1)
                acc = (preds == target).float().sum()

                for j in range(preds.shape[0]):
                    conf_matrix[preds[j]][target[j]] += 1

                eval_loss += loss               
                eval_acc += acc

            torch.distributed.all_reduce(eval_loss)
            torch.distributed.all_reduce(eval_acc)
            c3_eval_loss = eval_loss.item() / len(c3_test_dataset)
            c3_eval_acc = eval_acc.item() / len(c3_test_dataset)
            
            if args.is_master:
                print(f'C3 Evalutation: Loss {round(c3_eval_loss, 4)}, Acc {round(c3_eval_acc, 4)}')

        marco_eval_losses.append(marco_eval_loss)
        marco_eval_accs.append(marco_eval_acc)
        c3_eval_losses.append(c3_eval_loss)
        c3_eval_accs.append(c3_eval_acc)
 

        model.train()

        if args.is_master:
            torch.save(
                {
                    'model_state_dict': model.module.shadow.state_dict(),
                },
                os.path.join(args.out_dir, 'weights.pth')
            )
            if best:
                torch.save(
                    {
                        'model_state_dict': model.module.shadow.state_dict(),
                    },
                    os.path.join(args.out_dir, 'best-weights.pth')
                )
            # '/storage/nrosa/results/marco_retrain/weights.pth'
            fig, axes = plt.subplots()
            # axes.plot(marco_train_losses)
            # axes.plot(marco_train_accs)
            axes.plot(marco_eval_losses)
            axes.plot(c3_eval_losses)

            axes.plot(marco_eval_accs)
            axes.plot(c3_eval_accs)
            axes.legend(['Marco Test Loss', 'Marco Test Acc', 'C3 Test Loss', 'C3 Test Acc'])
            fig.savefig(os.path.join(args.out_dir, 'eval.png'))
            plt.close(fig)
            print(
                f'Time for epoch {epoch + 1} is {time.time()-start} sec',
                flush=True
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DDP usage example')

    # Ints
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--precrop_img_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=16)
    # FLoats
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--name', type=str, default='default')
    # Strings
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--data_dir',type=str, default='')
    parser.add_argument('--arch', type=str, default='resnet50')
    args = parser.parse_args()

    args.local_rank = int(os.environ['SLURM_PROCID']) % int(os.environ['SLURM_NTASKS_PER_NODE'])
    args.world_rank = int(os.environ['SLURM_PROCID'])

    import socket

    print(f'--local{args.local_rank}__world{args.world_rank}__{socket.gethostname()}--')


    

    main(args)

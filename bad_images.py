#from efficientnet_pytorch import EfficientNet
import torchvision, torch
import numpy as np
import time
import matplotlib.pyplot as plt
import copy 

import misc, data

OUT_FORMAT = '[{:3d}/{:3d}][{:4d}/{:4d}]  Loss: {:.4f} | Acc: {:.4f} | ({})'

BATCH_SIZE = 64
IMG_SIZE = 224
PRE_CROP_SIZE = 256
WORKERS = 16

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(PRE_CROP_SIZE),
    torchvision.transforms.CenterCrop(IMG_SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


class ImageFolderName(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = torchvision.datasets.folder.default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

c3_test_data_set =  ImageFolderName(
    '/storage/nrosa/datasets/c3_testset_256/vis',
    transform=test_transforms,
)
c3_test_data_loader = torch.utils.data.DataLoader(
    c3_test_data_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS
)

model = misc.EMA(torchvision.models.resnet18(num_classes=4), decay=0.95)
model.cuda()

checkpoint = torch.load('/storage/nrosa/results/marco_retrain/weights.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
hard_precips = []
with torch.no_grad():
    for data_batch in c3_test_data_loader:
        imgs = data_batch[0].cuda()
        target = data_batch[1].cuda()
        preds = torch.argmax(model(imgs), dim=1)

        for i in range(preds.shape[0]):
            if preds[i] == 1 and target[i] ==3:
                hard_precips.append(data_batch[2][i])

print(hard_precips)

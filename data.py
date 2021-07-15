import torch, torchvision

import torchvision.datasets.vision
import torchvision.datasets.folder

import os
import numpy as np



class CombinedImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
            self,  
            base_dataset_folder: str,
            extra_dataset_folder: str,
            extras: int,
            transform=None,
        ):

        self.base_dataset_folder = base_dataset_folder
        self.extra_dataset_folder = extra_dataset_folder

        self.classes, self.class_to_idx = self._find_classes(self.base_dataset_folder)

        self.samples = self._find_samples(self.base_dataset_folder)
        extra_samples = self._find_samples(self.extra_dataset_folder)

        # Shuffle the extra samples
        rng = np.random.default_rng()
        rng.shuffle(extra_samples)
        if extras > len(extra_samples):
            extras = len(extra_samples)

        self.samples += extra_samples[:extras]

        self.transform = transform


    def _find_classes(self, dir: str):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _find_samples(self, dir):
        samples = []
        for klass in self.classes:
            for file in os.listdir(os.path.join(dir, klass)):
                if 'jpg' in file:
                    samples.append((
                        os.path.join(dir, klass, file),
                        self.class_to_idx[klass]
                    ))

        return samples

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = torchvision.datasets.folder.default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

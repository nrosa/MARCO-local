import os
import shutil

import data
import tqdm

reduc = 0.7
cache_dir = '/scratch1/ros282/cache'
dataset_loc = '/scratch1/ros282'
full_dataset = 'marco_full'

dataset = data.SemanticReducedDataset(
    os.path.join(dataset_loc, full_dataset, 'train'),
    reduc,
    model=None,
    transform=None,
    rep_transform=None,
    repr_cache=os.path.join(cache_dir, 'marco_train_repr.dat'),
    repr_load_cache=True,
    reduc_cache=os.path.join(cache_dir, f'marco_train_{reduc}.dat'),
    reduc_load_cache=True,
)

new_dir = os.path.join(dataset_loc, f'marco_{reduc}','train')
os.makedirs(os.path.join(new_dir, 'Clear'))
os.makedirs(os.path.join(new_dir, 'Crystals'))
os.makedirs(os.path.join(new_dir, 'Precipitate'))
os.makedirs(os.path.join(new_dir, 'Other'))

# Copy the training data
for sample in tqdm.tqdm(dataset.samples):
    sample_new_loc = os.path.join(
        new_dir,
        *sample[0].split('/')[-2:]
    )

    if not os.path.isfile(sample_new_loc):
        shutil.copy(sample[0], sample_new_loc)

# Copy the val data
shutil.copytree(
    os.path.join(dataset_loc, full_dataset, 'val'),
    os.path.join(dataset_loc, f'marco_{reduc}', 'val')
)

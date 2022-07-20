import os
import shutil
import random

# Split the dataset into two parts

# new_dir1 = '/tmp/ros282.6230231/c3_testset_vis_p1'
# new_dir2 = '/tmp/ros282.6230231/c3_testset_vis_p2'

# source_dir = os.path.join('/tmp/ros282.6230231','c3_testset', 'vis')

# for label in ['crystal', 'clear', 'other', 'precipitate']:
#     os.makedirs(os.path.join(new_dir1, label), exist_ok=True)
#     os.makedirs(os.path.join(new_dir2, label), exist_ok=True)

#     files = [x for x in os.listdir(os.path.join(source_dir, label)) if os.path.isfile(os.path.join(source_dir, label, x))]
#     random.shuffle(files)

#     split = len(files) // 2

#     p1_files = files[:split]
#     p2_files = files[split:]

#     for f in p1_files:
#         shutil.copy(
#             os.path.join(source_dir, label, f),
#             os.path.join(new_dir1, label, f)
#         )

#     for f in p2_files:
#         shutil.copy(
#             os.path.join(source_dir, label, f),
#             os.path.join(new_dir2, label, f)
#         )

# Add images to marco dataset

source_dir = '/tmp/ros282.6230231/c3_testset_vis_p1'
marco_dir = '/tmp/ros282.6230231/marco_full_wc3p1/train'

labels = [
    ('crystal','Crystals'),
    ('clear', 'Clear'),
    ('other', 'Other'),
    ('precipitate','Precipitate')
]

for label, label2 in labels:
    files = [x for x in os.listdir(os.path.join(source_dir, label)) if os.path.isfile(os.path.join(source_dir, label, x))]

    

    print(
        label,
        len(os.listdir(os.path.join(marco_dir, label2)))
        )

    for f in files:
        shutil.copy(
            os.path.join(source_dir, label, f),
            os.path.join(marco_dir, label2, f)
        )

    print(
        label,
        len(os.listdir(os.path.join(marco_dir, label2)))
    )
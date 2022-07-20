# Local MARCO 

A project to train a crystal outcome classification model using the MARCO dataset supplemented with local laboratory images.

## Datasets
Datasets are provided as a directory. The MARCO dataset should contain both train and test folders.

```
.
└── marco-dataset/
    ├── train/
    │   ├── clear/
    │   │   ├── clear-train-img1.jpg
    │   │   ├── clear-train-img2.jpg
    │   │   └── ...
    │   ├── crystal/
    │   │   ├── crystal-train-img1.jpg
    │   │   ├── crystal-train-img2.jpg
    │   │   └── ...
    │   ├── precipitate/
    │   │   ├── precipitate-train-img1.jpg
    │   │   ├── precipitate-train-img2.jpg
    │   │   └── ...
    │   └── other/
    │       ├── other-train-img1.jpg
    │       ├── other-train-img2.jpg
    │       └── ...
    └── test/
        ├── clear/
        │   ├── clear-test-img1.jpg
        │   ├── clear-test-img2.jpg
        │   └── ...
        ├── crystal/
        │   ├── crystal-test-img1.jpg
        │   ├── crystal-test-img2.jpg
        │   └── ...
        ├── precipitate/
        │   ├── precipitate-test-img1.jpg
        │   ├── precipitate-test-img2.jpg
        │   └── ...
        └── other/
            ├── other-test-img1.jpg
            └── other-test-img2.jpg
```

The local dataset should only contain the four label folders.

```
.
└── local-dataset/
    ├── clear/
    │   ├── clear-img1.jpg
    │   ├── clear-img2.jpg
    │   └── ...
    ├── crystal/
    │   ├── crystal-img1.jpg
    │   ├── crystal-img2.jpg
    │   └── ...
    ├── precipitate/
    │   ├── precipitate-img1.jpg
    │   ├── precipitate-img2.jpg
    │   └── ...
    └── other/
        ├── other-img1.jpg
        ├── other-img2.jpg
        └── ...
```


## Training Models

### From MARCO data only
The following command can be used to train a model from the MARCO data only.
```
python run.py $path_to_MARCO_dataset
```

Different models and hyper-parameters can be adjusted with the arguements. The full set of arguments can be found by looking in `run.py`.

### With MARCO data and local data
THe following command can be used to train a model with the MARCO data and local data

```
python run.py $path_to_MARCO_dataset --add_local_images --local_data_dir $path_to_local_dataset
```

## Semantic Reduction


### Reduced sets from the paper

The reduced datasets that were used to generate results in the paper can be found in `\sem_reduc\reduced_datasets`. They are python pickle files that contain a list of the samples used for each set.
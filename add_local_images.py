import argparse, shutil, os
import numpy as np

LABEL_DIST = {
    'clear' : {
        'uniform' : 0.25,
        'marco' : 0.336
    },
    'crystals' : {
        'uniform' : 0.25,
        'marco' : 0.128
    },
    'precipitate' : {
        'uniform' : 0.25,
        'marco' : 0.48
    },
    'other' : {
        'uniform' : 0.25,
        'marco' : 0.056
    }
}

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class Image(object):
    def __init__(self, filepath, label):
        self.filepath = filepath
        self.label = label

    def __str__(self):
        return f'filepath: {self.filepath}, label: {self.label}'


def has_file_allowed_extension(filename: str, extensions) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

class ImageFactory(object):
    def __init__(self, dataset_path):
        self.labels, self.class_to_idx = self.find_classes(dataset_path)
        samples = self.make_dataset(dataset_path, self.class_to_idx, IMG_EXTENSIONS)

        self.idx_to_class = dict()
        for key in self.class_to_idx:
            self.idx_to_class[self.class_to_idx[key]] = key

        self.images = sorted([Image(x[0], self.idx_to_class[x[1]]) for x in samples],key=lambda x: x.filepath)

    def find_classes(self, directory: str):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def make_dataset(
        self,
        directory,
        class_to_idx = None,
        extensions = None,
    ):
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances


    def get_image_lst(self, label=None):
        if label is None:
            return self.images
        else:
            return [x for x in self.images if x.label==label]


def _copy_images(imgs, amount, destination, seed=0):
    assert len(imgs) >= amount
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(imgs)

    for img in imgs[:amount]:
        shutil.copy(
            img.filepath,
            os.path.join(destination, img.label)
        )

    print(f"Copied {amount} images...  ")


def add_images(image_cnt, label_dist, input_dir, output_dir, seed=0):
    img_fact = ImageFactory(input_dir)

    if label_dist is None:
        images = img_fact.get_image_lst()

        _copy_images(images, image_cnt, output_dir, seed=seed)

    else:
        for label in LABEL_DIST.keys():
            images = img_fact.get_image_lst(label=label)
            img_cnt = int(round(image_cnt * LABEL_DIST[label][label_dist]))
            _copy_images(images, img_cnt, output_dir)

def main(args):
    add_images(args.image_cnt, args.label_dist, args.input_dir, args.output_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser('')


    # Strings
    parser.add_argument('--label_dist', type=str, choices=['none', 'uniform', 'marco'])
    parser.add_argument('--image_cnt', type=int, default=1200)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    

    args = parser.parse_args()

    main(args)
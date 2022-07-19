import json, argparse, shutil, os
import numpy as np

LABEL_DIST = {
    'Clear' : {
        'uniform' : 0.25,
        'marco' : 0.336
    },
    'Crystals' : {
        'uniform' : 0.25,
        'marco' : 0.128
    },
    'Precipitate' : {
        'uniform' : 0.25,
        'marco' : 0.48
    },
    'Other' : {
        'uniform' : 0.25,
        'marco' : 0.056
    }
}

class Image(object):
    def __init__(self, entropy, cross_entropy, filepath):
        self.entropy = entropy
        self.cross_entropy = cross_entropy
        self.filepath = filepath
        self.label = filepath.split('/')[4].title()

    def __str__(self):
        return f'filepath: {self.filepath}, label: {self.label}, entropy: {self.entropy}, cross_entropy: {self.cross_entropy}'



class ImageFactory(object):
    def __init__(self, entropy_json_path, cross_entropy_json_path):
        with open(entropy_json_path) as fp:
            entropy_data = json.load(fp)
        with open(cross_entropy_json_path) as fp:
            cross_entropy_data = json.load(fp)   

        self.images = [Image(x[0], y[0], x[1]) for x, y in zip(entropy_data, cross_entropy_data)]

    def get_image_lst(self, label=None):
        if label is None:
            return self.images
        else:
            return [x for x in self.images if x.label==label]


def _copy_images(imgs, method, amount, destination):
    assert len(imgs) >= amount
    if method == 'random':
        rng = np.random.default_rng()
        rng.shuffle(imgs)
    elif method == 'entropy':
        imgs.sort(key=lambda x: x.entropy, reverse=True)
    elif method == 'cross_entropy':
        imgs.sort(key=lambda x: x.cross_entropy, reverse=True)

    for img in imgs[:amount]:
        shutil.copy(
            img.filepath,
            os.path.join(destination, img.label)
        )

    print(f"Copied {amount} images...  ")


def add_images(method, image_cnt, label_dist, data_dir):
    img_fact = ImageFactory('entropy.json', 'cross_entropy.json')

    if label_dist == 'none':
        images = img_fact.get_image_lst()

        _copy_images(images, method, image_cnt, data_dir)

    else:
        for label in ['Clear', 'Crystals', 'Precipitate', 'Other']:
            images = img_fact.get_image_lst(label=label)
            img_cnt = int(round(image_cnt * LABEL_DIST[label][label_dist]))
            _copy_images(images, method, img_cnt, data_dir)

def main(args):
    add_images(args.method, args.image_cnt, args.label_dist, args.output_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser('')


    # Strings
    parser.add_argument('--label_dist', type=str, choices=['none', 'uniform', 'marco'])
    parser.add_argument('--method', type=str, choices=['random', 'entropy', 'cross_entropy'])
    parser.add_argument('--output_dir', type=str)

    

    args = parser.parse_args()
    print(args)

    main(args)
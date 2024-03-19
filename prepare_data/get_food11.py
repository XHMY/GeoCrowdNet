import os
from glob import glob
import numpy as np
from os.path import join
import cv2
from tqdm import tqdm


# kaggle datasets download -d trolukovich/food11-image-dataset
# unzip food11-image-dataset.zip -d food11

# transform the Image Folder classification Dataset to numpy array
def read_to_ndarray(images_dir, labels_map):
    images = []
    labels = []
    for image in tqdm(glob(join(images_dir, "*", "*.jpg"))):
        img = cv2.imread(image)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        images.append(img)
        labels.append(labels_map[image.split(os.sep)[-2]])

    return np.array(images), np.array(labels).astype(np.uint64)


def main():
    labels_map = {name: i for i, name in enumerate(sorted(os.listdir(join("data/food11", "training"))))}
    print(labels_map)
    train_images, train_labels = read_to_ndarray(join("data/food11", "training"), labels_map)
    test_images, test_labels = read_to_ndarray(join("data/food11", "validation"), labels_map)

    np.save("data/food11/train_images.npy", train_images)
    np.save("data/food11/train_labels.npy", train_labels)
    np.save("data/food11/test_images.npy", test_images)
    np.save("data/food11/test_labels.npy", test_labels)


if __name__ == '__main__':
    main()

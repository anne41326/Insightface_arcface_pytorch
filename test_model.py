"""Caculate accuracy and threshold of model"""
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import os
import os.path as osp
from pathlib import Path

from structure import FaceMobileNet

THIS_FILE_PATH = Path(__file__)
PAIRS_TXT_PATH = THIS_FILE_PATH.parent / "pairs.txt"
CHECKPOINS_PATH = THIS_FILE_PATH.parent / "checkpoints/23.pth"
test_root = THIS_FILE_PATH.parent / "data/val"

embedding_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_shape = [1, 128, 128]
test_transform = T.Compose([
    T.Grayscale(),
    T.Resize(input_shape[1:]),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])
test_batch_size = 64


def unique_image(pair_list=PAIRS_TXT_PATH) -> set:
    unique = set()
    """Return unique image path in pair_list.txt"""
    with open(PAIRS_TXT_PATH, 'r', encoding='UTF-8') as fd:
        pairs = fd.read().split("\n")

        #print(pairs)
    for i, pair in enumerate(pairs):
        print(i, pair.split())
        id1, id2, _ = pair.split()
        unique.add(id1)
        unique.add(id2)
    return unique


def group_image(images: set, batch) -> list:
    """Group image paths by batch size"""
    images = list(images)
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch + i, size)
        res.append(images[i:end])
    return res


def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)
    return data


def featurize(images: list, transform, net, device) -> dict:
    """featurize each image and save into a dictionary
    Args:
        images: image paths
        transform: test transform
        net: pretrained model
        device: cpu or cuda
    Returns:
        Dict (key: imagePath, value: feature)
    """
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data)
    res = {img: feature for (img, feature) in zip(images, features)}
    return res


def cosin_metric(x1, x2):
    # for more:https://blog.csdn.net/weixin_43977640/article/details/115579153
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


def compute_accuracy(feature_dict, test_root):
    """Use featurized dicts to compute similarity of photos in val folder"""
    pair_list = PAIRS_TXT_PATH
    with open(pair_list, 'r', encoding='UTF-8') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()

        img1 = osp.join(test_root, img1)

        img2 = osp.join(test_root, img2)

        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold


def model_accuracy(embedding_size, device, test_root, test_batch_size,
                   test_transform):
    """Compute the accuracy and threshold of checkpoints 
      Args: 
        device: cuda or cpu
        test_root: the path of val photo folder
        embedding_size: default, 512
        test_batch_size: default,64
        test_transform: default
      Output:
        None
    """

    model = FaceMobileNet(embedding_size)
    model = nn.DataParallel(model)
    test_model = CHECKPOINS_PATH
    model.load_state_dict(torch.load(test_model, map_location=device))
    model.eval()

    images = unique_image(pair_list=PAIRS_TXT_PATH)
    images = [osp.join(test_root, img) for img in images]
    groups = group_image(images, test_batch_size)

    feature_dict = dict()
    for group in groups:
        d = featurize(group, test_transform, model, device)
        feature_dict.update(d)
    accuracy, threshold = compute_accuracy(feature_dict, test_root)

    print(f"Test Model: {test_model}\n"
          f"Accuracy: {accuracy:.3f}\n"
          f"Threshold: {threshold:.3f}\n")


if __name__ == '__main__':
    model_accuracy(embedding_size, device, test_root, test_batch_size,
                   test_transform)

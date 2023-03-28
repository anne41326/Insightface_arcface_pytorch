import os
import os.path as osp
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from structure import FaceMobileNet, ResIRSE
from structure import ArcFace, CosFace
from structure import FocalLoss
from load_data import load_data

input_shape = [1, 128, 128]
train_transform = T.Compose([
    T.Grayscale(),
    T.RandomHorizontalFlip(),
    T.Resize((144, 144)),
    T.RandomCrop(input_shape[1:]),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])
train_batch_size = 64
pin_memory = True  # if memory is large, set it True to speed up a bit
num_workers = 4  # dataloader
checkpoints = "checkpoints"
restore = False
restore_model = ""
epoch = 24
drop_ratio = 0.5
lr = 1e-1
lr_step = 10
lr_decay = 0.95
weight_decay = 5e-4
embedding_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

THIS_FILE_PATH = Path(__file__)
dataroot = THIS_FILE_PATH.parent / "data/train"

dataloader, class_num = load_data(dataroot,
                                  train_transform,
                                  train_batch_size,
                                  pin_memory,
                                  num_workers,
                                  training=True)


# Network Setup
def start_train(backbone, metric, loss, optimizer, checkpoints, restore,
                epoch):
    if backbone == 'resnet':
        net = ResIRSE(embedding_size, drop_ratio).to(device)
    else:
        net = FaceMobileNet(embedding_size).to(device)

    if metric == 'arcface':
        metric = ArcFace(embedding_size, class_num).to(device)
    else:
        metric = CosFace(embedding_size, class_num).to(device)

    net = nn.DataParallel(net)
    metric = nn.DataParallel(metric)

    if loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()

    if optimizer == 'sgd':
        optimizer = optim.SGD([{
            'params': net.parameters()
        }, {
            'params': metric.parameters()
        }],
                              lr=lr,
                              weight_decay=weight_decay)
    else:
        optimizer = optim.Adam([{
            'params': net.parameters()
        }, {
            'params': metric.parameters()
        }],
                               lr=lr,
                               weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=lr_step,
                                          gamma=0.1)

    # Checkpoints Setup
    checkpoints = checkpoints
    os.makedirs(checkpoints, exist_ok=True)

    if restore:
        weights_path = osp.join(checkpoints, restore_model)
        net.load_state_dict(torch.load(weights_path, map_location=device))

    # Start training
    net.train()

    for e in range(epoch):
        for data, labels in tqdm(dataloader,
                                 desc=f"Epoch {e}/{epoch}",
                                 ascii=True,
                                 total=len(dataloader)):
            if not len(data) > 1:
                continue
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = net(data)
            thetas = metric(embeddings, labels)
            loss = criterion(thetas, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {e}/{epoch}, Loss: {loss}")

        backbone_path = osp.join(checkpoints, f"{e}.pth")
        print(backbone_path)

        torch.save(net.state_dict(), backbone_path)
        scheduler.step()


if __name__ == '__main__':
    start_train('fmobile', 'arcface', 'focal_loss', 'sgd', "checkpoints",
                False, 24)

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

#from config import config as conf
input_shape = [1, 128, 128]
train_transform = T.Compose([
            T.Grayscale(),
            T.RandomHorizontalFlip(),
            T.Resize((144, 144)),
            T.RandomCrop(input_shape[1:]),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
test_transform = T.Compose([
            T.Grayscale(),
            T.Resize(input_shape[1:]),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

train_batch_size = 64
test_batch_size = 60
pin_memory = True  # if memory is large, set it True to speed up a bit
num_workers = 4  # dataloader

def load_data(dataroot,transform,bactch_size,pin_memory,num_workers,training=True):
    if training:
        
        transform = train_transform
        batch_size = train_batch_size
    else:
        
        transform = test_transform
        batch_size = test_batch_size

    data = ImageFolder(dataroot, transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, 
        pin_memory=pin_memory, num_workers=num_workers)
    return loader, class_num



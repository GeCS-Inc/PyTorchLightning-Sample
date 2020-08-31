import os
import torch
import torchvision


def get_dataset(env, transform):

    assert env.dataset in ["custom", "imagenet"]

    if env.dataset == "custom":
        dataset = torchvision.datasets.ImageFolder(
            env.dataset_root, transform=transform)
        N = len(dataset)
        num_train = int(N * 0.8)
        num_val = N - num_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val])

    return train_dataset, val_dataset

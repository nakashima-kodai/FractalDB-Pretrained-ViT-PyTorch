import os

import hydra
from torchvision.datasets.folder import ImageFolder, default_loader
from hydra.utils import instantiate


def create_dataloader(dtcfg):
    transform = instantiate(dtcfg.transform)
    print(f'Data augmentation for training is as follows \n{transform}\n')

    dataset = instantiate(dtcfg.set, transform=transform)
    num_classes = len(dataset.classes)
    print(f'{len(dataset)} images and {num_classes} classes was found in {dtcfg.set.root}\n')

    sampler = None
    if dtcfg.sampler._target_ is not None:
        sampler = instantiate(dtcfg.sampler, dataset=dataset)
    dataloader = instantiate(dtcfg.loader, dataset=dataset, sampler=sampler)
    return dataloader, num_classes

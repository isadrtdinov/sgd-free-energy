import numpy as np
import torch
import torchvision
from torchvision import transforms
import os
from torch.utils.data import RandomSampler, TensorDataset

__all__ = ['loaders', 'get_loaders', 'get_datasets']
c10_classes = np.array([[0, 1, 2, 8, 9], [3, 4, 5, 6, 7]], dtype=np.int32)

transform_f = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

def transform_dataset(dataset, device):
    images = torch.tensor(dataset.data, dtype=torch.float32) / 255.0
    images = images.permute(0, 3, 1, 2)
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1) 
    
    normalized_images = (images - mean) / std
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    tensor_dataset = TensorDataset(normalized_images.to(device), labels.to(device))
    return tensor_dataset


def get_datasets(
    dataset,
    path,
    device,
    samples_per_label = None
):
    path = os.path.join(path, dataset.lower())
    ds = getattr(torchvision.datasets, dataset)

    raw_train = ds(root=path, train=True, download=True)
    raw_test = ds(root=path, train=False, download=True)
    num_classes = max(raw_train.targets) + 1

    if samples_per_label is not None:
        targets_np = np.array(raw_train.targets)
        chosen_idxs = []

        for cls in range(num_classes):
            idxs = np.where(targets_np == cls)[0]
            if len(idxs) < samples_per_label:
                raise ValueError(
                    f"Class {cls} only has {len(idxs)} samples, "
                    f"but you requested {samples_per_label}"
                )
            # randomly pick without replacement
            pick = np.random.choice(idxs, samples_per_label, replace=False)
            chosen_idxs.extend(pick.tolist())

        # re‑index the raw data and targets
        np.random.shuffle(chosen_idxs)
        raw_train.data    = raw_train.data[chosen_idxs]
        raw_train.targets = targets_np[chosen_idxs].tolist()

    train_set = transform_dataset(raw_train, device)
    test_set = transform_dataset(raw_test, device)
    return {"train": train_set, "test": test_set}, num_classes


def get_loaders(
        dataset,
        path,
        batch_size,
        num_iters,
        num_workers,
        shuffle_train=True
):
    path = os.path.join(path, dataset.lower())
    dl_dict = dict()

    ds = getattr(torchvision.datasets, dataset)

    train_set = ds(root=path, train=True, download=True, transform=transform_f)
    num_classes = max(train_set.targets) + 1

    sampler = RandomSampler(train_set, replacement=True, num_samples=batch_size * num_iters)

    test_set = ds(
        root=path, train=False, download=True, transform=transform_f
    )
    dl_dict.update({
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            # shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler
        ),
        "test": torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    })
    return dl_dict, num_classes

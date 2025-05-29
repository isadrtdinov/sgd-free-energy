import os
import json
from pathlib import Path


def check_si_name(n, model_name):
    if model_name == 'resnet':
        return 'linear' not in n
    elif 'convnet' in model_name:
        return 'conv_layers' in n
    return False


def fix_si_pnorm(model, si_pnorm_0, model_name):
    """
    Fix SI-pnorm to si_pnorm_0 value
    """
    si_pnorm = np.sqrt(sum((p ** 2).sum().item() for n, p in model.named_parameters() if check_si_name(n, model_name)))
    p_coef = si_pnorm_0 / si_pnorm
    for n, p in model.named_parameters():
        if check_si_name(n, model_name):
            p.data *= p_coef
            
def download_folder(folder_name):
    triples = []
    it = 0
    data = []
    for p in Path(folder_name).iterdir():
        name = p.name
        if name.startswith("lr"):
            lr = float(name[2:])
            if not os.path.exists(f'{folder_name}/{name}/trace.pt'):
                continue
            f = torch.load(f'{folder_name}/{name}/trace.pt', map_location=torch.device('cpu'))
            tr = f['trace']
            with open(f'{folder_name}/{name}/config.json', 'r') as f:
                cfg = (json.load(f))
            triples.append((lr, tr, cfg))
    triples.sort(key=lambda x: x[0])
    lrs = [x[0] for x in triples]
    trace = [x[1] for x in triples]
    config = [x[2] for x in triples]
    num_params = len(trace[0]['weight'][0])
    return lrs, trace, config, num_params

def get_train_loss(model, criterion, train_dataset, test_dataset, eval_batch_size=1024):
    model.train()
    model.zero_grad()
    train_loss = 0.0
    train_acc = 0.0
    i = 0
    while i < len(train_dataset):
        j = min(i + eval_batch_size, len(train_dataset))
        x, y = train_dataset[i:j]
        loss, output = criterion(model, x, y, "sum")
        train_loss += loss.item()
        i += eval_batch_size
    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)
    return train_loss

from data import get_datasets
from utility import *
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
from torch.nn.utils import vector_to_parameters
from entropy_logger import EntropyLogger
from models.convnet import ConvNetDepth as conv_net
from models.resnet import make_resnet18k

model_name='convnet'
dataset='cifar100'
num_classes = 100

for num_channels in [8]:

    batch_size = 128
    if model_name == "convnet":
        model = conv_net(
            num_classes=num_classes,
            init_channels=num_channels,
        )
    else:
        model = make_resnet18k(k=num_channels, num_classes=num_classes)
    lrs, trace, config, npp = download_folder(f'exp-{model_name}{num_channels}-seed1-{dataset}-50000obj-ce')

    set_seed(config[0]['data_seed'])
    data, _ = get_datasets(dataset.upper(), config[0]['data_path'], 'cuda', samples_per_label=500)

    criterion = cross_entropy
    num_params = sum([p.numel() for p in model.parameters()])
    si_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    lnn_params = num_params - si_params

    matrix_logger = EntropyLogger('cuda', 1.0, si_params)
    results_entropy=[]
    for _ in tqdm(range(10)):
        weights = torch.randn((1000, si_params))
        weights /= weights.norm(dim=1, keepdim=True)
        _, ent = matrix_logger.get_metrics(weights)
        results_entropy.append(ent)
    
    results = []
    for _ in tqdm(range(100)):
        weights = torch.randn(si_params)
        weights /= weights.norm()
        weights = torch.concat([weights, trace[0]['weight'][0][-lnn_params:]])
        vector_to_parameters(weights, model.parameters())
        model.to('cuda')
        tl = get_train_loss(model, criterion, data['train'], data['test'])
        results.append(tl)
    torch.save({'loss': results, 'entropy': results_entropy}, f'metrics/loss_{model_name}{num_channels}-{dataset}.pt')
import os
import json
import torch
from train import train_model, check_si_name, fix_si_pnorm
import torch.nn as nn
from data import get_datasets
from utility import set_seed, cross_entropy, mse
from torch.nn.utils import vector_to_parameters
from models.convnet import ConvNetDepth as conv_net
from models.resnet import make_resnet18k

def pretrain(config):
    os.makedirs(config.savedir, exist_ok=True)
    with open(f'{config.savedir}/config.json', 'w') as f:
        json.dump(config.__dict__, f)

    if config.data_seed is not None:
        set_seed(config.data_seed)
    datasets, num_classes = get_datasets(config.dataset.upper(), config.data_path, config.device, config.samples_per_label)
    print(f"Train dataset size: {len(datasets['train'])}")

    # initialize model
    if config.init_point_seed is not None:  # set the same initialization point for all LRs
        set_seed(config.init_point_seed)
    if config.model_name == 'convnet':
        model = conv_net(
            num_classes=num_classes,
            init_channels=config.num_channels,
            init_scale=config.last_layer_norm
        )
    elif config.model_name == 'resnet':
        model = make_resnet18k(k=config.num_channels, num_classes=num_classes)

    # initialize last layer
    if config.last_layer_seed is not None:  # set the same last layer for all LRs
        set_seed(config.last_layer_seed)
    
        if config.model_name == 'convnet':
            fin = nn.Linear(model.linear_layers[-1].in_features, model.linear_layers[-1].out_features)
            alpha = config.last_layer_norm
            W = fin.weight.data
            model.linear_layers[-1].weight.data = alpha * W / W.norm()
            model.linear_layers[-1].bias.data = fin.bias.data
        elif config.model_name == 'resnet':
            fin = nn.Linear(model.linear.in_features,model.linear.out_features,bias=False)
            alpha = config.last_layer_norm
            W = fin.weight.data
            model.linear.weight.data = alpha * W / W.norm()

    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'SI parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    model.to(config.device)

    set_seed(config.pt_seed)

    param_groups = [
        {'params': [p for n, p in model.named_parameters() if check_si_name(n, config.model_name)]},  # SI params are convolutions
        {'params': [p for n, p in model.named_parameters() if not check_si_name(n, config.model_name)], 'lr': 0.0},  # other params
    ]

    fix_si_pnorm(model, config.si_pnorm_0, config.model_name)
    with torch.no_grad():
        lr = config.elr * config.si_pnorm_0 ** 2

    criterion = {'ce': cross_entropy, 'mse': mse}[config.loss_fn]
    optimizer = torch.optim.SGD(param_groups, lr=lr)

    logfile = f'{config.savedir}/log.log'
    ckpt_path = f'{config.savedir}/trace.pt'
    trace = train_model(
        model, config.model_name, criterion, optimizer, logfile, datasets['train'], datasets['test'],
        config.batch_size, config.num_iters, config.device, lr, config.si_pnorm_0,
        config.ckpt_iters, config.log_iters, config.queue_size, ckpt_path, config.iid_batches
    )

    torch.save({
        'model': model.state_dict(),
        'trace': trace
    }, ckpt_path)

    
    
def evaluate_on_checkpoints(config):
    os.makedirs(config.savedir, exist_ok=True)
    with open(f'{config.savedir}/config_eval.json', 'w') as f:
        json.dump(config.__dict__, f)

    if config.data_seed is not None:
        set_seed(config.data_seed)
    datasets, num_classes = get_datasets(config.dataset.upper(), config.data_path, config.device, config.samples_per_label)
    print(f"Train dataset size: {len(datasets['train'])}")

    logfile = f'{config.savedir}/log_eval.log'
    ckpt_path = f'{config.savedir}/trace_eval.pt'
    weights = torch.load(f'{config.savedir}/trace.pt', map_location=torch.device(config.device))['trace']['weight']
    trace = {}
    for ckpt in weights:
        if config.model_name == 'convnet':
            model = conv_net(
                num_classes=num_classes,
                init_channels=config.num_channels,
                init_scale=config.last_layer_norm
            )
        elif config.model_name == 'resnet':
            model = make_resnet18k(k=config.num_channels, num_classes=num_classes) 
        vector_to_parameters(ckpt, model.parameters())
        model.to(config.device)

        param_groups = [
            {'params': [p for n, p in model.named_parameters() if check_si_name(n, config.model_name)]},  # SI params are convolutions
            {'params': [p for n, p in model.named_parameters() if not check_si_name(n, config.model_name)], 'lr': 0.0},  # other params
        ]

        with torch.no_grad():
            lr = config.elr * config.si_pnorm_0 ** 2

        criterion = {'ce': cross_entropy, 'mse': mse}[config.loss_fn]
        optimizer = torch.optim.SGD(param_groups, lr=lr)

        trace_ckpt = train_model(
            model, config.model_name, criterion, optimizer, logfile, datasets['train'], datasets['test'],
            config.batch_size, config.num_iters, config.device, lr, config.si_pnorm_0,
            config.ckpt_iters, config.log_iters, config.queue_size, ckpt_path, config.iid_batches
        )
        for k,v in trace_ckpt.items():
            if k not in trace.keys():
                trace[k] = [v]
            else:
                trace[k].append(v)

    torch.save(trace, ckpt_path)

    
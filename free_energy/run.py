import argparse
import numpy as np
from types import SimpleNamespace
from scripts import pretrain, evaluate_on_checkpoints


def get_config(seed, device, num_iters, lr, num_channels, model_name, dataset='cifar10', samples_per_label=5000, loss_fn='ce', regime="pretrain"):
    if regime == "pretrain":
        if num_iters is None:
            num_iters = 1_600_000
        log_iters = np.unique(np.array(
            np.arange(10).tolist() + np.logspace(1, np.log10(num_iters), 193).tolist()
        ).astype(int))
        ckpt_iters = np.array([0] + np.logspace(4, np.log10(num_iters), 9).tolist()).astype(int)
    elif regime == "ckpt":
        if num_iters is None:
            num_iters = 1000
        log_iters = np.array([num_iters])
        ckpt_iters = np.array([])
    dataset_size = samples_per_label * (10 if dataset == 'cifar10' else 100)
    return SimpleNamespace(
        dataset=dataset, # 'cifar10 or cifar100
        model_name=model_name, # convnet or resnet
        samples_per_label=samples_per_label,  # size of the dataset / 10 for CIFAR-10; default is 5000
        batch_size=128,  # training batch size
        iid_batches=True,  # sample batches independently or train by epochs
        si_pnorm_0=1.0,  # parameter norm
        num_channels=num_channels,  # number of channels in ConvNet
        last_layer_norm=10.0 if loss_fn == 'ce' else 1.5,  # last layer norm in MLP (last layer is fixed and not trained)
        dtype='float32',  # training data type
        device=device,  # computational device
        loss_fn=loss_fn,  # loss function used for training (ce or mse)
        num_iters=num_iters,  # pre-training iterations
        ckpt_iters=ckpt_iters.tolist(),  # how often to checkpoint model
        log_iters=log_iters.tolist(),  # how often to calculate metrics
        pt_seed=seed,  # training seed
        data_seed=1,  # dataset generation seed
        init_point_seed=1,  # random init seed
        last_layer_seed=4,  # classifier head initialization seed
        savedir=f'exp-{model_name}{num_channels}-seed{seed}-{dataset}-{dataset_size}obj-{loss_fn}/lr{lr:2e}',  # pre-training path
        elr=lr,  # learning rate range to use
        queue_size=1000,  # queue size for entropy calculation and snr estimation
        data_path='./datasets'
    )


parser = argparse.ArgumentParser()
parser.add_argument('--seeds', nargs="+", type=int)
parser.add_argument('--device', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--samples_per_label', type=int, default=5000)
parser.add_argument('--loss_fn', type=str, default="ce")
parser.add_argument('--regime', type=str, default="pretrain") # pretrain or ckpt
parser.add_argument('--model', type=str, default='convnet')
parser.add_argument('--num_iters', type=int, default=None)
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

for seed in args.seeds:
    config = get_config(seed=seed, device=args.device, num_iters=args.num_iters, lr=args.lr, 
                        num_channels=args.channels, model_name=args.model, dataset=args.dataset,
                        samples_per_label=args.samples_per_label, loss_fn=args.loss_fn, regime=args.regime)
    if args.regime == "pretrain":
        pretrain(config)
    else:
        evaluate_on_checkpoints(config)

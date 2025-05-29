import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model_parameters(model, requires_grad=False):
    if not requires_grad:
        return parameters_to_vector(model.parameters())
    else:
        return parameters_to_vector([param for param in model.parameters() if param.requires_grad])


def get_model_gradients(model, requires_grad=False):
    flattened_gradients = []
    for param in model.parameters():
        if param.grad is not None and param.requires_grad:
            flattened_gradients.append(param.grad.view(-1))
        elif not requires_grad:
            flattened_gradients.append(torch.zeros_like(param).view(-1))
    return torch.cat(flattened_gradients)


def cross_entropy(model, x, target, reduction="mean"):
    """standard cross-entropy loss function"""
    if model is not None:
        output = model(x)
    else:
        output = x

    loss = F.cross_entropy(output, target, reduction=reduction)

    if model is not None:
        return loss, output

    return loss


def mse(model, x, target, reduction="mean", num_classes=10):
    if model is not None:
        output = model(x)
    else:
        output = x

    loss = F.mse_loss(
        output,
        F.one_hot(target, num_classes=num_classes).to(dtype=output.dtype),
        reduction='none'
    ).sum(dim=-1)

    if reduction is None or reduction == "none":
        loss = loss
    if reduction == 'mean':
        loss = torch.mean(loss)
    if reduction == 'sum':
        loss = torch.sum(loss)
        
    if model is not None:
        return loss, output

    return loss

def eval_model(model, criterion, train_dataset, test_dataset, eval_batch_size=1024):
    model.train()
    model.zero_grad()
    train_loss = 0.0
    train_acc = 0.0
    i = 0
    while i < len(train_dataset):
        j = min(i + eval_batch_size, len(train_dataset))
        x, y = train_dataset[i:j]
        loss, output = criterion(model, x, y, "sum")
        loss.backward()
        train_loss += loss.item()
        train_acc += (torch.argmax(output, dim=1) == y).sum().item()
        i += eval_batch_size
    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)
    full_grad = get_model_gradients(model, requires_grad=True).cpu() / len(train_dataset)

    model.eval()
    model.zero_grad()
    test_loss = 0.0
    test_acc = 0.0
    i = 0
    while i < len(test_dataset):
        j = min(i + eval_batch_size, len(test_dataset))
        x, y = test_dataset[i:j]
        loss, output = criterion(model, x, y, "sum")
        test_loss += loss.item()
        test_acc += (torch.argmax(output, dim=1) == y).sum().item()
        i += eval_batch_size
    test_loss /= len(test_dataset)
    test_acc /= len(test_dataset)

    return {
        'train_loss': train_loss, 'train_acc': train_acc,
        'test_loss': test_loss, 'test_acc': test_acc
    }, full_grad


def get_stoch_grads(model, criterion, dataset, number_of_stoch_grads, batch_size, requires_grad=False):
    model.train()
    stoch_grads = []
    for _ in range(number_of_stoch_grads):
        iid = torch.randint(0, len(dataset), (batch_size,))
        x, y = dataset[iid]
        model.zero_grad()
        loss, output = criterion(model, x, y, "mean")
        loss.backward()
        stoch_grad = get_model_gradients(model, requires_grad=requires_grad).cpu()
        stoch_grads.append(stoch_grad)
    return torch.stack(stoch_grads, dim=0)


@torch.no_grad()
def get_snr(full_grad, stoch_grads):
    full_grad_norm = full_grad.norm().item()
    centered_stoch_grads = stoch_grads - full_grad.unsqueeze(0)
    stoch_grad_rms = (centered_stoch_grads ** 2).sum(axis=1).mean(axis=0).sqrt().item()
    stoch_grad_mean_norm = stoch_grads.norm(dim=1).mean().item()
    stoch_grad_mean_norm_squared = (stoch_grads ** 2).sum(dim=1).mean().item()
    noise_mean_norm = centered_stoch_grads.norm(dim=1).mean().item()
    snr_old = full_grad_norm / noise_mean_norm
    snr_new = full_grad_norm / stoch_grad_rms

    return {'snr_new': snr_new, 'snr_old': snr_old,
            'full_grad_norm': full_grad_norm, 'noise_mean_norm': noise_mean_norm,
            'stoch_grad_rms': stoch_grad_rms, 'stoch_grad_mean_norm': stoch_grad_mean_norm,
             'stoch_grad_mean_norm_squared': stoch_grad_mean_norm_squared}

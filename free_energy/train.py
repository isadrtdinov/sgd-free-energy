import time
from tqdm import tqdm
from entropy_logger import EntropyLogger
from utility import *


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


def train_model(model, model_name, criterion, optimizer, logfile, train_dataset, test_dataset,
                batch_size, num_iters, device, lr, si_pnorm_0,
                ckpt_iters, log_iters, queue_size, ckpt_path, iid_batches):
    trace_keys = [
        'weight', 'stoch_grad',
        'train_loss', 'train_acc', 'test_loss', 'test_acc', 'stoch_grad_norm',
        'snr_new', 'snr_old', 'full_grad_norm', 'noise_mean_norm', 'stoch_grad_rms', 'stoch_grad_mean_norm',
        'stoch_grad_mean_norm_squared',
        'traj_ents', 'stoch_ents'
    ]
    trace = {key: [] for key in trace_keys}
    matrix_logger = EntropyLogger(
        device, lr, sum(p.numel() for p in model.parameters() if p.requires_grad), queue_size
    )

    if logfile:
        with open(logfile, "a") as file:
            print("Start training", file=file)
            start_time = time.time()

    def log(stoch_grad_norm):
        metrics, full_grad = eval_model(model, criterion, train_dataset, test_dataset)
        for key, value in metrics.items():
            trace[key].append(value)
        trace['stoch_grad_norm'].append(stoch_grad_norm)

        stoch_grads = get_stoch_grads(model, criterion, train_dataset,
                                      queue_size, batch_size, requires_grad=True)
        snr_dict = get_snr(full_grad, stoch_grads)
        for key, value in snr_dict.items():
            trace[key].append(value)

        traj_ents, stoch_ents = matrix_logger.get_metrics(stoch_grads)
        trace['traj_ents'].append(traj_ents)
        trace['stoch_ents'].append(stoch_ents)

    if 0 in log_iters:
        log(stoch_grad_norm=torch.nan)

    if 0 in ckpt_iters:
        weights = parameters_to_vector(model.parameters())
        trace['weight'].append(weights.cpu())
        trace['stoch_grad'].append(torch.full_like(weights, fill_value=torch.nan))

    if not iid_batches:
        cur_i = 0
        iid = torch.randperm(len(train_dataset))
    for it in tqdm(range(1, num_iters + 1)):
        optimizer.zero_grad()
        if iid_batches:
            iid = torch.randint(0, len(train_dataset), (batch_size,))
            input, target = train_dataset[iid]
        else:
            j = min(cur_i + batch_size, len(train_dataset))
            input, target = train_dataset[iid[cur_i:j]]
            if cur_i + batch_size >= len(train_dataset):
                cur_i = 0
                iid = torch.randperm(len(train_dataset))
            else:
                cur_i += batch_size
        model.train()
        loss, output = criterion(model, input, target, "mean")
        loss.backward()
        optimizer.step()
        fix_si_pnorm(model, si_pnorm_0, model_name)

        stoch_grad = get_model_gradients(model)
        matrix_logger.add_weights(get_model_parameters(model, requires_grad=True).detach())

        if it in log_iters:
            log(stoch_grad_norm=stoch_grad.norm().item())

            if logfile:
                with open(logfile, "a") as file:
                    cur_time = time.time()
                    print(
                        f'Elapsed: {cur_time - start_time:.2f}s. '
                        f'Logged metrics at iteration {it}', file=file
                    )

        if it in ckpt_iters:
            weights = parameters_to_vector(model.parameters())
            trace['weight'].append(weights.cpu())
            trace['stoch_grad'].append(stoch_grad.cpu())

            torch.save({
                'model': model.state_dict(),
                'trace': trace
            }, ckpt_path)

            if logfile:
                with open(logfile, "a") as file:
                    cur_time = time.time()
                    print(
                        f'Elapsed: {cur_time - start_time:.2f}s. '
                        f'Checkpointed weights and grads at iteration {it}', file=file
                    )

    return trace

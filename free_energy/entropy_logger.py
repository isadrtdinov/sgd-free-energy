import torch
import numpy as np
from collections import deque
from scipy.sparse.csgraph import minimum_spanning_tree
import time


def parallel_transition(A, B, AB, theta_AB, v):
    v_par = (v * AB).sum(dim=-1, keepdim=True) * AB
    v_ort = v - v_par

    v_par_rot1 = theta_AB.cos() * v_par - theta_AB.sin() * v_par.norm(dim=-1, keepdim=True) * A
    v_par_rot2 = theta_AB.cos() * v_par + theta_AB.sin() * v_par.norm(dim=-1, keepdim=True) * A

    abs_prod1 = (v_par_rot1 * B).sum(dim=-1, keepdim=True).abs()
    abs_prod2 = (v_par_rot2 * B).sum(dim=-1, keepdim=True).abs()

    mask = (abs_prod1 < abs_prod2).tile(1, v.shape[-1])
    v_par_rot = torch.where(mask, v_par_rot1, v_par_rot2)
    v_B = v_ort + v_par_rot
    return v_B


class EntropyLogger(object):
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]  # number of neighbors to estimate entropy
    gamma = 1.0  # power for entropy calculation
    max_queue_memory = 20 * 10 ** 9  # 20 Gb - maximal size for processing distances on GPU
    # alpha = 0.9

    def __init__(self, device, lr, num_params, queue_size=1000):
        self.device = device
        self.lr = lr
        self.num_params = num_params
        self.queue_size = queue_size
        self.weight_queue = deque(maxlen=queue_size)
        # self.grad_queue = deque(maxlen=graph_queue_size)

    def add_weights(self, weights):
        self.weight_queue.append(weights.cpu())

    '''
    def get_residuals(self):
        r = torch.zeros(self.num_params, dtype=torch.float32, device=self.device)
        rs = [r.clone()]
        for t in range(self.graph_queue_size - 1):
            x = self.weight_queue[t]

            g = -self.grad_queue[t]
            g = g - (g * x).sum(dim=-1, keepdim=True) * x
            g_norm = g.norm(dim=-1, keepdim=True).clip(min=1e-10)
            normed_g = g / g_norm

            angles = g_norm * self.lr
            x = self.weight_queue[t + 1]
            r = self.alpha * r + self.lr * g
            rs.append(r.clone())

            out = parallel_transition(self.grad_queue[t], self.grad_queue[t + 1], normed_g, angles, r)
            if out.isnan().any():
                break

            r = out
            r = r - (r * x).sum(dim=-1, keepdim=True) * x
            r = r.squeeze()
        return torch.stack(rs, dim=0)
    '''

    def calculate_entropy(self, dist_matrix, dim):
        # calculate kNN graphs
        dist_matrix, _ = torch.sort(dist_matrix ** self.gamma, dim=-1)
        knn_ents = torch.tensor([
            torch.clip(dist_matrix[..., :k + 1].sum(), min=1e-8).item() for k in self.ks
        ])  # (len(ks), )
        knn_ents = dim / self.gamma * (knn_ents.log() - (dim - self.gamma) / dim * np.log(self.queue_size))

        return knn_ents

    def get_metrics(self, stoch_grads):
        if len(self.weight_queue) == self.queue_size:
            w = torch.stack(list(self.weight_queue), dim=0)
            # if w.element_size() * w.nelement() < self.max_queue_memory:
            #     w = w.to(self.device)
            w = w.to(dtype=torch.float64)
            # cosine distance deprecated
            # weight_angles = torch.clip(w @ w.T, min=-1, max=1).arccos().cpu()
            weight_distances = torch.cdist(w, w).cpu()
            del w
            traj_ents = self.calculate_entropy(weight_distances, dim=self.num_params - 1)
        else:
            traj_ents = torch.full((len(self.ks), ), torch.nan)

        # if stoch_grads.element_size() * stoch_grads.nelement() < self.max_queue_memory:
        #     stoch_grads = stoch_grads.to(self.device)
        stoch_grad_dists = self.lr * torch.cdist(stoch_grads, stoch_grads)
        del stoch_grads
        stoch_ents = self.calculate_entropy(stoch_grad_dists, dim=self.num_params - 1)

        return traj_ents, stoch_ents

# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.autograd as autograd

def RBF(X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

    # Apply the median heuristic (PyTorch does not give true median)
    np_dnorm2 = dnorm2.detach().cpu().numpy()
    h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
    sigma = np.sqrt(h).item()

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()

    return K_XY

def phi(score_func, K, X):
    X.requires_grad_(True)
    K_XX = K(X, X.detach())
    grad_K = -autograd.grad(K_XX.sum(), X)[0]
    phi = (-K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
    X.requires_grad_(False)
    return -phi
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as normal
import torch.distributions.uniform as uniform
import math

def heuristic_kernel_width(x_samples, x_basis):
    n_samples = x_samples.size()[-2]
    n_basis = x_basis.size()[-2]
    x_samples_expand = x_samples.unsqueeze(-2)
    x_basis_expand = x_basis.unsqueeze(-3)
    pairwise_dist = (x_samples_expand - x_basis_expand).pow(2).sum(-1).pow(0.5)
    if n_samples * n_basis >= 2:
        k = n_basis // 2
        top_k_values, top_k_indices = torch.topk(pairwise_dist.view(-1, n_samples * n_basis), k, dim=1)  # top_k_vaues shape (1, k)
        kernel_width = top_k_values[:, -1].view(-1, 1)
    else:
        kernel_width = pairwise_dist[0, 0]
    return kernel_width.detach()

def rbf_kernel(z_1, z_2):
    z_2_ = z_2.unsqueeze(1)
    pdist_square = (z_1 - z_2_)**2
    bandwidth = heuristic_kernel_width(z_1, z_2)
    kzz = (pdist_square.sum(dim=-1)/(-2.*bandwidth**2)).exp()
    return kzz, bandwidth

class KernelNet(nn.Module):  # input noises, output unknown distributions same dimension as x
    def __init__(self, input_dim, output_dim):
        super(KernelNet, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x_1 = F.relu(self.layer_1(x))
        x_2 = F.relu(self.layer_2(x_1))
        x_3 = self.layer_3(x_2)
        return x_3

def imp_kernel(z_1, z_2, noise_num, noise_dim, kernel_net, cuda):  # z_1 shape (batch_size, dim_Z)
    kzz_rbf, bandwidth = rbf_kernel(z_1, z_2)  # kzz_rbf shape (batch_size, batch_size), bandwidth shape (1, 1)
    if cuda:
        # try uniform, might better
        #kernel_noise = normal.Normal(0., 1.).sample((noise_num, noise_dim)).cuda()
        kernel_noise = uniform.Uniform(-1., 1.).sample((noise_num, noise_dim)).cuda()
    else:
        kernel_noise = normal.Normal(0., 1.).sample((noise_num, noise_dim))
    kernel_omega = kernel_net(kernel_noise)  # output transferred noises
    kernel_omega_t = kernel_omega.t()  # shape (dim_Z, noise_num)
    if cuda:
        kernel_bias = torch.rand((1, noise_num)).cuda()*math.pi*2  # shape (1, noise_num)
    else:
        kernel_bias = torch.rand((1, noise_num))* math.pi * 2
    fourier_1 = torch.cos(z_1.mm(kernel_omega_t)/bandwidth + kernel_bias)*math.sqrt(2)  # shape (batch_size, noise_num)
    fourier_2 = torch.cos(z_2.mm(kernel_omega_t)/bandwidth + kernel_bias)*math.sqrt(2)  # shape (batch_size, noise_num)
    kzz_omega = fourier_1.mm(fourier_2.t())/noise_num  # shape (batch_size ,batch_size) devided by noise_num because of the matrix multiplication
    return kzz_omega + kzz_rbf

def imp_kernel_no_rbf(z_1, z_2, noise_num, noise_dim, kernel_net, cuda):  # z_1 shape (batch_size, dim_Z)
    if cuda:
        # try uniform, might better
        #kernel_noise = normal.Normal(0., 1.).sample((noise_num, noise_dim)).cuda()
        kernel_noise = uniform.Uniform(-1., 1.).sample((noise_num, noise_dim)).cuda()
    else:
        kernel_noise = normal.Normal(0., 1.).sample((noise_num, noise_dim))
    kernel_omega = kernel_net(kernel_noise)  # output transferred noises
    kernel_omega_t = kernel_omega.t()  # shape (dim_Z, noise_num)
    if cuda:
        kernel_bias = torch.rand((1, noise_num)).cuda()*math.pi*2  # shape (1, noise_num)
    else:
        kernel_bias = torch.rand((1, noise_num))* math.pi * 2
    fourier_1 = torch.cos(z_1.mm(kernel_omega_t) + kernel_bias)*math.sqrt(2)  # shape (batch_size, noise_num)
    fourier_2 = torch.cos(z_2.mm(kernel_omega_t) + kernel_bias)*math.sqrt(2)  # shape (batch_size, noise_num)
    kzz_omega = fourier_1.mm(fourier_2.t())/noise_num  # shape (batch_size ,batch_size) devided by noise_num because of the matrix multiplication
    return kzz_omega

def imp_kernel_with_bw(z_1, z_2, noise_num, noise_dim, kernel_net, cuda, sigma_list):  # z_1 shape (batch_size, dim_Z)
    if cuda:
        # try uniform, might better
        #kernel_noise = normal.Normal(0., 1.).sample((noise_num, noise_dim)).cuda()
        kernel_noise = uniform.Uniform(-1., 1.).sample((noise_num, noise_dim)).cuda()
    else:
        kernel_noise = normal.Normal(0., 1.).sample((noise_num, noise_dim))
    kernel_omega = kernel_net(kernel_noise)  # output transferred noises
    kernel_omega_t = kernel_omega.t()  # shape (dim_Z, noise_num)
    if cuda:
        kernel_bias = torch.rand((1, noise_num)).cuda()*math.pi*2  # shape (1, noise_num)
    else:
        kernel_bias = torch.rand((1, noise_num))* math.pi * 2
    
    fourier_1 = 0.0
    fourier_2 = 0.0
    for bandwidth in sigma_list:    
        fourier_1 = torch.cos(z_1.mm(kernel_omega_t)/bandwidth + kernel_bias)*math.sqrt(2)  # shape (batch_size, noise_num)
        fourier_2 = torch.cos(z_2.mm(kernel_omega_t)/bandwidth + kernel_bias)*math.sqrt(2)  # shape (batch_size, noise_num)
    kzz_omega = fourier_1.mm(fourier_2.t())/noise_num  # shape (batch_size ,batch_size) devided by noise_num because of the matrix multiplication
    return kzz_omega

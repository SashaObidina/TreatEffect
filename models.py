import torch
import numpy as np
import time
import copy
import functional as F
from torch import tensor
from torch import nn
from torch.nn.functional import pairwise_distance
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
from scipy.stats import uniform
from generation import *

default_device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

'''
p = uniform(0.1, 0.5)
linear1_output = {128, 256, 512, 1024}
linear2_output = {128, 256, 512, 1024}
linear3_output = {16, 32, 64, 128, 256}
num_heads = {4, 6, 8}
num_layers = {6, 12}
'''

class MHAlayer_exp_mahal(nn.Module):
    def __init__(self, num_features=10, num_heads=2):
        super(MHAlayer_exp_mahal, self).__init__()
        self.num_heads = num_heads

        self.batch_norm = nn.BatchNorm1d(num_features) # Batch normalization layer
        self.linear = nn.Linear(num_features, num_features) # Linear layer
        self.self_attn = F.MultiheadAttention(num_features, self.num_heads, batch_first=True, dropout=0.0) # MultiheadAttention layers

    def forward(self, X_train, y_train, treat_train, treat_train_inverted, mask):
        attn, betas = self.self_attn(X_train, X_train, X_train, attn_mask=mask)  # MultiheadAttention -> embeddings
        x1 = self.batch_norm(attn.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalization
        x2 = self.linear(x1) # Linear layer
        x3 = self.batch_norm(x2.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalization
        return x3


class MHAlayer_exp(nn.Module):
    def __init__(self, num_features=10, num_heads=2):
        super(MHAlayer_exp, self).__init__()
        self.num_heads = num_heads

        self.batch_norm = nn.BatchNorm1d(num_features) # Batch normalization layer
        self.linear = nn.Linear(num_features, num_features) # Linear layer
        self.self_attn = nn.MultiheadAttention(num_features, self.num_heads, batch_first=True, dropout=0.0) # MultiheadAttention layers

    def forward(self, X_train, y_train, treat_train, treat_train_inverted, mask):
        attn, betas = self.self_attn(X_train, X_train, X_train, attn_mask=mask)  # MultiheadAttention -> embeddings
        x1 = self.batch_norm(attn.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalization
        x2 = self.linear(x1) # Linear layer
        x3 = self.batch_norm(x2.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalization
        return x3


class MHAlayer(nn.Module):
    def __init__(self, num_features=10, num_heads=2):
        super(MHAlayer, self).__init__()
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(num_features, self.num_heads, batch_first=True) # MultiheadAttention layers

    def forward(self, X_train, y_train, treat_train, treat_train_inverted, mask):
        attn, betas = self.self_attn(X_train, X_train, X_train, attn_mask=mask)  # MultiheadAttention -> embeddings
        return attn


class ModelExp_mahal(nn.Module):
    def __init__(self, input_dim=10, p=0.1, linear1_output=128, linear2_output=128, linear3_output=16, num_heads=4, num_layers=6):
        super(ModelExp_mahal, self).__init__()
        self.fc1 = nn.Linear(input_dim, linear1_output)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(linear1_output, linear2_output)
        self.dropout = nn.Dropout(p)  # Слой Dropout для регуляризации
        self.fc3 = nn.Linear(linear2_output, linear3_output)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            MHAlayer_exp_mahal(linear3_output, num_heads) for _ in range(num_layers)
        ])
        self.batch_norm = nn.BatchNorm1d(linear3_output) # Batch normalization layer
        self.self_attn = F.MultiheadAttention(linear3_output, self.num_heads, batch_first=True) # MultiheadAttention layers

    def forward_FCNet(self, x):
        # FCNet
        if x.size(1) < self.input_dim:  # если входной размер данных меньше input_dim, зануляем недостающие признаки
            padding = torch.zeros(x.size(0), self.input_dim - x.size(1)).to(default_device)
            x = torch.cat((x, padding), dim=1)
            #print('x with padding', x.shape)
            x = x * self.input_dim / x.size(1)
            #print('x standard', x.shape)
        #print(x.size)
        x = self.fc1(x)
        #print(x.size)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def forward_MHA(self, x, y, treat, treat_inverted, mask):
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch normalization
        for layer in self.layers:
            x = layer.forward(x, y, treat, treat_inverted, mask)
        attn, betas = self.self_attn(x, x, x, attn_mask=mask)  # MultiheadAttention

        # Compute treatment effect
        l = torch.div(1, torch.einsum('npb,nb->np', betas, treat))
        r = torch.div(1, torch.einsum('npb,nb->np', betas, treat_inverted))
        # Check for NaN and replace with 1
        l = torch.where(torch.isnan(l), torch.ones_like(l), l)
        r = torch.where(torch.isnan(r), torch.ones_like(r), r)

        tr_1 = torch.mul(torch.einsum('npb,nb->np', betas, y * treat), l)
        tr_0 = torch.mul(torch.einsum('npb,nb->np', betas, y * treat_inverted), r)
        y_factual = tr_1.mul(treat) + tr_0.mul(treat_inverted)
        y_cfactual = tr_1.mul(treat_inverted) + tr_0.mul(treat)
        cate = tr_1 - tr_0

        return y_factual, y_cfactual, cate

    def forward(self, x, y, treat, treat_inverted, mask):
        x = self.forward_FCNet(x)
        y_factual, y_cfactual, cate = self.forward_MHA(x.unsqueeze(0), y.unsqueeze(0), treat.unsqueeze(0), treat_inverted.unsqueeze(0), mask)
        return y_factual, y_cfactual, cate


class ModelExp(nn.Module):
    def __init__(self, input_dim=10, p=0.1, linear1_output=128, linear2_output=128, linear3_output=16, num_heads=4, num_layers=6):
        super(ModelExp, self).__init__()
        self.fc1 = nn.Linear(input_dim, linear1_output)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(linear1_output, linear2_output)
        self.dropout = nn.Dropout(p)  # Слой Dropout для регуляризации
        self.fc3 = nn.Linear(linear2_output, linear3_output)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            MHAlayer_exp(linear3_output, num_heads) for _ in range(num_layers)
        ])
        self.batch_norm = nn.BatchNorm1d(linear3_output) # Batch normalization layer
        self.self_attn = nn.MultiheadAttention(linear3_output, self.num_heads, batch_first=True) # MultiheadAttention layers

    def forward_FCNet(self, x):
        # FCNet
        if x.size(1) < self.input_dim:  # если входной размер данных меньше input_dim, зануляем недостающие признаки
            padding = torch.zeros(x.size(0), self.input_dim - x.size(1)).to(default_device)
            x = torch.cat((x, padding), dim=1)
            #print('x with padding', x.shape)
            x = x * self.input_dim / x.size(1)
            #print('x standard', x.shape)
        #print(x.size)
        x = self.fc1(x)
        #print(x.size)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def forward_MHA(self, x, y, treat, treat_inverted, mask):
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch normalization
        for layer in self.layers:
            x = layer.forward(x, y, treat, treat_inverted, mask)
        attn, betas = self.self_attn(x, x, x, attn_mask=mask)  # MultiheadAttention

        # Compute treatment effect
        l = torch.div(1, torch.einsum('npb,nb->np', betas, treat))
        r = torch.div(1, torch.einsum('npb,nb->np', betas, treat_inverted))
        # Check for NaN and replace with 1
        l = torch.where(torch.isnan(l), torch.ones_like(l), l)
        r = torch.where(torch.isnan(r), torch.ones_like(r), r)

        tr_1 = torch.mul(torch.einsum('npb,nb->np', betas, y * treat), l)
        tr_0 = torch.mul(torch.einsum('npb,nb->np', betas, y * treat_inverted), r)
        y_factual = tr_1.mul(treat) + tr_0.mul(treat_inverted)
        y_cfactual = tr_1.mul(treat_inverted) + tr_0.mul(treat)
        cate = tr_1 - tr_0

        return y_factual, y_cfactual, cate

    def forward(self, x, y, treat, treat_inverted, mask):
        x = self.forward_FCNet(x)
        y_factual, y_cfactual, cate = self.forward_MHA(x.unsqueeze(0), y.unsqueeze(0), treat.unsqueeze(0), treat_inverted.unsqueeze(0), mask)
        return y_factual, y_cfactual, cate


class Model(nn.Module):
    def __init__(self, input_dim=100, p=0.1, linear1_output=128, linear2_output=128, linear3_output=16, num_heads=4, num_layers=6):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, linear1_output)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(linear1_output, linear2_output)
        self.dropout = nn.Dropout(p)  # Слой Dropout для регуляризации
        self.fc3 = nn.Linear(linear2_output, linear3_output)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            MHAlayer_exp(linear3_output, num_heads) for _ in range(num_layers)
        ])
        self.batch_norm = nn.BatchNorm1d(linear3_output) # Batch normalization layer
        self.self_attn = F.MultiheadAttention(linear3_output, self.num_heads, batch_first=True) # MultiheadAttention layers

    def forward_FCNet(self, x):
        # FCNet
        if x.size(1) < self.input_dim: # если входной размер данных меньше input_dim, зануляем недостающие признаки
            padding = torch.zeros(x.size(0), self.input_dim - x.size(1)).to(default_device)
            x = torch.cat((x, padding), dim=1)
            x = x * self.input_dim / x.size(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def forward_MHA(self, x, y, treat, treat_inverted, mask):
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch normalization
        for layer in self.layers:
            x = layer.forward(x, y, treat, treat_inverted, mask)
        attn, betas = self.self_attn(x, x, x, attn_mask=mask)  # MultiheadAttention

        # Compute treatment effect
        l = torch.div(1, torch.einsum('npb,nb->np', betas, treat))
        r = torch.div(1, torch.einsum('npb,nb->np', betas, treat_inverted))
        # Check for NaN and replace with 1
        l = torch.where(torch.isnan(l), torch.ones_like(l), l)
        r = torch.where(torch.isnan(r), torch.ones_like(r), r)

        tr_1 = torch.mul(torch.einsum('npb,nb->np', betas, y * treat), l)
        tr_0 = torch.mul(torch.einsum('npb,nb->np', betas, y * treat_inverted), r)
        y_factual = tr_1.mul(treat) + tr_0.mul(treat_inverted)
        y_cfactual = tr_1.mul(treat_inverted) + tr_0.mul(treat)
        cate = tr_1 - tr_0

        return y_factual, y_cfactual, cate

    def forward(self, x, y, t, t_inverted, mask):
        x = self.forward_FCNet(x)
        y_factual, y_cfactual, cate = self.forward_MHA(x.unsqueeze(0), y.unsqueeze(0), t.unsqueeze(0), t_inverted.unsqueeze(0), mask)
        return y_factual, y_cfactual, cate


def compute_inver_covariance_matrix(data):
    inv_cov_matrices = []
    for batch in data:
        covariance_matrix = torch.cov(batch)
        inverse_covariance_matrix = torch.linalg.inv(covariance_matrix)
        print('mult', torch.matmul(inverse_covariance_matrix, covariance_matrix))
        inv_cov_matrices.append(abs(inverse_covariance_matrix))
    inv_cov_matrices = torch.stack(inv_cov_matrices, dim=0)
    return inv_cov_matrices


def compute_inv_covariance_matrix(k, epsilon=1e-6):
    batch_size, feature_dim, seq_len = k.size()
    covariance_matrix = torch.zeros((batch_size, feature_dim, feature_dim), device=k.device)

    for i in range(batch_size):
        #print(k[i].shape)
        covariance_matrix[i] = torch.cov(k[i])

    # Добавление джиттера для предотвращения сингулярности
    identity_matrix = torch.eye(feature_dim, device=k.device)
    regularized_cov_matrix = covariance_matrix + epsilon * identity_matrix.unsqueeze(0)
    inv_covariance_matrix = torch.linalg.inv(regularized_cov_matrix)

    # Проверка, является ли inv_cov_matrix обратной к regularized_cov_matrix
    #product_matrix = torch.bmm(regularized_cov_matrix, inv_covariance_matrix)
    #print(product_matrix)
    # Сравнение с единичной матрицей
    #is_inverse = torch.allclose(product_matrix, identity_matrix, atol=1e-5)
    #print(is_inverse)

    return inv_covariance_matrix


#covariance_matrix = compute_inv_covariance_matrix(data.permute(0, 2, 1))


def mahalanobis_distance_matrix(query, key, cov_inv):
    B, Nq, E = query.shape
    assert key.shape[0] == B and key.shape[1] == Nq, "Number of keys must match number of queries"

    diff = query - key
    dist_matrix = torch.bmm(diff, cov_inv)
    dist_matrix = abs(torch.bmm(dist_matrix, diff.transpose(-2, -1)))

    # B x Nq x Nk
    return dist_matrix #.reshape(B, Nq, Nq)


def mahalanobis_attention(query, key, attn_mask, cov_inv):
    dist_matrix = mahalanobis_distance_matrix(query, key, cov_inv)
    dist_matrix = -torch.sqrt(dist_matrix)

    for i in range(dist_matrix.size(0)):
        finite_tensor = dist_matrix[i][attn_mask[0] != -float('inf')]

        if finite_tensor.numel() > 0:
            min_value = finite_tensor.min()
            max_value = finite_tensor.max()
            range_value = max_value - min_value
            if range_value > 0:
                dist_matrix[i] = (dist_matrix[i] - min_value) / range_value
    if attn_mask is not None:
        dist_matrix = attn_mask + dist_matrix

    return dist_matrix


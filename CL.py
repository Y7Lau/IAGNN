import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import Parameter, ReLU
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_add, gather_csr, scatter
from collections import Counter

import yaml
from yaml import SafeLoader
import copy
from copy import deepcopy
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

from typing import Optional, Tuple

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing, GATConv, GCNConv, SAGEConv
from torch_geometric import datasets
from torch_geometric.utils import to_undirected, scatter, dropout_edge
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import BatchNorm, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.dense.linear import Linear

from data_utils import get_dataset, set_random_seeds
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--judge', type=int, default=0)
parser.add_argument('--seed', type=int, default=44, help='random seed') #cora:44, pubmed:44
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'])

parser.add_argument('--hidden', type=int, default=128, help='hidden size')
parser.add_argument('--net', type=str, default='GraphSage')
parser.add_argument('--config', type=str, default='config.yaml')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.2)

args = parser.parse_args()

set_random_seeds(args.seed)


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
        # GRACE投影头
        self.proj_head = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.ELU(),
            nn.Linear(256, out_channels)
        )
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
    def project(self, z):
        return self.proj_head(z)

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def contrastive_loss(z1, z2, tau=0.5):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    
    between_sim = torch.exp(torch.mm(z1, z2.t()) / tau)
    refl_sim = torch.exp(torch.mm(z1, z1.t()) / tau)
    
    loss = -torch.log(
        between_sim.diag() / 
        (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    )
    return loss.mean()

def train(epoch, data, in_epoch, dataname):
    model.train()
    optimizer.zero_grad()
    
    edge_index1 = dropout_edge(data.edge_index, p=drop_edge_rate_1)[0]
    edge_index2 = dropout_edge(data.edge_index, p=drop_edge_rate_2)[0]
    x1 = drop_feature(data.x, drop_feature_rate_1)
    x2 = drop_feature(data.x, drop_feature_rate_2)

    z1 = model(x1, edge_index1)
    z2 = model(x2, edge_index2)

    h1 = model.project(z1)
    h2 = model.project(z2)
    loss_contrast = contrastive_loss(h1, h2)
    

    out = model(data.x, data.edge_index)
    loss_class = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    

    alpha = args.alpha

    total_loss = alpha * loss_contrast + (1-alpha) * loss_class
    

    if epoch > in_epoch:
        if dataname == 'pubmed':
            total_loss += minority_cross_entropy(out[data.train_mask], data.y[data.train_mask], minority_classes=[2])
        else:
            total_loss = minority_cross_entropy(out[data.train_mask], data.y[data.train_mask], minority_classes=[6,4,5])
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()



def test(model, data):
    model.eval()
    

    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=False)
    results = metric(data, y_pred)
    return results




device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

dataname = args.dataset
data, num_classes, dataset = get_dataset('./data', args.dataset,
                                         transform=T.NormalizeFeatures(),
                                         num_train_per_class=20)
data = data.to(device)

cf = f'{args.dataset}_{args.net}'
config = yaml.load(open(args.config), Loader=SafeLoader)[cf]
drop_edge_rate_1 = config['drop_edge_rate_1']
drop_edge_rate_2 = config['drop_edge_rate_2']
drop_feature_rate_1 = config['drop_feature_rate_1']
drop_feature_rate_2 = config['drop_feature_rate_2']
tau = config['tau']
in_epoch = config['in_epoch']
learning_rate = config['learning_rate']
args.hidden = config['num_hidden']

model = GNN(in_channels=data.x.size(1), hidden_channels=args.hidden, out_channels=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
KLloss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

best_loss = 100000000
M_L= 0
cnt = 0
patience = 80
best_val_acc = 0
for epoch in tqdm(range(1, 800)):
    loss = train(epoch, data, in_epoch, dataname)

    if epoch % 1 == 0:
        results = test(model, data)
        if results['val_acc_mean'] > best_val_acc:
            best_val_acc = results['val_acc_mean']   
            
            print("F1_scores: {:.2f}%".format(results['test_micro_f1'] * 100))
            print("BAcc: {:.2f}%".format(results['test_acc_mean'] * 100))
            print("Acc_std: {:.2f}%".format(results['test_acc_var'] * 100))
            class_f1 = results['test_class_f1']
            if class_f1 is not None:  
                M_L = torch.argmin(torch.tensor(class_f1)).item()
                for class_idx, f1 in enumerate(class_f1):
                    print(f"{class_idx:<10} {f1:.4f}")
                print("=" * 40)
                torch.save({'model': model.state_dict()}, f'./param/PT_param/{args.dataset}/{args.dataset}_model.pt')
                cnt = 0
        else:
            cnt += 1
        if cnt == patience:
            print('early stopping!!!')
            break


checkpoint = torch.load(f'./param/PT_param/{args.dataset}/{args.dataset}_model.pt',
                        map_location=device)
model.load_state_dict(checkpoint['model'], strict=True)
model.to(device)


class_thresholds = get_class_thresholds(args.dataset, args.net)

labeling_info = label_high_confidence_nodes(model, data, class_thresholds, default_threshold=0.4) #pubmed:0.4



torch.save(data,f'./output/Data_Augmentation/{args.net}/{dataname}/data_0.2.pt')

unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
unlabeled_indices = unlabeled_mask.nonzero().squeeze()

print("\nDataset Statistics:")
print("=" * 50)
print(f"{'Class':<10} {'Total':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
print("-" * 50)
for c in range(num_classes):
    total = (data.y == c).sum().item()
    T = ((data.y == c) & data.train_mask).sum().item()
    V = ((data.y == c) & data.val_mask).sum().item()
    Q = ((data.y == c) & data.test_mask).sum().item()
    print(f"{c:<10} {total:<10} {T:<10} {V:<10} {Q:<10}")
print("=" * 50)







































def find_potential_minority_nodes(model, data, majority_classes, confidence_threshold=0.9):
    """
    筛选潜在少数类候选节点：
    1. 选择所有无标记节点
    2. 排除被模型高置信预测为多数类的节点
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # 获取所有无标记节点（不在训练/验证/测试集中）
        unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
        unlabeled_indices = unlabeled_mask.nonzero().squeeze().cpu().numpy()
        
        if len(unlabeled_indices) == 0:
            return []
        
        # 获取模型预测结果
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        
        # 标记需要排除的节点（高置信多数类预测）
        exclude_mask = torch.zeros_like(unlabeled_mask, dtype=torch.bool)
        for cls in majority_classes:
            exclude_mask |= (probs.argmax(dim=1) == cls) & (probs[:, cls] > confidence_threshold)
        
        # 候选节点 = 无标记节点 - 高置信多数类节点
        candidate_mask = unlabeled_mask & ~exclude_mask
        candidate_indices = candidate_mask.nonzero().squeeze().cpu()
        
        # 处理维度问题（确保总是返回列表）
        if candidate_indices.dim() == 0:
            candidate_indices = candidate_indices.unsqueeze(0)
        return candidate_indices.tolist()
    
    
def evaluate_candidate_nodes(model, data, candidate_indices, minority_classes, device):
    """
    评估候选节点并选择置信度前90%的少数类样本加入训练集
    :param model: 训练好的模型
    :param data: PyG数据对象
    :param candidate_indices: 候选节点索引列表
    :param minority_classes: 少数类标签列表(如[4,5,6])
    :param device: 设备(CPU或GPU)
    :return: (全局准确率, 各类别准确率)
    """
    model.eval()
    with torch.no_grad():
        # 转换候选索引为张量
        candidate_indices = torch.tensor(candidate_indices, device=device)
        
        # 获取预测结果和置信度
        logits = model(data.x, data.edge_index)
        pred_probs = torch.softmax(logits, dim=1)
        pred_labels = pred_probs.argmax(dim=1)
        max_probs = pred_probs.max(dim=1).values  # 获取最大预测概率

        # 筛选候选节点结果
        candidate_labels = pred_labels[candidate_indices]
        candidate_probs = max_probs[candidate_indices]
        true_labels = data.y[candidate_indices]

        # 1. 筛选预测为少数类的样本
        minority_pred_mask = torch.isin(candidate_labels, torch.tensor(minority_classes, device=device))
        selected_indices = candidate_indices[minority_pred_mask]
        selected_labels = candidate_labels[minority_pred_mask]
        selected_probs = candidate_probs[minority_pred_mask]


        final_selected_indices = []
        final_selected_labels = []

        for cls in minority_classes:
            # 获取当前类别的所有候选节点
            cls_mask = (selected_labels == cls)
            cls_indices = selected_indices[cls_mask].tolist()
            
            # 随机抽样最多20个
            if len(cls_indices) > 1000:
                import random
                cls_indices = random.sample(cls_indices, 20)
            
            final_selected_indices.extend(cls_indices)
            final_selected_labels.extend([cls] * len(cls_indices))

        # 3. 将选中的节点加入训练集
        for idx, label in zip(final_selected_indices, final_selected_labels):
            data.train_mask[idx] = True
            data.y[idx] = label

        # 保存更新后的数据
        torch.save(data, f'./output/Data_Augmentation/{dataname}/data.pt')

        # 评估性能(仅针对少数类)
        minority_mask = torch.isin(true_labels, torch.tensor(minority_classes, device=device))
        pred_minority = candidate_labels[minority_mask]
        true_minority = true_labels[minority_mask]

        if len(true_minority) == 0:
            return 0.0, {}

        # 计算全局准确率
        global_acc = (pred_minority == true_minority).float().mean().item()

        # 计算各类别准确率
        class_acc = {}
        for cls in minority_classes:
            cls_mask = (true_minority == cls)
            if cls_mask.sum() > 0:
                class_acc[cls] = (pred_minority[cls_mask] == cls).float().mean().item()
            else:
                class_acc[cls] = 0.0

        return global_acc, class_acc

def evaluate_minority(model, data, minority_classes):
    """ 评估少数类在验证集和测试集的表现 """
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        
        # 验证集表现
        val_mask = data.val_mask & torch.isin(data.y, torch.tensor(minority_classes).to(device))
        val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
        
        # 测试集表现
        test_mask = data.test_mask & torch.isin(data.y, torch.tensor(minority_classes).to(device))
        test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
        
        # 各类别F1
        f1_scores = {}
        for cls in minority_classes:
            cls_mask = data.test_mask & (data.y == cls)
            if cls_mask.sum() > 0:
                f1_scores[cls] = f1_score(
                    data.y[cls_mask].cpu(), 
                    pred[cls_mask].cpu(), 
                    average='macro'
                )
        
        return {'val_acc': val_acc, 'test_acc': test_acc, 'f1_scores': f1_scores}
    

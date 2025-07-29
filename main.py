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

import copy
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
from torch_geometric.utils import to_undirected, scatter
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
parser.add_argument('--seed', type=int, default=44, help='random seed')
parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'citeseer', 'pubmed'])
parser.add_argument('--net', type=str, default='GraphSage')
parser.add_argument('--hidden', type=int, default=128, help='hidden size')# 128 128
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train the base model')
parser.add_argument('--k', type=int, default=2, help='k')#cora:8,3 citeseer:8,3 pubmed:3

parser.add_argument('--alpha', type=float, default=0.1, help='tolerance to stop EM algorithm')#0.1 0.1/0.3 0.1
parser.add_argument('--beta', type=float, default=0.7, help='tolerance to stop EM algorithm')# 0.7 0.7/0.5 0.9
parser.add_argument('--gamma', type=float, default=0.3, help='tolerance to stop EM algorithm')#0.3 0.3 0.3

parser.add_argument('--sigma1', type=float, default=0.5, help='tolerance to stop EM algorithm')
parser.add_argument('--sigma2', type=float, default=0.5, help='tolerance to stop EM algorithm')




parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

set_random_seeds(args.seed)
                  
    
class ReactionNet(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, args, in_channels: int, out_channels: int, bias: bool = False,
                cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
       
        self.k = args.k
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.sigma1 = args.sigma1
        self.sigma2 = args.sigma2
    
        self.dropout = args.dropout
        self.calg = 'g3'

        if args.dataset == 'pubmed':
            self.calg = 'g4'
        self.out_channels = out_channels
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.conv1 = GCNConv(in_channels, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.hidden)

        self.lin1 = Linear(in_channels, args.hidden, bias=False, weight_initializer='glorot')
        self.lin2 = Linear(args.hidden, out_channels, bias=False, weight_initializer='glorot')

        
        self.relu = ReLU()
        self.reg_params = list(self.lin1.parameters())
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        self.lin1.reset_parameters()
        self.conv1.reset_parameters()
        self.lin2.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        ed =edge_index
        
        if self.normalize:
            edgei = edge_index
            edgew = edge_weight
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edgei, edgew, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
                edge_index2, edge_weight2 = gcn_norm(  # yapf: disable
                    edgei, edgew, x.size(self.node_dim), False,
                    False, dtype=x.dtype)

                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]
            ew = edge_weight.view(-1, 1)
            ew2 = edge_weight2.view(-1, 1)
        

        
        if args.dataset =='pubmed':
            x = self.conv1(x, ed)
            x = F.relu(x)
            x = F.dropout(x,p= 0.3, training=self.training) 
            x = self.conv2(x, ed)
        else:
            x = self.lin1(x)
            
            x = F.relu(x)
            x = F.dropout(x,p= 0.3, training=self.training) 
            
            x = self.lin2(x)
            
        h = x
        
        for k in range(self.k):

            if self.calg == 'g3' or self.calg == 'cal_gradient_2':  # TODO
                g = cal_g_gradient3(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)

            elif self.calg == 'g4':
                g = cal_g_gradient4(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)

            adj = torch.sparse_coo_tensor(edge_index, edge_weight, [x.size(0), x.size(0)])
            Ax = torch.spmm(adj, x)
            Gx = torch.spmm(adj, g)

            pred_prob = F.softmax(x, dim=-1)  # [n_nodes, n_classes]
            uncertainty = - (pred_prob * torch.log(pred_prob + 1e-20)).sum(dim=1)  # [n_nodes]
            uncertainty = uncertainty / torch.log(torch.tensor(self.out_channels)).to(device)

            adaptive_beta = self.beta * (1 + 0.1 *uncertainty.unsqueeze(1))  # [n_nodes, 1]

            x = self.alpha * h + (1 - self.alpha - self.beta) * x  \
                + adaptive_beta * Ax \
                + (adaptive_beta * self.gamma) * Gx
        
        out = F.log_softmax(x, dim=-1)
        
        return x

    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # return edge_weight
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.k}, alpha={self.alpha})'

def train(epoch, data):
    model.train()
    optimizer.zero_grad()
 
    y_all = []
    y_hat = model(data.x, data.edge_index)
    y_pred = y_hat.argmax(dim=-1, keepdim=False)
    results = metric(data, y_pred)

    loss = F.cross_entropy(y_hat[data.train_mask], data.y[data.train_mask]) 


    l2_reg = torch.tensor(0.).to(device)
    
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += 5e-4 * l2_reg
    
    loss.backward()
    optimizer.step()

    return loss.item()



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
data.y = data.y.squeeze()
data.edge_index = to_undirected(data.edge_index).to(device)

edge_index = data.edge_index
data = torch.load(f'./output/Data_Augmentation/{args.net}/{dataname}/data.pt')
data = data.to(device)
data = mix_nodes(data, args.dataset, args.net)

data = data.to(device)
print(data)
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



model = ReactionNet(args, dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
KLloss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

M_L= 0
cnt = 0
patience = 100
best_val_acc = 0
for epoch in tqdm(range(1, 1999)):
    loss = train(epoch, data)
    if epoch % 1 == 0:
        results = test(model, data)
        if results['val_acc_mean'] > best_val_acc:
            best_val_acc = results['val_acc_mean']   
            
            print("F1_scores: {:.2f}%".format(results['test_micro_f1'] * 100))
            print("BAcc: {:.2f}%".format(results['test_acc_mean'] * 100))
            print("Acc_std: {:.2f}%".format(results['test_acc_var'] * 100))
            class_f1 = results['test_class_f1']
            if class_f1 is not None:  # 确保有结果
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
model.eval()

out = model(data.x, data.edge_index)

if not os.path.exists(f'./output/PT/{dataname}'):
    os.makedirs(f'./output/PT/{dataname}')
if args.judge <1 :    
    torch.save(out.detach(), f'./output/PT/{dataname}/'+dataname+f'_{args.seed}.pt')

unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)

emb = SSGC_Embedding(data.x, data.edge_index, k=5, alpha=0.5)
colors = ['#ffc0cb', '#bada55', '#008080', '#3e1f59', '#7fe5f0', '#065535', '#ffd700']
PLOT_out = out
z = TSNE(n_components=2).fit_transform(PLOT_out.cpu().detach().numpy())
y = data.y.cpu().numpy()
plt.figure(figsize=(8, 8))
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray',zorder=1)
for i in range(num_classes):
    plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i],zorder=2)

plt.savefig(f'{args.dataset}_visualization.pdf', format="pdf", dpi=800, bbox_inches='tight')

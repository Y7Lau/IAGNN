import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from collections import Counter
import copy
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.metrics import f1_score,accuracy_score  
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric import datasets
from torch_geometric.utils import to_undirected, scatter, degree, subgraph, spmm
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import BatchNorm, MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import zeros

from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_add
from torch_scatter import gather_csr, scatter

def metric(data, y_pred):

    results = {
        'train_acc': 0.0,
        'val_acc': 0.0,
        'test_acc': 0.0,
        'train_micro_f1': 0.0,
        'val_micro_f1': 0.0,
        'test_micro_f1': 0.0,
        'train_class_f1': None,  
        'val_class_f1': None,
        'test_class_f1': None,
        'train_f1_mean': 0.0,   
        'val_f1_mean': 0.0,
        'test_f1_mean': 0.0,
        'train_f1_var': 0.0,  
        'val_f1_var': 0.0,
        'test_f1_var': 0.0,
        'train_class_acc': None,
        'val_class_acc': None,
        'test_class_acc': None,
        'train_acc_mean': 0.0,   
        'val_acc_mean': 0.0,
        'test_acc_mean': 0.0,
        'train_acc_var': 0.0,  
        'val_acc_var': 0.0,
        'test_acc_var': 0.0,
    }
    for mask_name, mask in zip(['train', 'val', 'test'], [data.train_mask, data.val_mask, data.test_mask]):
        if mask.sum() == 0:  
            continue


        correct = (y_pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total
        results[f'{mask_name}_acc'] = acc


        micro_f1 = f1_score(data.y[mask].cpu(), y_pred[mask].cpu(), average='macro')
        results[f'{mask_name}_micro_f1'] = micro_f1


        class_f1 = f1_score(data.y[mask].cpu(), y_pred[mask].cpu(), average=None)
        results[f'{mask_name}_class_f1'] = class_f1.tolist()


        if class_f1.size > 0:  
            results[f'{mask_name}_f1_mean'] = np.mean(class_f1)
            results[f'{mask_name}_f1_var'] = np.var(class_f1)
            

        unique_classes = torch.unique(data.y[mask])
        class_acc = []
        for c in unique_classes:
            class_mask = (data.y[mask] == c)
            correct_class = (y_pred[mask][class_mask] == c).sum().item()
            total_class = class_mask.sum().item()
            acc_class = correct_class / total_class if total_class > 0 else 0.0
            class_acc.append(acc_class)
        
        results[f'{mask_name}_class_acc'] = class_acc


        if len(class_acc) > 0:
            results[f'{mask_name}_acc_mean'] = np.mean(class_acc)
            results[f'{mask_name}_acc_var'] = np.std(class_acc)
    return results



def feature_norm(fea):
    device = fea.device
    epsilon = 1e-12
    fea_sum = torch.norm(fea, p=1, dim=1)
    fea_inv = 1 / np.maximum(fea_sum.detach().cpu().numpy(), epsilon)
    fea_inv = torch.from_numpy(fea_inv).to(device)
    fea_norm = fea * fea_inv.view(-1, 1)

    return fea_norm

def cal_g_gradient3(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]

    onestep = scatter((x[col] - x[row]) * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')

    # onestep = scatter((x[col] - x[row]) * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    # twostep = scatter(onestep[col] * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = feature_norm(twostep)
    return twostep

def cal_g_gradient4(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    ones = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(ones, col, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    onestep = scatter(deg_inv[row].view(-1, 1) * (x[col] - x[row]), row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = feature_norm(twostep)
    return twostep


def normalize_adjacency(edge_index, num_nodes):
    edge_index, edge_weight = gcn_norm(edge_index, None, num_nodes, False,
                                       True, 'source_to_target')
    sp_adj = torch.sparse_coo_tensor(edge_index, edge_weight, 
                                     (num_nodes, num_nodes))
    return sp_adj.t()  


def SSGC_Embedding(x, edge_index, k: int, alpha: float):
    sp_adj = normalize_adjacency(edge_index, x.size(0))
    ret_x = torch.zeros_like(x)
    tmp_x = x
    for _ in range(k):
        tmp_x = spmm(sp_adj, tmp_x) 
        ret_x += (1-alpha) * tmp_x + alpha * x  
    return ret_x / k  # 平均化




def minority_cross_entropy(logits, labels, minority_classes):

    mask = torch.isin(labels, torch.tensor(minority_classes).to(labels.device))
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return F.cross_entropy(logits[mask], labels[mask])

def mix_label_nodes(data, target_label, other_labels, mix_ratio=0.8, k_per_class=1):
    
    emb = SSGC_Embedding(data.x, data.edge_index, k=6, alpha=0.5)
    data.x =emb
    train_mask = data.train_mask
    target_nodes = torch.where((data.y[train_mask] == target_label))[0]
    indices = torch.randperm(target_nodes.numel())[:14] #citeseer,pubmed:12

    target_nodes = target_nodes[indices]
    original_target_nodes = train_mask.nonzero()[target_nodes].squeeze()
    

    other_class_nodes = {label: train_mask.nonzero()[torch.where(data.y[train_mask] == label)[0]].squeeze() 
                        for label in other_labels}
    

    

    x_normalized = F.normalize(data.x, p=2, dim=1) 
    similarity_matrix = torch.mm(x_normalized, x_normalized.t())
    
    new_nodes_x = []
    new_nodes_y = []
    new_edges = []
    
    for node in original_target_nodes:

        for label, nodes in other_class_nodes.items():
            if len(nodes) == 0:
                continue
                

            node_similarities = similarity_matrix[node, nodes]
            

            _, topk_indices = torch.topk(node_similarities, 
                                       k=min(k_per_class, len(nodes)), 
                                       largest=True)
            selected_nodes = nodes[topk_indices]
            
            for neighbor in selected_nodes:

                mix_features = mix_ratio * data.x[node] + (1 - mix_ratio) * data.x[neighbor]
                new_nodes_x.append(mix_features)
                new_nodes_y.append(target_label)  
                

                node_edges = data.edge_index[:, data.edge_index[0] == node]
                neighbor_edges = data.edge_index[:, data.edge_index[0] == neighbor]
                

                if node_edges.size(1) > 0:
                    num_node_edges = max(1, int(node_edges.size(1) * mix_ratio))
                    selected_node_edges = node_edges[:, torch.randperm(node_edges.size(1))[:num_node_edges]]
                else:
                    selected_node_edges = node_edges
                    

                if neighbor_edges.size(1) > 0:
                    num_neighbor_edges = max(1, int(neighbor_edges.size(1) * (1 - mix_ratio)))
                    selected_neighbor_edges = neighbor_edges[:, torch.randperm(neighbor_edges.size(1))[:num_neighbor_edges]]
                else:
                    selected_neighbor_edges = neighbor_edges
                    
  
                new_node_idx = data.num_nodes + len(new_nodes_x) - 1  
                
                if selected_node_edges.size(1) > 0:
                    new_edges_from_node = torch.stack([
                        torch.full((selected_node_edges.size(1),), new_node_idx, device=data.edge_index.device),
                        selected_node_edges[1]
                    ])
                    new_edges.append(new_edges_from_node)
                    
                if selected_neighbor_edges.size(1) > 0:
                    new_edges_from_neighbor = torch.stack([
                        torch.full((selected_neighbor_edges.size(1),), new_node_idx, device=data.edge_index.device),
                        selected_neighbor_edges[1]
                    ])
                    new_edges.append(new_edges_from_neighbor)
    
    if not new_nodes_x:
        return data
    

    new_nodes_x = torch.stack(new_nodes_x)
    new_nodes_y = torch.tensor(new_nodes_y, device=data.y.device)
    

    data.x = torch.cat([data.x, new_nodes_x], dim=0)
    data.y = torch.cat([data.y, new_nodes_y], dim=0)
    

    if new_edges:
        new_edges = torch.cat(new_edges, dim=1)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        data.edge_index = torch.cat([data.edge_index, new_edges.flip([0])], dim=1)
    

    num_new_nodes = len(new_nodes_x)
    new_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.train_mask.device)
    new_val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.val_mask.device)
    new_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.test_mask.device)
    

    original_node_count = data.num_nodes - num_new_nodes
    new_train_mask[:original_node_count] = data.train_mask[:original_node_count]
    new_val_mask[:original_node_count] = data.val_mask[:original_node_count]
    new_test_mask[:original_node_count] = data.test_mask[:original_node_count]
    

    new_train_mask[original_node_count:] = True
    
    data.train_mask = new_train_mask
    data.val_mask = new_val_mask
    data.test_mask = new_test_mask
    
    return data

def label_high_confidence_nodes(model, data, class_thresholds=None, default_threshold=0.9):
    #gat_cora:25 ,
    print(class_thresholds)
    max_per_class=25
    model.eval()
    with torch.no_grad():

        unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
        unlabeled_indices = unlabeled_mask.nonzero().squeeze()
        
        if len(unlabeled_indices) == 0:
            return {
                'total_new_labels': 0,
                'global_accuracy': 0.0,
                'class_accuracy': {},
                'class_distribution': {}
            }
        

        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        pred_labels = probs.argmax(dim=1)
        max_probs = probs.max(dim=1).values


        

        threshold_tensor = torch.full_like(max_probs, fill_value=default_threshold, dtype=torch.float)
        for cls, thresh in class_thresholds.items():
            threshold_tensor[pred_labels == cls] = thresh
        

        high_conf_mask = (max_probs > threshold_tensor)
        candidate_indices = unlabeled_indices[high_conf_mask[unlabeled_indices]]
        candidate_labels = pred_labels[candidate_indices]
        

        selected_indices = []
        selected_labels = []
        

        unique_classes = torch.unique(candidate_labels)
        for cls in unique_classes:

            cls_mask = (candidate_labels == cls)
            cls_indices = candidate_indices[cls_mask]
            

            perm = torch.randperm(len(cls_indices))
            cls_indices = cls_indices[perm]
            

            cls_indices = cls_indices[:min(max_per_class, len(cls_indices))]
            
            selected_indices.append(cls_indices)
            selected_labels.append(torch.full((len(cls_indices),), cls.item(), device=cls_indices.device))
        
        if len(selected_indices) == 0:
            return {
                'total_new_labels': 0,
                'global_accuracy': 0.0,
                'class_accuracy': {},
                'class_distribution': {}
            }
        
        selected_indices = torch.cat(selected_indices)
        selected_labels = torch.cat(selected_labels)
        

        true_labels = data.y[selected_indices]
        

        for idx, label in zip(selected_indices, selected_labels):
            if label >= 0: #pubmed:0
                data.train_mask[idx] = True
                data.y[idx] = label


        accuracy_info = {
            'total_new_labels': len(selected_indices),
            'global_accuracy': 0.0,
            'class_accuracy': {},
            'class_distribution': {},
            'used_thresholds': class_thresholds.copy()
        }
        accuracy_info['used_thresholds']['default'] = default_threshold
        
        if len(selected_indices) > 0:

            correct = (selected_labels == true_labels).sum().item()
            accuracy_info['global_accuracy'] = correct / len(selected_indices)
            

            unique_classes = torch.unique(torch.cat([selected_labels, true_labels]))
            for cls in unique_classes:
                cls_mask = (selected_labels == cls)
                if cls_mask.sum() > 0:
   
                    cls_correct = ((selected_labels == true_labels) & cls_mask).sum().item()
                    accuracy_info['class_accuracy'][cls.item()] = cls_correct / cls_mask.sum().item()
                    

                    accuracy_info['class_distribution'][cls.item()] = cls_mask.sum().item()
        
        return accuracy_info
    
def mix_nodes(data, dataset, backbone):
    if backbone =='GAT':
        if dataset == 'cora':
            #data = mix_label_nodes(data, target_label=6, other_labels=[6],mix_ratio=0.8, k_per_class=1)
            data = mix_label_nodes(data, target_label=6, other_labels=[0,5],mix_ratio=0.4, k_per_class=2)
            data = mix_label_nodes(data, target_label=4, other_labels=[4],mix_ratio=0.7, k_per_class=1)
            data = mix_label_nodes(data, target_label=5, other_labels=[5],mix_ratio=0.8, k_per_class=2)
        if dataset == 'citeseer':
            data = mix_label_nodes(data, target_label=3, other_labels=[3],mix_ratio=0.5, k_per_class=1)
            data = mix_label_nodes(data, target_label=3, other_labels=[1,2],mix_ratio=0.2, k_per_class=1)
            data = mix_label_nodes(data, target_label=5, other_labels=[2],mix_ratio=0.6, k_per_class=1)
        if dataset == 'pubmed':
            data = mix_label_nodes(data, target_label=2, other_labels=[1,0],mix_ratio=0.7, k_per_class=2)
            
    if backbone =='GraphSage':
        if dataset == 'cora':
            print("1")
            #data = mix_label_nodes(data, target_label=6, other_labels=[6],mix_ratio=0.8, k_per_class=1)
            data = mix_label_nodes(data, target_label=6, other_labels=[0,5],mix_ratio=0.4, k_per_class=2)
            data = mix_label_nodes(data, target_label=4, other_labels=[4],mix_ratio=0.8, k_per_class=1)
            data = mix_label_nodes(data, target_label=5, other_labels=[5],mix_ratio=0.8, k_per_class=2)
        if dataset == 'citeseer':
            data = mix_label_nodes(data, target_label=3, other_labels=[3],mix_ratio=0.5, k_per_class=1)
            data = mix_label_nodes(data, target_label=3, other_labels=[1,2],mix_ratio=0.2, k_per_class=1)
            data = mix_label_nodes(data, target_label=5, other_labels=[2],mix_ratio=0.6, k_per_class=1)
        if dataset == 'pubmed':
            data = mix_label_nodes(data, target_label=2, other_labels=[1,0],mix_ratio=0.6, k_per_class=2)
    return data

def get_class_thresholds(dataset, backbone):
    if backbone =='GAT':
        if dataset == 'cora':
            class_thresholds = {
                0: 0.8,  
                1: 0.8,  
                2: 0.8,  
                3: 0.8,
                4: 0.161,
                5: 0.18,
                6: 0.16
            }
        if dataset == 'citeseer':
            class_thresholds = {
                0: 0.8,  
                1: 0.7,  
                2: 0.6, 
                3: 0.19,
                4: 0.20,
                5: 0.1
            }
        if dataset == 'pubmed':
            class_thresholds = {
                0: 0.95,  
                1: 0.80,  
                2: 0.35,  

            }
            
    if backbone =='GraphSage':
        if dataset == 'cora':
            
            class_thresholds = {
                0: 0.8,  
                1: 0.8,  
                2: 0.8,  
                3: 0.8,
                4: 0.6,
                5: 0.6,
                6: 0.15
            }
            
            '''
            class_thresholds = {
                0: 0.80,#0.92 0.8 copy copy
                1: 0.85,#0.95 0.85 copy copy
                2: 0.85,#0.95 0.85 copy copy
                3: 0.8,#0.8 0.8 copy copy
                4: 0.7,#0.7 0.7 copy copy
                5: 0.7,#0.7 0.7 copy copy
                6: 0.6#0.6 0.6 copy copy
            }
            '''
        if dataset == 'citeseer':
            class_thresholds = {
                0: 0.9,  
                1: 0.9,  
                2: 0.9, 
                3: 0.22,
                4: 0.15,
                5: 0.4
            }
        if dataset == 'pubmed':
            class_thresholds = {
                0: 0.95,   
                1: 0.80,  
                2: 0.64,  

            }
    return class_thresholds
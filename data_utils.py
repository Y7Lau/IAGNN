import random
import os

import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics.pairwise import euclidean_distances

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)



def get_dataset(root, dataname, transform=NormalizeFeatures(), num_train_per_class=20, num_val_per_class=30):
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
        'flickr': (datasets.Flickr, None),
        'cora': (datasets.Planetoid, 'Cora'),
        'citeseer': (datasets.Planetoid, 'Citeseer'),
        'pubmed': (datasets.Planetoid, 'PubMed'),
        'reddit': (datasets.Reddit2, None),
        'yelp': (datasets.Yelp, None),
    }

    assert dataname in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))
    
    dataset_class, name = pyg_dataset_dict[dataname]
    if name:
        dataset = dataset_class(root, name=name, transform=transform, split='public')
        data = dataset[0]
        print(data)
        
        for c in range(dataset.num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            train_idx = idx[data.train_mask[idx]]
            if c >= (dataset.num_classes - dataset.num_classes // 2):
         
                keep_num = max(1, len(train_idx) // 10)
                #keep_num = max(1, 10)
                data.train_mask[train_idx[keep_num:]] = False

                #train_idx = idx[perm_idx[:(num_train_per_class//10)]]

        
        #data.val_mask = ~(data.train_mask+data.test_mask)
        
            
    else:
        dataset = dataset_class(root+'/'+dataname, transform=transform)
        data = dataset[0]
    
    num_classes = dataset.num_classes
    '''
    for i in range(dataset.num_classes //2 ):
        
        L = i + (dataset.num_classes - dataset.num_classes // 2)
        data = enhance_nodes(data, label = L, num_neighbors=2, interpolation_alpha=1)
    '''
    '''
    #num_neighbors:
        cora: 2 
        citeseer : 2
        pubmed: 3

    #interpolation_alpha: 
        cora: 0.5 
        citeseer: 0.5
        pubmed :0.7
    '''
    print("\nDataset Statistics:")
    print("=" * 50)
    print(f"{'Class':<10} {'Total':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 50)
    for c in range(num_classes):
        total = (data.y == c).sum().item()
        train = ((data.y == c) & data.train_mask).sum().item()
        val = ((data.y == c) & data.val_mask).sum().item()
        test = ((data.y == c) & data.test_mask).sum().item()
        print(f"{c:<10} {total:<10} {train:<10} {val:<10} {test:<10}")
    print("=" * 50)
    
    
    print(data)
    return data, dataset.num_classes, dataset



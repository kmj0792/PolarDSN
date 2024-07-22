import numpy as np
import torch
import os
import random
import argparse
import sys
from torch_geometric.utils import negative_sampling

def get_args():
    parser = argparse.ArgumentParser('Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit',
                        choices=['wiki-RfA', 'epinions', 'bitcoinalpha', 'bitcoinotc'],
                        default='bitcoinotc')
    
    parser.add_argument('--pos_dim', type=int, default=64, help='dimension of the positional embedding')
    parser.add_argument('--time_dim', type=int, default=64, help='dimension of the positional embedding')
    parser.add_argument('--task', type=str, default='link_sign', help='link or link_sign or sign') 
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--early_stop', type=int, default=5, help='link or link_sign or sign')

    # method-related hyper-parameters
    parser.add_argument('--n_degree', type=int, default=32, 
                        help='number of paths(i.e., alpha)')
    parser.add_argument('--n_layer', type=int, default=3, help='length of each path(i.e., beta)')  
    parser.add_argument('--bias', default=1e-6, type=float, help='temporal decay(i.e., gamma)')
   
    # options
    parser.add_argument('--train_time_encoding', type=str, default='learn',  choices=['learn', 'nonlearn'], help='bitcoin-alpha and epinions: learn / bitcoin-otc and wiki-RfA: nonlearn')
    parser.add_argument('--pos_sample', type=str, default='binary', choices=['multinomial', 'binary'], help='two equivalent sampling method with empirically different running time')
    parser.add_argument('--edge_embedding', type=str, default='concat', choices=['concat', 'mean'], help='two equivalent sampling method with empirically different running time')
    parser.add_argument('--neigh_agg', type=str, default='rnn', choices=['rnn', 'attn', 'new_attn'], help='two type of neighbors aggregator method')
    parser.add_argument('--path_agg', type=str, default='mean', choices=['mean', 'attn'], help='two type of paths aggregator method')
    parser.add_argument('--walk_type', type=str, default='before', choices=['before', 'point'], help='two type of temporal random walk type')
    parser.add_argument('--direct', type=str, default='add', choices=['add', 'non'], help='directed temporal randomwalk?')
    parser.add_argument('--co_occ', type=str, default='learn_add',  choices=['learn_add', 'non'], help='add co-occurrence feature?')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--tolerance', type=float, default=0, help='tolerated marginal improvement for early stopper')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, num_nodes, seed=None):
        self.seed = None
        self.num_node = num_nodes
        self.src_list = np.concatenate(src_list)  
        self.dst_list = np.concatenate(dst_list)
        self.src_list_uni = np.unique(self.src_list)
        self.dst_list_uni = np.unique(self.dst_list)
        self.edge = list(zip(self.src_list, self.dst_list))   
        self.edge_list = torch.stack([torch.Tensor(self.src_list), torch.Tensor(self.dst_list)])

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)
        
    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list_uni), size)
            dst_index = np.random.randint(0, len(self.dst_list_uni), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list_uni), size)
            dst_index = self.random_state.randint(0, len(self.dst_list_uni), size)
        return self.src_list_uni[src_index], self.dst_list_uni[dst_index]
    
    def sample_semba(self, size): 
        null_ei_batch = negative_sampling(self.edge_list, num_nodes=self.num_node, num_neg_samples=size)
        
        while ((null_ei_batch==0).any()) == True:
            null_ei_batch = negative_sampling(self.edge_list, num_nodes=self.num_node, num_neg_samples=size)
           
        return np.array(null_ei_batch[0]), np.array(null_ei_batch[1])
    
    def sample_mj(self, src, dst, size): 
        null_dst_batch = []
        for i in range(size):
            u = src[i]
            v = random.choice(range(self.num_node))+ 1
            while (u,v) in self.edge:
                v = random.choice(range(self.num_node)) + 1
            null_dst_batch.append(v)
          
        return src, np.array(null_dst_batch)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

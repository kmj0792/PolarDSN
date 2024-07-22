import logging
import time
import numpy as np
import torch
import multiprocessing as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from position import *
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import math
import pandas as pd


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, edge_embedding_type):
        super().__init__()

        self.edge_embedding_type = edge_embedding_type
        if self.edge_embedding_type == 'concat':
            self.fc1 = torch.nn.Linear(dim1 + dim2, dim3) 
        elif self.edge_embedding_type == 'mean': 
            self.fc1 = torch.nn.Linear(dim1, dim3) 
        self.fc2 = torch.nn.Linear(dim3, dim4) 
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.edge_embedding_type == 'concat':
            x = torch.cat([x1, x2], dim=-1)
        elif self.edge_embedding_type == 'mean':
            x = x1 + x2
       
        h = self.act(self.fc1(x)) 
        z = self.fc2(h)

        return z, z_walk
    
class MergeLayer_(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, edge_embedding_type):
        super().__init__()
       
        self.edge_embedding_type = edge_embedding_type
        if self.edge_embedding_type == 'concat':
            self.fc1 = torch.nn.Linear(dim1 + dim2 + dim3, dim3)
        elif self.edge_embedding_type == 'mean': 
            self.fc1 = torch.nn.Linear(dim1, dim3) 
        self.fc2 = torch.nn.Linear(dim3, dim4) 

        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

       
    def forward(self, x1, x2, x3):
        z_walk = None
        if self.edge_embedding_type == 'concat':
            x_ = torch.cat([x1, x2], dim=-1)
            x = torch.cat([x_, x3], dim=-1)
        elif self.edge_embedding_type == 'mean':
            x = x1 + x2 + x3
        h = self.act(self.fc1(x))
        z = self.fc2(h) 

        return z, z_walk

class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        
        attn = torch.bmm(q, k.transpose(-1, -2)) 
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) 
        attn = self.dropout(attn) 

        output = torch.bmm(attn, v)  
        return output, attn


class MultiHeadAttention(nn.Module):


    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        B, N_src, _ = q.size()
        B, N_ngh, _ = k.size() 
        B, N_ngh, _ = v.size() 
        assert(N_ngh % N_src == 0)
        num_neighbors = int(N_ngh / N_src)
        residual = q

        q = self.w_qs(q).view(B, N_src, 1, n_head, d_k)  
        k = self.w_ks(k).view(B, N_src, num_neighbors, n_head, d_k)  
        v = self.w_vs(v).view(B, N_src, num_neighbors, n_head, d_v) 

        q = q.transpose(2, 3).contiguous().view(B*N_src*n_head, 1, d_k)  
        k = k.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, d_k) 
        v = v.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, d_v)  
        mask = mask.view(B*N_src, 1, num_neighbors).repeat(n_head, 1, 1) 
        output, attn_map = self.attention(q, k, v, mask=mask) 

        output = output.view(B, N_src, n_head*d_v)  
        output = self.dropout(self.fc(output))  
        output = self.layer_norm(output + residual)  
        attn_map = attn_map.view(B, N_src, n_head, num_neighbors)
        return output, attn_map


class MapBasedMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) 
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        k = torch.unsqueeze(k, dim=1) 
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) 

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) 

        mask = mask.repeat(n_head, 1, 1) 

        q_k = torch.cat([q, k], dim=3) 
        attn = self.weight_map(q_k).squeeze(dim=3) 

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) 
        attn = self.dropout(attn) 

        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) 

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn


def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, time_encoder_type, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.alpha = math.sqrt(self.time_dim)
        self.beta = math.sqrt(self.time_dim)
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
        self.parameter_requires_grad = time_encoder_type
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / self.alpha ** np.linspace(0, self.beta-1, self.time_dim))).float())

        self.w = nn.Linear(1, self.time_dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / self.alpha ** np.linspace(0, self.beta-1, self.time_dim))).float().reshape(self.time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.time_dim))

        if not self.parameter_requires_grad == 'nonlearn': 
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False


    def forward(self, ts):
        if self.parameter_requires_grad == 'learn':
            batch_size = ts.size(0)
            seq_len = ts.size(1)

            ts = ts.view(batch_size, seq_len, 1) 
            map_ts = ts * self.basis_freq.view(1, 1, -1)  
            map_ts += self.phase.view(1, 1, -1)

            output = torch.cos(map_ts)

        elif self.parameter_requires_grad == 'nonlearn':
            ts = ts.unsqueeze(dim=2)
            output = torch.cos(self.w(ts))
        
        return output 

class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        self.att_dim = feat_dim + edge_dim + time_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)

        _, (hn, _) = self.lstm(seq_x)

        hn = hn[-1, :, :]

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) 
        hn = seq_x.mean(dim=1) 
        output = self.merger(hn, src_x)
        return output, None


class POLAR(torch.nn.Module):
    def __init__(self, task='link_sign', time_dim=64, pos_dim=32, edge_feature_dim = 1,
                 num_layers=3, num_neighbors=64, 
                 get_checkpoint_path=None, time_encoder_type = 'learn', edge_embedding_type = 'concat', neigh_agg='rnn', path_agg='mean', co_occurence = 'learn_add', node_num=0):
        super(POLAR, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.task = task
        self.num_neighbors, self.num_layers = num_neighbors, num_layers
        self.ngh_finder = None
        self.neigh_agg = neigh_agg
        self.path_agg = path_agg
        self.node_num = node_num
        self.co_occurence = co_occurence

        self.time_dim = time_dim 
        self.pos_dim = pos_dim
        self.edge_feature_dim = edge_feature_dim
        self.model_dim = self.pos_dim // 2
        self.feat_dim = self.time_dim + self.pos_dim + self.edge_feature_dim 
        self.logger.info('neighbors: {}, relativeID dim: {}, time dim: {}, feature dim: {}'.format(self.num_neighbors, self.pos_dim, self.time_dim, self.edge_feature_dim))

        self.time_encoder_type = time_encoder_type
        self.time_encoder = self.init_time_encoder()
        
        self.position_encoder = PositionEncoder(enc_dim=self.pos_dim, num_layers=self.num_layers, ngh_finder=self.ngh_finder,
                                                logger=self.logger)


        self.random_walk_attn_model = RandomWalkAttention(feat_dim=self.feat_dim, pos_dim=self.pos_dim,
                                                     model_dim=self.model_dim,
                                                     logger=self.logger, neigh_agg=self.neigh_agg, path_agg=self.path_agg)

        if self.task == 'link': 
            self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1, edge_embedding_type) 
        elif self.task =="link_sign":
            self.affinity_score = MergeLayer(self.pos_dim, self.pos_dim, self.pos_dim, 3, edge_embedding_type) 
            self.affinity_score_ = MergeLayer_(self.pos_dim, self.pos_dim,  self.pos_dim, 3, edge_embedding_type) 
        self.get_checkpoint_path = get_checkpoint_path

        self.flag_for_cur_edge = True  
        self.common_node_percentages = {'pos': [], 'neg': []}
        self.walk_encodings_scores = {'encodings_p': [], 'scores': []}

        self.common_ngh_encoder = torch.nn.Linear(4, self.pos_dim, bias=True)
        nn.init.xavier_uniform_(self.common_ngh_encoder.weight)
        nn.init.zeros_(self.common_ngh_encoder.bias)
    
    def init_time_encoder(self):
        self.logger.info('Using time encoding')
        time_encoder = TimeEncode(expand_dim=self.time_dim, time_encoder_type=self.time_encoder_type)
    
        return time_encoder

    def update_mask(self, subgraph, i):
        mask_out = ((subgraph[5][i] == 1) & (subgraph[5][i+1] == 1) & (subgraph[3][i] == 1) & (subgraph[3][i+1] == 1)) \
            | ((subgraph[5][i] == 1) & (subgraph[5][i+1] == -1) & (subgraph[3][i] == -1) & (subgraph[3][i+1] == 1)) \
            | ((subgraph[5][i] == -1) & (subgraph[5][i+1] == 1) & (subgraph[3][i] == 1) & (subgraph[3][i+1] == -1)) \
            | ((subgraph[5][i] == -1) & (subgraph[5][i+1] == -1) & (subgraph[3][i] == -1) & (subgraph[3][i+1] == -1))
        
        mask_in = ((subgraph[5][i] == 1) & (subgraph[5][i+1] == 1) & (subgraph[3][i] == -1)  & (subgraph[3][i+1] == -1)) \
                    | ((subgraph[5][i] == 1) & (subgraph[5][i+1] ==  -1) & (subgraph[3][i] == 1)  & (subgraph[3][i+1] == -1)) \
                    | ((subgraph[5][i] ==  -1) & (subgraph[5][i+1] == 1) & (subgraph[3][i] == -1)  & (subgraph[3][i+1] == 1)) \
                    | ((subgraph[5][i] ==  -1) & (subgraph[5][i+1] ==  -1) & (subgraph[3][i] == 1)  & (subgraph[3][i+1] == 1)) 
        
        mask_bi = ((subgraph[5][i] == 1) & (subgraph[5][i+1] == 1) & (subgraph[3][i] == 1)  & (subgraph[3][i+1] == -1)) \
                    | ((subgraph[5][i] == 1) & (subgraph[5][i+1] ==  -1) & (subgraph[3][i] == 1)  & (subgraph[3][i+1] == 1)) \
                    | ((subgraph[5][i] ==  1) & (subgraph[5][i+1] == 1) & (subgraph[3][i] == -1)  & (subgraph[3][i+1] == 1)) \
                    | ((subgraph[5][i] ==  1) & (subgraph[5][i+1] == -1) & (subgraph[3][i] == -1)  & (subgraph[3][i+1] == -1)) \
                    | ((subgraph[5][i] ==  -1) & (subgraph[5][i+1] == 1) & (subgraph[3][i] == 1)  & (subgraph[3][i+1] == 1)) \
                    | ((subgraph[5][i] ==  -1) & (subgraph[5][i+1] ==  -1) & (subgraph[3][i] == 1)  & (subgraph[3][i+1] == -1)) \
                    | ((subgraph[5][i] ==  -1) & (subgraph[5][i+1] == 1) & (subgraph[3][i] == -1)  & (subgraph[3][i+1] == -1)) \
                    | ((subgraph[5][i] ==  -1) & (subgraph[5][i+1] == -1) & (subgraph[3][i] == -1)  & (subgraph[3][i+1] == 1))

        subgraph[5][i+1][mask_out] = 1
        subgraph[5][i+1][mask_in] = -1
        subgraph[5][i+1][mask_bi] = 2
    
    def contrast_for_multiclass(self, adj_list, src_idx_l, dst_idx_l, src_idx_l_z, dst_idx_l_z, cut_time_l, label_l_cut, e_idx_l=None, test=False):
        start = time.time()

        subgraph_src = self.grab_subgraph(src_idx_l, cut_time_l, label_l_cut, e_idx_l=e_idx_l) 
        subgraph_tgt = self.grab_subgraph(dst_idx_l, cut_time_l, label_l_cut, e_idx_l=e_idx_l) 

        subgraph_tgt_z = self.grab_subgraph(dst_idx_l_z, cut_time_l, label_l_cut, e_idx_l=None) 

        for i in range(len(subgraph_src[3])-1):
            subgraph_src[3][i+1] = subgraph_src[3][i] * subgraph_src[3][i+1]
            subgraph_tgt[3][i+1] = subgraph_tgt[3][i] * subgraph_tgt[3][i+1]
            subgraph_tgt_z[3][i+1] = subgraph_tgt_z[3][i] * subgraph_tgt_z[3][i+1]


        for i in range(len(subgraph_src[4])-1, -1, -1): 
            a = subgraph_src[4][i].copy()
            b = subgraph_tgt[4][i].copy()
            c = subgraph_tgt_z[4][i].copy()
            cnt = 1
            for j in range(i): 
               a +=  subgraph_src[4][j]
               b +=  subgraph_tgt[4][j]
               c +=  subgraph_tgt_z[4][j]
               cnt += 1
            massssk1 = subgraph_src[4][i] == 0
            massssk2 = subgraph_tgt[4][i] == 0
            massssk3 = subgraph_tgt_z[4][i] == 0 
            a[massssk1] = 0
            b[massssk2] = 0
            c[massssk3] = 0
            subgraph_src[4][i] = a / float(cnt)
            subgraph_tgt[4][i] = b / float(cnt)
            subgraph_tgt_z[4][i] = c / float(cnt)

    
        for i in range(len(subgraph_src[5])-1):
            self.update_mask(subgraph_src, i)
            self.update_mask(subgraph_tgt, i)
            self.update_mask(subgraph_tgt_z, i)

        
        end = time.time()

        self.flag_for_cur_edge = True 
        emb = self.forward(adj_list, src_idx_l, dst_idx_l, cut_time_l, (subgraph_src, subgraph_tgt), test=test) 

        self.flag_for_cur_edge = False 
    
        zero_emb = self.forward(adj_list, src_idx_l_z, dst_idx_l_z, cut_time_l, (subgraph_src, subgraph_tgt_z), test=test)
    
        return emb, zero_emb 
    
    def generate_edge_embedding(self, adj_list, src_idx_l, tgt_idx_l, cut_time_l):
        edge_emb = []
        for i, (u, v, t) in enumerate(zip(src_idx_l, tgt_idx_l, cut_time_l)):
            src_ngh = adj_list[u]
            dst_ngh = adj_list[v]
            src_nghs_nid, src_nghs_eid, src_nghs_ts, src_nghs_sign, src_nghs_weight, src_nghs_direction = zip(*src_ngh)
            dst_nghs_nid, dst_nghs_eid, dst_nghs_ts, dst_nghs_sign, dst_nghs_weight, dst_nghs_direction = zip(*dst_ngh)
            src_ts_mask = src_nghs_ts < t
            dst_ts_mask = dst_nghs_ts < t
            src_nghs_nid_, src_nghs_sign_ = np.array(src_nghs_nid)[src_ts_mask], np.array(src_nghs_sign)[src_ts_mask]
            dst_nghs_nid_, dst_nghs_sign_ = np.array(dst_nghs_nid)[dst_ts_mask], np.array(dst_nghs_sign)[dst_ts_mask]
            
            src_nghs_nid_ = src_nghs_nid_.astype(np.int64)
            dst_nghs_nid_ = dst_nghs_nid_.astype(np.int64)
            
            unique_nghs_num = len(set(np.concatenate((src_nghs_nid_, dst_nghs_nid_), axis=0)))

            if unique_nghs_num > 0:
                df_src = pd.DataFrame({'id': src_nghs_nid_, 'sign_s':src_nghs_sign_})
                df_dst = pd.DataFrame({'id': dst_nghs_nid_, 'sign_d':dst_nghs_sign_})
                
                df_src = df_src.drop_duplicates(['id'], keep='last')
                df_dst = df_dst.drop_duplicates(['id'], keep='last')
                common_ngh = pd.merge(df_src, df_dst) 
                common_same_ngh = len(common_ngh.loc[common_ngh.sign_s == common_ngh.sign_d])
                common_opp_ngh = len(common_ngh.loc[common_ngh.sign_s != common_ngh.sign_d]) 
                common = common_same_ngh + common_opp_ngh 
                if common == 0: 
                    emb = [0, 0, 0, unique_nghs_num / self.node_num]
                else:
                    emb = [common_same_ngh/common, common_opp_ngh/common, common / self.node_num, unique_nghs_num / self.node_num]
            else:
                emb = [0,0,0,0]
            edge_emb.append(emb)
        return edge_emb


    def forward(self, adj_list, src_idx_l, tgt_idx_l, cut_time_l, subgraphs=None, test=False):  
        if subgraphs is not None:
            subgraph_src, subgraph_tgt = subgraphs 
        else: 
            subgraph_src = self.grab_subgraph(src_idx_l, cut_time_l, e_idx_l=None)  
            subgraph_tgt = self.grab_subgraph(tgt_idx_l, cut_time_l, e_idx_l=None) 
        self.position_encoder.init_internal_data(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt) 
        
        subgraph_src_ = self.subgraph_tree2walk(src_idx_l, cut_time_l, subgraph_src)
        subgraph_tgt_ = self.subgraph_tree2walk(tgt_idx_l, cut_time_l, subgraph_tgt)
        src_embed_pos_neg = self.forward_msg(src_idx_l, cut_time_l, subgraph_src_, test=test) 
        tgt_embed_pos_neg = self.forward_msg(tgt_idx_l, cut_time_l, subgraph_tgt_, test=test) 
        
        src_embed = torch.cat([src_embed_pos_neg[0], src_embed_pos_neg[1]], dim=1) 
        tgt_embed = torch.cat([tgt_embed_pos_neg[0], tgt_embed_pos_neg[1]], dim=1)

        if self.co_occurence  == 'non':
            score, score_walk = self.affinity_score(src_embed, tgt_embed) 
        elif  self.co_occurence  == 'learn_add':
            edge_embed = self.generate_edge_embedding(adj_list, src_idx_l, tgt_idx_l, cut_time_l)
            device = src_embed.device
            edge_embed = torch.Tensor(edge_embed).to(device)
            edge_embed = self.common_ngh_encoder(edge_embed)
            score, score_walk = self.affinity_score_(src_embed, tgt_embed, edge_embed)
        if self.task == 'link':
            score.squeeze_(dim=-1)
 
        return score
    
    def grab_subgraph(self, src_idx_l, cut_time_l, label_l_cut, e_idx_l=None):
        subgraph = self.ngh_finder.find_k_hop(self.num_layers, src_idx_l, cut_time_l, label_l_cut, num_neighbors=self.num_neighbors, e_idx_l=e_idx_l)
        return subgraph

    def subgraph_tree2walk(self, src_idx_l, cut_time_l, subgraph_src):

        node_records, eidx_records, t_records, sign_records, weight_records, direction_records = subgraph_src
        node_records_tmp = [np.expand_dims(src_idx_l, 1)] + node_records
        eidx_records_tmp = [np.zeros_like(node_records_tmp[0])] + eidx_records
        t_records_tmp = [np.expand_dims(cut_time_l, 1)] + t_records
        sign_records_tmp = [np.ones_like(node_records_tmp[0])] + sign_records
        weight_records_tmp = [np.ones_like(node_records_tmp[0])] + weight_records
        direction_records_tmp = [2*np.ones_like(node_records_tmp[0])] + direction_records 

  
        new_node_records = self.subgraph_tree2walk_one_component(node_records_tmp)
        new_eidx_records = self.subgraph_tree2walk_one_component(eidx_records_tmp)
        new_t_records = self.subgraph_tree2walk_one_component(t_records_tmp)
        new_sign_records = self.subgraph_tree2walk_one_component(sign_records_tmp)
        new_weight_records = self.subgraph_tree2walk_one_component(weight_records_tmp)
        new_direction_records = self.subgraph_tree2walk_one_component(direction_records_tmp)

        return new_node_records, new_eidx_records, new_t_records, new_sign_records, new_weight_records, new_direction_records

    def subgraph_tree2walk_one_component(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(record_list), record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            assert(n_walks % hop_record.shape[-1] == 0)
            record_matrix[:, :, hop_idx] = np.repeat(hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix

    def forward_msg(self, src_idx_l, cut_time_l, subgraph_src_, test=False):

        node_records, eidx_records, t_records, sign_records, weight_records, direction_records = subgraph_src_  
        masks = self.get_mask(node_records)
        time_features = self.retrieve_time_features(cut_time_l, t_records)  
        
        position_pos_features, position_neg_features = self.retrieve_position_features(src_idx_l, node_records, cut_time_l, t_records, sign_records, test=test) 
        final_node_embeddings = self.forward_msg_walk_di(time_features, position_pos_features, position_neg_features, sign_records, weight_records, direction_records, masks) 
           
        return final_node_embeddings


    
    def get_mask(self, node_records):
        device = next(self.parameters()).device
        node_records_th = torch.from_numpy(node_records).long().to(device)
        masks = (node_records_th != 0).sum(dim=-1).long()  
        return masks
    
    def retrieve_time_features(self, cut_time_l, t_records):
        device = next(self.parameters()).device
        batch = len(cut_time_l)
        path_length = np.shape(t_records)[2]

        t_records_th = torch.from_numpy(t_records).float().to(device) 
        t_records_th = t_records_th.select(dim=-1, index=0).unsqueeze(dim=2) - t_records_th 
        
        n_walk, len_walk = t_records_th.size(1), t_records_th.size(2)
        time_features = self.time_encoder(t_records_th.view(batch, -1)).view(batch, n_walk, len_walk, self.time_encoder.time_dim)
        
        return time_features

    

    def retrieve_position_features(self, src_idx_l, node_records, cut_time_l, t_records, s_records, test=False):
        start = time.time()
        encode = self.position_encoder 

        if encode.enc_dim == 0:
            return None
        batch, n_walk, len_walk = node_records.shape
        node_records_r, t_records_r, s_records_r = node_records.reshape(batch, -1), t_records.reshape(batch, -1), s_records.reshape(batch, -1)
       
        position_features_p, common_nodes_p, position_features_n, common_nodes_n = encode(node_records_r, t_records_r, s_records_r) 
        position_features_p = position_features_p.view(batch, n_walk, len_walk, self.pos_dim) 
        position_features_n = position_features_n.view(batch, n_walk, len_walk, self.pos_dim) 
        self.update_common_node_percentages(common_nodes_p)
        self.update_common_node_percentages(common_nodes_n)
    
        end = time.time()
        return position_features_p, position_features_n  
    

    def forward_msg_walk(self, time_features, position_pos_features, position_neg_features, sign, weight, masks):
        emb = self.random_walk_attn_model.forward_one_node(time_features, position_pos_features, position_neg_features, sign, weight, masks)
        return emb 
    

    def forward_msg_walk_di(self, time_features, position_pos_features, position_neg_features, sign, weight, direction, masks):
        emb = self.random_walk_attn_model.forward_one_node_di(time_features, position_pos_features, position_neg_features, sign, weight, direction, masks)
        return emb 
    
    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder
        self.position_encoder.ngh_finder = ngh_finder

    def update_common_node_percentages(self, common_node_percentage):
        if self.flag_for_cur_edge:
            self.common_node_percentages['pos'].append(common_node_percentage)
        else:
            self.common_node_percentages['neg'].append(common_node_percentage)

    def save_common_node_percentages(self, dir):
        torch.save(self.common_node_percentages, dir + '/common_node_percentages.pt')

    def save_walk_encodings_scores(self, dir):
        torch.save(self.walk_encodings_scores, dir + '/walk_encodings_scores.pt')


class PositionEncoder(nn.Module):
    def __init__(self, num_layers, enc_dim=2, ngh_finder=None, logger=None):
        super(PositionEncoder, self).__init__()
        self.enc_dim = enc_dim
        self.num_layers = num_layers
        self.nodetime2emb_maps = None
        self.projection = nn.Linear(1, 1) 

        self.ngh_finder = ngh_finder
        self.logger = logger

        self.trainable_embedding = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))  

    def init_internal_data(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
        if self.enc_dim == 0:
            return
        start = time.time()
   
        self.nodetime2emb_maps = self.collect_pos_mapping_ptree(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt)
        end = time.time()
       
    def collect_pos_mapping_ptree(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt): 
       
        subgraph_src_node, _, subgraph_src_ts, subgraph_src_sign, subgraph_src_weight, subgraph_src_direction = subgraph_src  
        subgraph_tgt_node, _, subgraph_tgt_ts, subgraph_tgt_sign, subgraph_tgt_weight, subgraph_tgt_direction = subgraph_tgt
        nodetime2emb_maps = {} 
        for row in range(len(src_idx_l)):
            src = src_idx_l[row]
            tgt = tgt_idx_l[row]
            cut_time = cut_time_l[row]

            src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
            src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
            src_neighbors_relation = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_sign] 

            tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
            tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
            tgt_neighbors_relation = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_sign] 


            nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
                                                                src_neighbors_node, src_neighbors_ts, src_neighbors_relation, 
                                                                tgt_neighbors_node, tgt_neighbors_ts, tgt_neighbors_relation, batch_idx=row)
            nodetime2emb_maps.update(nodetime2emb_map)
        
        return nodetime2emb_maps

    @staticmethod
    def collect_pos_mapping_ptree_sample(src, tgt, cut_time, src_neighbors_node, src_neighbors_ts, src_neighbors_sign,
                                         tgt_neighbors_node, tgt_neighbors_ts, tgt_neighbors_sign, batch_idx):

        n_hop = len(src_neighbors_node) 
        makekey = nodets2key 
        nodetime2emb = {}

        src_neighbors_node, src_neighbors_ts, src_neighbors_sign = [[src]] + src_neighbors_node, [[cut_time]] + src_neighbors_ts, [[1]] + src_neighbors_sign
        tgt_neighbors_node, tgt_neighbors_ts, tgt_neighbors_sign = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts, [[1]] + tgt_neighbors_sign
        for k in range(n_hop+1):
            k_hop_total = len(src_neighbors_node[k])
            for src_node, src_ts, src_rel, tgt_node, tgt_ts, tgt_rel in zip(src_neighbors_node[k], src_neighbors_ts[k], src_neighbors_sign[k],
                                                            tgt_neighbors_node[k], tgt_neighbors_ts[k], tgt_neighbors_sign[k]):
                src_key, tgt_key = makekey(batch_idx, src_node, src_ts, src_rel), makekey(batch_idx, tgt_node, tgt_ts, tgt_rel) 

                if src_key not in nodetime2emb:
                    nodetime2emb[src_key] = np.zeros((2, n_hop+1), dtype=np.float32) 
                if tgt_key not in nodetime2emb:
                    nodetime2emb[tgt_key] = np.zeros((2, n_hop+1), dtype=np.float32)

                nodetime2emb[src_key][0, k] += 1/k_hop_total   
                nodetime2emb[tgt_key][1, k] += 1/k_hop_total   
        null_key = makekey(batch_idx, 0, 0.0, 0)
        nodetime2emb[null_key] = np.zeros((2, n_hop + 1), dtype=np.float32)
            
        return nodetime2emb

    def forward(self, node_record, t_record, s_record):
        device = next(self.projection.parameters()).device
   
        batched_keys_pos, batched_keys_neg = make_batched_keys(node_record, t_record, s_record)
     
        unique_p, inv_p = np.unique(batched_keys_pos, return_inverse=True) 
        unique_n, inv_n = np.unique(batched_keys_neg, return_inverse=True)
       

        unordered_encodings_p, unordered_encodings_n = [], []
        for key in unique_p:
            if key in self.nodetime2emb_maps:
                unordered_encodings_p.append(self.nodetime2emb_maps[key]) #
            else: 
                unordered_encodings_p.append(np.zeros([2, np.shape(self.nodetime2emb_maps['0-0-0-0'])[1]]))  
        unordered_encodings_p = np.array(unordered_encodings_p)
        encodings_p = unordered_encodings_p[inv_p, :]
        encodings_p = torch.tensor(encodings_p).to(device)

        for key in unique_n:
            if key in self.nodetime2emb_maps:
                unordered_encodings_n.append(self.nodetime2emb_maps[key])
            else:
                unordered_encodings_n.append(np.zeros([2, np.shape(unordered_encodings_p[0])[1]])) 
        unordered_encodings_n = np.array(unordered_encodings_n)
        encodings_n = unordered_encodings_n[inv_n, :]
        encodings_n = torch.tensor(encodings_n).to(device)


        common_nodes_p = (((encodings_p.sum(-1) > 0).sum(-1) == 2).sum().float() / (encodings_p.shape[0] * encodings_p.shape[1])).item()
        encodings_p = self.get_trainable_encodings(encodings_p) 

        common_nodes_n = (((encodings_n.sum(-1) > 0).sum(-1) == 2).sum().float() / (encodings_n.shape[0] * encodings_n.shape[1])).item()
        encodings_n = self.get_trainable_encodings(encodings_n) 
        return encodings_p, common_nodes_p, encodings_n, common_nodes_n

    @staticmethod
    def collect_pos_mapping_ptree_sample_mp(args):
        src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt, row, enc = args
        subgraph_src_node, _, subgraph_src_ts = subgraph_src 
        subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
        src = src_idx_l[row]
        tgt = tgt_idx_l[row]
        cut_time = cut_time_l[row]
        src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
        src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
        tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
        tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
        nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
                                                                            src_neighbors_node, src_neighbors_ts,
                                                                            tgt_neighbors_node, tgt_neighbors_ts, enc=enc)
        return nodetime2emb_map

    def get_trainable_encodings(self, encodings_p):
        encodings_p = self.trainable_embedding(encodings_p.float())   
        encodings_p = encodings_p.sum(dim=-2)  
    
        return encodings_p


class RandomWalkAttention(nn.Module):
    def __init__(self, feat_dim, pos_dim, model_dim, logger,  neigh_agg, path_agg):

        super(RandomWalkAttention, self).__init__()
        self.feat_dim = feat_dim 
        self.pos_dim = pos_dim 
        self.model_dim = model_dim 
        self.logger = logger
        self.neigh_agg = neigh_agg
        self.path_agg = path_agg

        self.aggregator = FeatureEncoder(self.feat_dim, self.model_dim, self.neigh_agg, self.path_agg, self.logger)     

    def forward_one_node(self, time_features, position_pos_features, position_neg_features, sign, weight, masks=None):
        batch, n_walk, len_walk = np.shape(sign)[0], np.shape(sign)[1], np.shape(sign)[2]

        combined_features_p = self.concat_features(position_pos_features, time_features, weight) 
        combined_features_n = self.concat_features(position_neg_features, time_features, weight)

        pos_and_neg = self.aggregator([combined_features_p, combined_features_n], sign, masks) 
       
        return pos_and_neg 
    
    def forward_one_node_di(self, time_features, position_pos_features, position_neg_features, sign, weight, direction, masks=None):
        batch, n_walk, len_walk = np.shape(sign)[0], np.shape(sign)[1], np.shape(sign)[2]

        combined_features_p = self.concat_features_di(position_pos_features, time_features, weight, direction) 
        combined_features_n = self.concat_features_di(position_neg_features, time_features, weight, direction)

        pos_and_neg = self.aggregator([combined_features_p, combined_features_n], sign, masks) 
       
        return pos_and_neg 


    def concat_features(self, time_features, position_features, weight):
        batch, n_walk, len_walk, _ = position_features.shape
        device = time_features.device
        weight_ = torch.Tensor(weight).unsqueeze(-1).to(device)
        if position_features is None:
            assert(self.pos_dim == 0)
            combined_features = torch.cat([time_features], dim=-1)
        else:
            combined_features = torch.cat([time_features, position_features, weight_], dim=-1) 
        combined_features = combined_features.to(device)
        assert(combined_features.size(-1) == self.feat_dim) 
        return combined_features

    def transform_input(self, input_array, mapping_dict):
        vectorized_mapping = np.vectorize(lambda x: np.array(mapping_dict.get(x)), signature='()->(n)')
        result = vectorized_mapping(input_array[:, :, :, -1].cpu())
        return np.stack(result)

    def concat_features_di(self, time_features, position_features, weight, direction):
        batch, n_walk, len_walk, _ = position_features.shape
        device = time_features.device
        weight_ = torch.Tensor(weight).unsqueeze(-1).to(device)
        direction_ = torch.Tensor(direction).unsqueeze(-1).to(device)

        mapping_dict = {-1: [1, 0], 0: [0, 0], 1: [0, 1], 2: [1, 1]}

        if position_features is None:
            assert(self.pos_dim == 0)
            combined_features = torch.cat([time_features], dim=-1)
        else:
            direction__ = self.transform_input(direction_, mapping_dict)
            direction__ = torch.Tensor(direction__).to(device)
            combined_features = torch.cat([time_features, position_features, weight_, direction__], dim=-1) 
        combined_features = combined_features.to(device)
        assert(combined_features.size(-1) == self.feat_dim)
        return combined_features

class SelfAttention(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SelfAttention, self).__init__()
        self.embed_size = input_size
        self.W_q = torch.nn.Linear(input_size, output_size, bias=True)
        self.W_k = torch.nn.Linear(input_size, output_size, bias=True)
        self.W_v = torch.nn.Linear(input_size, output_size, bias=True)
    
    def initialize(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.zeros_(self.W_k.bias)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_v.bias)

    def forward(self, x):

        batch_size, sequence_length, embed_size = x.size()

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32)) 

        attention_weights = F.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, v)
        return attended_values, attention_weights
    
class SelfAttention_path(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SelfAttention_path, self).__init__()
        self.embed_size = input_size
        self.W_q = torch.nn.Linear(input_size, output_size, bias=True)
        self.W_k = torch.nn.Linear(input_size, output_size, bias=True)
        self.W_v = torch.nn.Linear(input_size, output_size, bias=True)
    
    def initialize(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.zeros_(self.W_k.bias)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_v.bias)

    def forward(self, x):

        batch_size, num_walk, embed_size = x.size()

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32))


        attention_weights = F.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, v)
        return attended_values, attention_weights
    
class FeatureEncoder(nn.Module):
    def __init__(self, feature_dim, model_dim, neigh_agg, path_agg, logger):
        super(FeatureEncoder, self).__init__()
        self.logger = logger
        self.model_dim_one_direction = feature_dim//2
        self.feature_dim = feature_dim
        self.model_dim = model_dim
        
        self.set_node_embedding_act = torch.nn.Linear(self.feature_dim, self.model_dim)


        if neigh_agg == 'new_attn':
            self.attn_encoder = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=1, dropout=0.1, bias=True, kdim=self.feature_dim, vdim=self.feature_dim, batch_first=True)
        elif neigh_agg == 'attn':
            self.attn_encoder = SelfAttention(input_size = self.feature_dim, output_size = self.feature_dim)
            self.attn_encoder.initialize()
        elif neigh_agg == 'rnn':
            self.lstm_encoder = nn.LSTM(input_size=self.feature_dim, hidden_size=self.feature_dim, batch_first=True, bidirectional=False)
        
        self.attn_encoder_path = SelfAttention_path(input_size = model_dim, output_size = model_dim)
        self.attn_encoder_path.initialize()

        self.neigh_agg = neigh_agg
        self.path_agg = path_agg

    def forward(self, X, sign, mask=None):

        batch, n_walk, len_walk = np.shape(sign)[0], np.shape(sign)[1], np.shape(sign)[2]
       
        pos_mask = sign > 0
        neg_mask = sign < 0
        x_pos, x_neg = X 
        device = x_pos.device
        batch, n_walk, len_walk, feat_dim = x_pos.shape
        positive = torch.where(torch.tensor(pos_mask).unsqueeze(-1).to(device), x_pos, torch.zeros(feat_dim, requires_grad=True).to(device)) 
        positive += torch.where(torch.tensor(neg_mask).unsqueeze(-1).to(device), x_neg, torch.zeros(feat_dim, requires_grad=True).to(device)) 

        negative = torch.where(torch.tensor(pos_mask).unsqueeze(-1).to(device), x_neg, torch.zeros(feat_dim, requires_grad=True).to(device)) 
        negative += torch.where(torch.tensor(neg_mask).unsqueeze(-1).to(device), x_pos, torch.zeros(feat_dim, requires_grad=True).to(device)) 

        if self.neigh_agg == 'rnn':
            positive_embedding = self.lstm_encoder(positive.view(-1, len_walk, feat_dim))[0][:, -1, :].view(batch, n_walk, self.feature_dim) 
            negative_embedding = self.lstm_encoder(negative.view(-1, len_walk, feat_dim))[0][:, -1, :].view(batch, n_walk, self.feature_dim) 
        
        elif self.neigh_agg == 'attn':
            attention_value_p, attention_weight_p = self.attn_encoder(positive.view(-1, len_walk, feat_dim)) 
            attention_value_n, attention_weight_n = self.attn_encoder(negative.view(-1, len_walk, feat_dim)) 

            positive_embedding = torch.sum(attention_value_p, dim=1).view(batch, n_walk, self.feature_dim) 
            negative_embedding = torch.sum(attention_value_n, dim=1).view(batch, n_walk, self.feature_dim)
        
        elif self.neigh_agg == 'new_attn':
            attention_value_p, attention_weight_p = self.attn_encoder(positive.view(-1, len_walk, feat_dim), positive.view(-1, len_walk, feat_dim), positive.view(-1, len_walk, feat_dim)) 
            attention_value_n, attention_weight_n = self.attn_encoder(negative.view(-1, len_walk, feat_dim), negative.view(-1, len_walk, feat_dim), negative.view(-1, len_walk, feat_dim)) 

            positive_embedding = torch.sum(attention_value_p, dim=1).view(batch, n_walk, self.feature_dim)  
            negative_embedding = torch.sum(attention_value_n, dim=1).view(batch, n_walk, self.feature_dim)
        
        positive_embedding = self.set_node_embedding_act(positive_embedding)
        negative_embedding = self.set_node_embedding_act(negative_embedding)

        
        if self.path_agg == 'mean':
            pos = positive_embedding.mean(dim=1) 
            neg = negative_embedding.mean(dim=1)
        elif self.path_agg == 'attn':
            attention_value_pos, attention_weight_pos = self.attn_encoder_path(positive_embedding.view(batch, n_walk, self.model_dim)) 
            attention_value_neg, attention_weight_neg = self.attn_encoder_path(negative_embedding.view(batch, n_walk, self.model_dim)) 
            
            pos = torch.sum(attention_value_pos, dim=1).view(batch, self.model_dim) 
            neg = torch.sum(attention_value_neg, dim=1).view(batch, self.model_dim) 

        return [pos, neg]


class SetPooler(nn.Module):

    def __init__(self, n_features, out_features, dropout_p=0.1, walk_linear_out=False):
        super(SetPooler, self).__init__()
        self.mean_proj = nn.Linear(n_features, n_features)
        self.max_proj = nn.Linear(n_features, n_features)
        self.attn_weight_mat = nn.Parameter(torch.zeros((2, n_features, n_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.attn_weight_mat.data[0])
        nn.init.xavier_uniform_(self.attn_weight_mat.data[1])
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Sequential(nn.Linear(n_features, out_features), nn.ReLU(), self.dropout)
        self.walk_linear_out = walk_linear_out

    def forward(self, X, aggr='sum'):
        if self.walk_linear_out: 
            return self.out_proj(X)
        if aggr == 'sum':
            return self.out_proj(X.sum(dim=-2))
        else:
            assert(aggr == 'mean')
            return self.out_proj(X.mean(dim=-2))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_t = src.transpose(0, 1)
        src2 = self.self_attn(src_t, src_t, src_t, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0].transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_t = tgt.transpose(0, 1)
        tgt2 = self.self_attn(tgt_t, tgt_t, tgt_t, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


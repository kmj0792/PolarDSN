import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from eval import *
from utils import *
args, sys_argv = get_args()
import wandb
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
args, sys_argv = get_args()

def train_val(train_val_data, model, bs, epochs, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger):
    train_data, val_data = train_val_data
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data
    train_rand_sampler, val_rand_sampler = rand_samplers
    partial_ngh_finder, full_ngh_finder = ngh_finders
 
    model.update_ngh_finder(partial_ngh_finder)
    
    device = next(model.parameters()).device
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / bs)
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    for epoch in range(epochs):
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)  
        logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx] 
            ts_l_cut = train_ts_l[batch_idx] 
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx] 
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
            optimizer.zero_grad()
            model.train()
            

            pos_prob, neg_prob = model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)   
            pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
            neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)
            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.eval()
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))


        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val', model, val_rand_sampler, val_src_l,val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
       
        
        logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
        logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
        logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
        if epoch == 0:
            checkpoint_dir = '/'.join(model.get_checkpoint_path(0).split('/')[:-1])
            model.ngh_finder.save_ngh_stats(checkpoint_dir) 
            model.save_common_node_percentages(checkpoint_dir)

        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))


def train_val_for_multiclass(partial_adj_list, train_val_data, model, bs, epochs, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger):
    train_data, val_data = train_val_data
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l, train_weight_l = train_data 
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l, val_weight_l = val_data
    train_rand_sampler, val_rand_sampler = rand_samplers
    
    partial_ngh_finder, full_ngh_finder = ngh_finders
    model.update_ngh_finder(partial_ngh_finder) 
   
    device = next(model.parameters()).device
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / bs)
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    for epoch in range(epochs):
        precision, accuracy, recall, weighted_f1, micro_f1, macro_f1, m_loss = [], [], [], [], [], [], []
        # non shuffle
        # np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
        logger.info('start {} epoch'.format(epoch))
        
        for k in tqdm(range(num_batch), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx] 
            ts_l_cut = train_ts_l[batch_idx] 
            e_l_cut = train_e_idx_l[batch_idx]  
            label_l_cut = train_label_l[batch_idx] 
            weight_l_cut = train_weight_l[batch_idx] 

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
           
            size = len(src_l_cut)
            size_non = len(src_l_fake)
            
            
            optimizer.zero_grad()
            model.train()
            
            emb, zero_emb = model.contrast_for_multiclass(partial_adj_list, src_l_cut, dst_l_cut, src_l_fake, dst_l_fake, ts_l_cut, label_l_cut, e_l_cut, test=False)

            label_l_cut_ = np.where(label_l_cut<0, 0, 1)
            label_l_cut_tensor =  torch.Tensor(label_l_cut_).to(device)


            mask_p = label_l_cut_tensor==1
            mask_n = label_l_cut_tensor==0
            emb_p = emb[mask_p]
            emb_n = emb[mask_n]

            pos_label = torch.Tensor(np.full(len(emb_p), 1)).to(device)
            neg_label = torch.Tensor(np.full(len(emb_n), 0)).to(device)
            non_label = torch.Tensor(np.full(size_non, 2)).to(device)

            if len(emb_p) == 0:
                loss1=0
            else:
                loss1 = criterion(emb_p, pos_label.long()) 
            if len(emb_n) == 0: 
                loss2 = 0
            else:
                loss2 = criterion(emb_n, neg_label.long()) 

            loss3 = criterion(zero_emb, non_label.long()) 
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

  
            with torch.no_grad():
                emb_ = torch.softmax(emb, 1) 
                zero_emb_ = torch.softmax(zero_emb, 1) 
                pred_score = np.concatenate([emb_.cpu().detach().numpy(), zero_emb_.cpu().detach().numpy()])

                pred_label = pred_score.argmax(1) 
                true_label = np.concatenate([label_l_cut_, np.full(size_non, 2)])

                precision.append(precision_score(true_label, pred_label, average='macro', zero_division=1)) 
                accuracy.append(accuracy_score(true_label, pred_label)) 
                recall.append(recall_score(true_label, pred_label, average='macro', zero_division=0))
                weighted_f1.append(f1_score(true_label, pred_label, average='weighted'))
                micro_f1.append(f1_score(true_label, pred_label, average='micro'))
                macro_f1.append(f1_score(true_label, pred_label, average='macro'))
            
                m_loss.append(loss.item())
    
        val_precision, val_accuracy, val_recall, val_weighted_f1, val_micro_f1, val_macro_f1 = eval_one_epoch_for_multiclass(partial_adj_list, 'val', model, val_rand_sampler, val_src_l, val_dst_l, val_ts_l, val_label_l, val_weight_l, val_e_idx_l)

        logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train precision: {}, val precision: {}'.format(np.mean(precision), val_precision))
        logger.info('train accuracy: {}, val accuracy: {}'.format(np.mean(accuracy), val_accuracy))
        logger.info('train recall: {}, val recall: {}'.format(np.mean(recall), val_recall))
        logger.info('train weighted_f1: {}, val weighted_f1: {}'.format(np.mean(weighted_f1), val_weighted_f1))
        logger.info('train micro_f1: {}, val micro_f1: {}'.format(np.mean(micro_f1), val_micro_f1))
        logger.info('train macro_f1: {}, val macro_f1: {}'.format(np.mean(macro_f1), val_macro_f1))
        total_val = val_accuracy +  val_weighted_f1 + val_micro_f1 + val_macro_f1
        
       
        if epoch == 0:
    
            checkpoint_dir = '/'.join(model.get_checkpoint_path(0).split('/')[:-1])
            model.ngh_finder.save_ngh_stats(checkpoint_dir)  # for data analysis
            model.save_common_node_percentages(checkpoint_dir)


        if early_stopper.early_stop_check(total_val): 
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))


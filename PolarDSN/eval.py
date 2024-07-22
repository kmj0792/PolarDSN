import math
import torch
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label, val_e_idx_l=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut, test=True)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(sklearn.metrics.average_precision_score(true_label, pred_score))
            val_auc.append(sklearn.metrics.roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)

def eval_one_epoch_for_multiclass(adj_list, hint, tgan, sampler, src, dst, ts, label, weight, val_e_idx_l=None): #label == sign
    val_precision, val_accuracy, val_recall, val_cm, val_weighted_f1, val_micro_f1, val_macro_f1 = [], [], [], [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 32
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            label_l_cut = label[s_idx:e_idx] # sign: +1 or -1
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None
            weight_l_cut = weight[s_idx:e_idx]
                 
            size = len(src_l_cut)
            

            src_l_fake, dst_l_fake = sampler.sample(size) 
            size_non = len(src_l_fake)
        
            emb, zero_emb = tgan.contrast_for_multiclass(adj_list, src_l_cut, dst_l_cut, src_l_fake, dst_l_fake, ts_l_cut, label_l_cut, e_l_cut, test=True)
            emb_ = torch.softmax(emb, 1)
            zero_emb_ = torch.softmax(zero_emb, 1)

            label_l_cut_ = np.where(label_l_cut<0, 0, 1)
            pred_score = np.concatenate([emb_.cpu().detach().numpy(), zero_emb_.cpu().detach().numpy()])

            pred_label = pred_score.argmax(1) 

            true_label = np.concatenate([label_l_cut_, np.full(size_non, 2)])

            val_precision.append(precision_score(true_label, pred_label, average='macro',zero_division=1))
            val_accuracy.append(accuracy_score(true_label, pred_label))
            val_recall.append(recall_score(true_label, pred_label, average='macro', zero_division=0))
            val_weighted_f1.append(f1_score(true_label, pred_label, average='weighted'))
            val_micro_f1.append(f1_score(true_label, pred_label, average='micro'))
            val_macro_f1.append(f1_score(true_label, pred_label, average='macro'))

    return np.mean(val_precision), np.mean(val_accuracy), np.mean(val_recall), np.mean(val_weighted_f1), np.mean(val_micro_f1), np.mean(val_macro_f1)
from numba import jit
import numpy as np
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


@jit(nopython=True)
def nodets2key(batch: int, node: int, ts: float, sign: int):
    ts = 0 
    key = '-'.join([str(batch), str(node), float2str(ts),str(sign)])
    return key


@jit(nopython=True)
def float2str(ts):
    return str(int(round(ts)))


def make_batched_keys(node_record, t_record, s_record):
    batch = node_record.shape[0] 
    support = node_record.shape[1]
    batched_keys_pos, batched_keys_neg = make_batched_keys_l(node_record, t_record, s_record, batch, support)
    batched_keys_pos = np.array(batched_keys_pos).reshape((batch, support))
    batched_keys_neg = np.array(batched_keys_neg).reshape((batch, support))
   
    return batched_keys_pos, batched_keys_neg


@jit(nopython=True)
def make_batched_keys_l(node_record, t_record, s_record, batch, support):
    batch_matrix = np.arange(batch).repeat(support).reshape((-1, support))

    batched_keys_pos = []
    batched_keys_neg = []
    for i in range(batch):
        for j in range(support):
            b = batch_matrix[i, j]
            n = node_record[i, j]
            t = t_record[i, j]
            s = s_record[i, j]
            batched_keys_pos.append(nodets2key(b, n, t, 1))
            batched_keys_neg.append(nodets2key(b, n, t, -1))
    return batched_keys_pos, batched_keys_neg
import numpy as np
from numpy.core.fromnumeric import shape
import scipy.io as scio
import torch
from typing import Union, Tuple, List
import path 

def get_balanced_sub_idx(y, mask_train):
    from collections import Counter
    ytr = y[mask_train]
    counter = Counter(ytr.tolist())
    p = min(counter.values())

    idx = []
    C = y.max().item() + 1

    for c in range(C):
        idx += torch.nonzero(ytr == c).squeeze().tolist()[:p]

    N = y.shape[0]
    mask_train = torch.LongTensor(idx) 
    mask_val = torch.LongTensor(list(set(range(N)) - set(idx) ))

    print(f'we use only Ntr = {len(idx)} ({p} per class x {C}) instead of {ytr.shape[0]}')
    return mask_train, mask_val

def get_split(Y, p=0.2):
    Y = Y.tolist()
    N, nclass = len(Y),  len(set(Y))
    D = [[] for _ in range(nclass)]
    for i, y in enumerate(Y):
        D[y].append(i)
    k = int(N * p / nclass)
    train_idx = torch.cat([torch.LongTensor(random.sample(idxs, k)) for idxs in D])
    test_idx = torch.LongTensor(list(set(range(N)) - set(train_idx.tolist())))
    assert train_idx.shape[0] > 0
    assert test_idx.shape[0] > 0
    print(f'label rate:{train_idx.shape[0] / N:.4f}')
    return train_idx, test_idx

def load_datav2(data_dir, selected_mod=(0, 1)):
    data = scio.loadmat(data_dir)
    y = torch.LongTensor(data['Y'].squeeze())
    if y.min() == 1: y = y - 1

    Xs = [torch.FloatTensor(data['X'][imod].item()) for imod in selected_mod]
    idx = torch.LongTensor(data['indices'].item().squeeze())
    idx_train = (idx == 1)
    idx_test = (idx == 0)

    X_train = [X[idx_train] for X in Xs]
    X_test = [X[idx_test] for X in Xs]
    y_train = y[idx_train]
    y_test = y[idx_test]
    return X_train, X_test, y_train, y_test

def generate_Hv2(X, k_nearest, Nnew, cached_dir=None, add_self_loop=True):
    fH = cached_dir
    if fH is not None and fH.exists():
        H = torch.load(fH)
    else:
        H = neighbor_distance(X, k_nearest)
        if fH is not None:
            torch.save(H, fH)
    if add_self_loop:
        N, M = H[0].max()+1, H[1].max()+1 
        # Hs = H.tolist()
        self_loops = torch.LongTensor(range(0, Nnew))
        self_loops = torch.stack((self_loops, self_loops+M-0))
        H = torch.hstack((H, self_loops))
    return create_sparse_H(H)

def generate_H(X, k_nearest, cached_dir=None, add_self_loop=True):
    fH = cached_dir
    if fH is not None and fH.exists():
        H = torch.load(fH)
    else:
        H = neighbor_distance(X, k_nearest)
        if fH is not None:
            torch.save(H, fH)
    if add_self_loop:
        N, M = H[0].max()+1, H[1].max()+1 
        self_loops = torch.LongTensor(range(0, N))
        self_loops = torch.stack((self_loops, self_loops+M-0))
        H = torch.hstack((H, self_loops))
    return H 
    return create_sparse_H(H)

def create_sparse_H(H: torch.Tensor):
    # H: 2 x nnz
    # return H: N x N 

    import numpy as np 
    import scipy.sparse as sp 
    import torch
    import torch_sparse
    # M = H[1].max() + 1
    H = sp.csr_matrix((torch.ones_like(H[0]), H.tolist())) 
    N, M = H.shape
    Dv = sp.spdiags(np.power(H.sum(1).reshape(-1), -0.5), 0, N, N)
    De = sp.spdiags(np.power(H.sum(0).reshape(-1), -1.), 0, M, M)
    Hv = Dv * H * De * H.transpose() * Dv # V x V, for HGNN

    (row, col), value = torch_sparse.from_scipy(Hv)
    # Hv = Hv.tocoo()
    # row, col, value = Hv.row, Hv.col, Hv.data
    H = torch.sparse.FloatTensor(torch.stack((row, col)), value, (N, N) ).float() # V x V
    return H 


def load_data(data_dir, selected_mod=(0, 1)):
    data = scio.loadmat(data_dir)
    y = torch.LongTensor(data['Y'].squeeze())
    if y.min() == 1: y = y - 1

    Xs = [torch.FloatTensor(data['X'][imod].item()) for imod in selected_mod]
    idx = torch.LongTensor(data['indices'].item().squeeze())
    idx_train = (idx == 1)
    idx_test = (idx == 0)

    # X_train = [X[idx_train] for X in Xs]
    # X_test = [X[idx_test] for X in Xs]
    # y_train = y[idx_train]
    # y_test = y[idx_test]
    # return X_train, X_test, y_train, y_test
    return Xs, y, idx_train, idx_test

def load_ft(data_dir, feature_name='GVCNN'):
    data = scio.loadmat(data_dir)
    lbls = data['Y'].astype(np.long)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        fts = data['X'][0].item().astype(np.float32)
    elif feature_name == 'GVCNN':
        fts = data['X'][1].item().astype(np.float32)
    else:
        print(f'wrong feature name{feature_name}!')
        raise IOError

    idx_train = (idx == 1)
    idx_test = (idx == 0)
    return torch.tensor(fts), torch.LongTensor(lbls).squeeze(), \
           torch.tensor(idx_train).squeeze().bool(), \
           torch.tensor(idx_test).squeeze().bool()

def pairwise_euclidean_distance(x: torch.Tensor):
    """
    generate N x N node distance matrix
    :param x: a tensor of size N x C (N nodes with C feature dimension)
    :return: a tensor of N x N (distance matrix for each node pair)
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 2
    x = x.float()

    x_transpose = torch.transpose(x, dim0=0, dim1=1)
    x_inner = torch.matmul(x, x_transpose)
    x_inner = -2 * x_inner
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    x_square_transpose = torch.transpose(x_square, dim0=0, dim1=1)
    dis = x_square + x_inner + x_square_transpose
    return dis

def neighbor_distance(x: torch.Tensor, k_nearest, dis_metric=pairwise_euclidean_distance):
    """
    construct hyperedge for each node in x matrix. Each hyperedge contains a node and its k-1 nearest neighbors.
    :param x: N x C matrix. N denotes node number, and C is the feature dimension.
    :param k_nearest:
    :return:
    """

    assert len(x.shape) == 2, 'should be a tensor with dimension (N x C)'

    # N x C
    node_num = x.size(0)
    dis_matrix = dis_metric(x)
    _, nn_idx = torch.topk(dis_matrix, k_nearest, dim=1, largest=False)
    hyedge_idx = torch.arange(node_num).to(x.device).unsqueeze(0).repeat(k_nearest, 1).transpose(1, 0).reshape(-1)
    H = torch.stack([nn_idx.reshape(-1), hyedge_idx])
    return H

def accuracy(Z, Y):
    return 100 * Z.argmax(1).eq(Y).float().mean().item()



def hyedge_concat(Hs: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], same_node=True):
    node_num = 0
    hyedge_num = 0
    Hs_new = []
    for H in Hs:
        _H = H.clone()
        if not same_node:
            _H[0, :] += node_num
        _H[1, :] += hyedge_num

        Hs_new.append(_H)

        hyedge_num += count_hyedge(H)
        node_num += count_node(H)
    Hs_new = torch.cat(Hs_new, dim=1)
    return contiguous_hyedge_idx(Hs_new)

def count_node(H, node_num=None):
    return H[0].max().item() + 1 if node_num is None else node_num


def count_hyedge(H, hyedge_num=None):
    return H[1].max().item() + 1 if hyedge_num is None else hyedge_num

def contiguous_hyedge_idx(H):
    node_idx, hyedge_idx = H
    unorder_pairs = [(hyedge_id, sequence_id) for sequence_id, hyedge_id in enumerate(hyedge_idx.numpy().tolist())]
    unorder_pairs.sort(key=lambda x: x[0])
    new_hyedge_id = -1
    pre_hyedge_id = None
    new_hyedge_idx = list()
    sequence_idx = list()
    for (hyedge_id, sequence_id) in unorder_pairs:
        if hyedge_id != pre_hyedge_id:
            new_hyedge_id += 1
            pre_hyedge_id = hyedge_id
        new_hyedge_idx.append(new_hyedge_id)
        sequence_idx.append(sequence_id)
    hyedge_idx[sequence_idx] = torch.LongTensor(new_hyedge_idx)
    return torch.stack([node_idx, hyedge_idx])
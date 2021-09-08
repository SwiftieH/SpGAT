from sklearn.preprocessing import normalize
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.special as ss
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import warnings
warnings.filterwarnings("ignore")
from weighting_func import laplacian,fourier,weight_wavelet,weight_wavelet_inverse


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str,alldata = True):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    if(alldata == True):
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally,ty))
        num = labels.shape[0]
        idx_train = range(num/5*3)
        idx_val = range(num/5*3, num/5*4)
        idx_test = range(num/5*4, num)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return labels,adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # print rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv,0)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt,0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return adj_normalized
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def wavelet_basis(dataset,adj,s,laplacian_normalize,sparse_ness,threshold,weight_normalize):

    L = laplacian(adj,normalized=laplacian_normalize)
    lamb, U = fourier(dataset,L)
    #import pdb; pdb.set_trace()
    Weight = weight_wavelet(s,lamb,U)
    inverse_Weight = weight_wavelet_inverse(s,lamb,U)
    del U,lamb

    if (sparse_ness):
        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0

    if (weight_normalize == True):
        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)
    t_k = [inverse_Weight,Weight]
    return(t_k)

def wavelet_basis_appro(dataset,adj,s,laplacian_normalize,sparse_ness,threshold,weight_normalize):

    L = laplacian(adj,normalized=laplacian_normalize)
    L = L - sp.eye(adj.shape[0])
    L = L.todense()
    # quick version for s = 2
    #Weight = 16.844 * sp.eye(adj.shape[0]) + 23.507 * L
    #inverse_Weight = 0.309 * sp.eye(adj.shape[0]) - 0.431 * L
    Weight = np.exp(s) * ss.iv(0,s) * np.eye(adj.shape[0]) + 2 * np.exp(s) * ss.iv(1,s) * L 
    inverse_Weight = np.exp(-s) * ss.iv(0,-s) * np.eye(adj.shape[0]) + 2 * np.exp(-s) * ss.iv(1,-s) * L 

    if (sparse_ness):
        Weight[Weight < threshold] = 0.0 
        inverse_Weight[inverse_Weight < threshold] = 0.0 

    if (weight_normalize == True):
        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)
    t_k = [inverse_Weight,Weight]
    return(t_k)

'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import args

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # # load the data: x, tx, allx, graph
    # names = ['x', 'tx', 'allx', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))
    # x, tx, allx, graph = tuple(objects)
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    # test_idx_range = np.sort(test_idx_reorder)
    #
    # if dataset == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #
    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #
    # edges = np.load('C:/Users/Dingge/Downloads/vgae_pytorch-master/data/Edges.npy')

    # return adj, features, edges

    # path = './data/{}/'.format(dataset)
    #
    # Features = sp.load_npz(path + 'Features.npz')
    # Features = Features.toarray()
    # Features = sp.csr_matrix(Features)
    #
    # Labels = sp.load_npz(path + 'Labels.npz')
    # Labels = Labels.toarray()
    # Labels = Labels.reshape(Labels.shape[1], 1)
    #
    # Adjacency = sp.load_npz(path + 'Adjacency.npz')
    # Adjacency = Adjacency.toarray()
    #
    # Edges = np.load(path + 'Edges.npy')

    if dataset == 'eveques':
        adjacency = pd.read_csv('C:/Users/Dingge/Downloads/deepLsm/data/eveques_new/ResoEvequesClean2021-A.csv', header=0,
                          sep=';').to_numpy()  # load simu data
        features = pd.read_csv('C:/Users/Dingge/Downloads/deepLsm/data/eveques_new/ResoEvequesClean2021-X.csv',
                               header=0, sep=';').to_numpy()
        for i in range(features.shape[0]):
            if np.isnan(features[i][0]) == True and np.isnan(features[i][1]) == False:
                features[i][0] = features[i][1]

            elif np.isnan(features[i][0]) == False and np.isnan(features[i][1]) == True:
                features[i][1] = features[i][0]

            elif np.isnan(features[i][0]) == True and np.isnan(features[i][1]) == True:
                features[i][0] = features[i][1] = 0

        for i in range(features.shape[0]):
            if features[i][0] == 0 and features[i][1] == 0:
                features[i][0] = np.mean(features[:,0])
                features[i][1] = np.mean(features[:,1])

        min_f = np.min(features[:,0:2])
        max_f = np.max(features[:,0:2])
        features[:,0:2] = (features[:,0:2] - min_f)/(max_f - min_f)

        edges_flat = pd.read_csv('C:/Users/Dingge/Downloads/deepLsm/data/eveques_new/ResoEvequesClean2021-Yflat.csv',
                                 header=0, sep=';').to_numpy()
        edges = np.zeros((args.num_points, args.num_points, args.nb_of_edges))
        pos = np.where(edges_flat != 0)
        for j in range(len(pos[0])):
            edges[pos[0][j], pos[1][j], (edges_flat[pos[0][j],pos[1][j]]-1)] = 1

    return features, adjacency, edges  # Labels,



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

    if dataset == 'eveques':
        adjacency = pd.read_csv('data/eveques_new/ResoEvequesClean2021-A.csv',
                                header=0, sep=';').to_numpy()  # load simu data

        if args.use_nodes == True:
            features = pd.read_csv('data/eveques_new/ResoEvequesClean2021-X.csv',
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
                    features[i][0] = np.mean(features[:, 0])
                    features[i][1] = np.mean(features[:, 1])

            min_f = np.min(features[:, 0:2])
            max_f = np.max(features[:, 0:2])
            features[:, 0:2] = (features[:, 0:2] - min_f) / (max_f - min_f)
        else:
            features = np.zeros((adjacency.shape[0], args.input_dim))
            np.fill_diagonal(features, 1)

        # edges_flat = pd.read_csv('data/eveques_new/ResoEvequesClean2021-Yflat.csv',
        #                          header=0, sep=';').to_numpy()
        edges_1 = pd.read_csv('data/eveques_new/ResoEvequesClean2021-Ydates.csv',
                                 header=0, sep=';').to_numpy()
        edges_2 = pd.read_csv('data/eveques_new/ResoEvequesClean2021-Yfonctions.csv',
                              header=0, sep=';').to_numpy()
        edges_3 = pd.read_csv('data/eveques_new/ResoEvequesClean2021-Yregions.csv',
                              header=0, sep=';').to_numpy()

        edges = np.zeros((args.num_points, args.num_points, args.nb_of_edges))
        # pos = np.where(edges_flat != 0)
        # for j in range(len(pos[0])):
        #     edges[pos[0][j], pos[1][j], (edges_flat[pos[0][j],pos[1][j]]-1)] = 1

        # pos_edges_1 = np.where(edges_1 < 0, 0, edges_1)  # keep only positive numbers
        # new_edges_1 = pos_edges_1 / np.max(pos_edges_1)  # between 0 and 1
        edges[:, :, 0] = edges_1

        # new_edges_2 = np.where(edges_2 == -1, 0, edges_2)
        edges[:, :, 1] = edges_2

        # new_edges_3 = np.where(edges_3 == -1, 0, edges_3)
        edges[:, :, 2] = edges_3

    elif dataset == 'cora':
        path = './data/{}/'.format(dataset)

        adjacency = sp.load_npz(path + 'Adjacency.npz')
        adjacency = adjacency.toarray()

        if args.use_nodes == True:
            features = sp.load_npz(path + 'Features.npz')
            features = features.toarray()
            # Features = sp.csr_matrix(Features)
        else:
            features = np.zeros((adjacency.shape[0], args.input_dim))
            np.fill_diagonal(features, 1)

        labels = sp.load_npz(path + 'Labels.npz')
        labels = labels.toarray()
        labels = labels.reshape(labels.shape[1], 1)

        edges = np.load(path + 'Edges.npy')
        edges_1 = np.where(edges == 0, -1, edges)

    return features, adjacency, edges  # Labels,



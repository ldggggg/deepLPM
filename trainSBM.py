import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt

from input_data import load_data
from preprocessing import *
import args
import model
from testB import *
from testB2 import *
from testC import *
import math
import pickle

############
import numpy as np
from scipy.spatial.distance import pdist, squareform
import args


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

def create_simu(N,K,D):
# N = args.num_points
# K = args.num_clusters
# D = args.hidden2_dim

    delta = 0.2
    mu1 = [0, 0]
    mu2 = [delta * 1.5, delta * 1.5]
    mu3 = [-1.5 * delta, delta * 1.5]
    z_mu = np.concatenate((mu1,mu2,mu3), axis=0)
    # mu4 = [3, 0]
    # mu5 = [-3, 0]
    sigma1 = [[0.1, 0],[0, 0.1]]
    sigma2 = [[0.3, 0],[0, 0.3]]
    sigma3 = [[0.1, 0],[0, 0.1]]
    z_log_sigma = np.concatenate((sigma1,sigma2,sigma3), axis=0)
    # sigma4 = [[0.1, 0],[0, 0.1]]
    # sigma5 = [[0.2, 0],[0, 0.2]]
    x1 = np.random.multivariate_normal(mu1, sigma1, N//K)
    x2 = np.random.multivariate_normal(mu2, sigma2, N//K)
    x3 = np.random.multivariate_normal(mu3, sigma3, N - 2*(N//K))
    # x4 = np.random.multivariate_normal(mu4, sigma4, 120)
    # x5 = np.random.multivariate_normal(mu5, sigma5, 120)

    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots(1,figsize=(15,10))
    # ax.scatter(x1[:,0], x1[:,1], color = 'r')
    # ax.scatter(x2[:,0], x2[:,1], color = 'b')
    # ax.scatter(x3[:,0], x3[:,1], color = 'g')
    # # ax.scatter(x4[:,0], x4[:,1], color = 'y')
    # # ax.scatter(x5[:,0], x5[:,1], color = 'purple')
    # ax.set_title("This is the origin embeddings!")
    # plt.show()

    X = np.concatenate((x1,x2,x3), axis=0)
    # np.savetxt('emb_3clusters.txt', X)
    # np.savetxt('mu_3clusters.txt', z_mu)
    # np.savetxt('cov_3clusters.txt', z_log_sigma)
    Label1 = np.repeat(0, N//K)
    Label2 = np.repeat(1, N//K)
    Label3 = np.repeat(2, N- 2*(N//K))
    # Label4 = np.repeat(3, 120)
    # Label5 = np.repeat(4, 120)
    Label = np.concatenate((Label1,Label2,Label3), axis=0)


    dst = pdist(X, 'euclidean')
    dst = squareform(dst)

    alpha = 0.2
    from scipy.special import expit
    from scipy.stats import bernoulli
    A = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            prob = expit(alpha - dst[i,j])
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    # np.savetxt('adj_simu_3clusters.txt', A)
    # np.savetxt('label_simu_3clusters.txt', Label)

    return A, Label, X

#########################################################

# Load data
if args.dataset == 'simuC':
    # simulated data
    adj, labels = create_simuC(args.num_points, args.num_clusters)  # , oX    args.num_points, args.num_clusters, args.hidden2_dim
    # load to test edge features
    # adj = np.loadtxt('/home/dliang/deepLsm/adj_simu_3clusters.txt')  # load simu data
    # labels = np.loadtxt('/home/dliang/deepLsm/label_simu_3clusters.txt')  # load simu labels
    # take a look
    # plt.scatter(oX[:,0], oX[:,1])
    # adj = np.loadtxt('C:/Users/Dingge/Downloads/deepLsm/adj_simu_3clusters.txt')  # load simu data
    # adj = sp.csr_matrix(adj)  # no sparse for SBM
    features = np.zeros((adj.shape[0], args.input_dim))
    np.fill_diagonal(features, 1)
    features = sp.csr_matrix(features)
    # with open('edges_simu_3clusters', 'rb') as fp:
    #     edges = pickle.load(fp)

    # load real labels
    labelC = []
    for idx in range(len(labels)):
        if labels[idx] == 0:
            labelC.append('red')
        elif labels[idx] == 1:
            labelC.append('green')
        # elif labels[idx] == 2:
        #     labelC.append('yellow')
        # elif labels[idx] == 3:
        #     labelC.append('purple')
        else:
            labelC.append('blue')

else:
    # adj, features, edges = load_data(args.dataset)
    features, labels, adj, edges = load_data(args.dataset)
    labels = labels.squeeze(1).tolist()
    adj = sp.coo_matrix(adj)
    edges = torch.tensor(edges)

    labelC = []
    for idx in range(len(labels)):
        if labels[idx] == 0:
            labelC.append('red')
        elif labels[idx] == 1:
            labelC.append('green')
        elif labels[idx] == 2:
            labelC.append('yellow')
        elif labels[idx] == 3:
            labelC.append('purple')
        elif labels[idx] == 4:
            labelC.append('grey')
        elif labels[idx] == 5:
            labelC.append('pink')
        else:
            labelC.append('blue')

from sparsebm import SBM
number_of_clusters = args.num_clusters
# A number of classes must be specify. Otherwise see model selection.
model1 = SBM(number_of_clusters, n_init=100)
model2 = SBM(number_of_clusters, n_init=1)
model3 = SBM(number_of_clusters, n_init=10)
model4 = SBM(number_of_clusters, n_init=1000)

model1.fit(adj, symmetric=True)
# print("Labels:", model.labels)
print("ARI_SBM_INI_100:", adjusted_rand_score(labels, model1.labels))
# model2.fit(adj.todense(), symmetric=True)
# # print("Labels:", model.labels)
# print("ARI_SBM_INI_1:", adjusted_rand_score(labels, model2.labels))
# model3.fit(adj.todense(), symmetric=True)
# # print("Labels:", model.labels)
# print("ARI_SBM_INI_10:", adjusted_rand_score(labels, model3.labels))
# model4.fit(adj.todense(), symmetric=True)
# # print("Labels:", model.labels)
# print("ARI_SBM_INI_1000:", adjusted_rand_score(labels, model4.labels))


from SBM_package.src import SBM

elbo, tau, tau_init, count, time_list = SBM.sbm(adj, args.num_clusters, algo='vbem', type_init='kmeans')
c = np.argmax(tau, axis=1)
print("ARI_SBM_init_kmeans:", adjusted_rand_score(labels, c))

# elbo1, tau1, _, count1, time_list1 = SBM.sbm(A=adj,
#                                      Q=3,
#                                      max_iter=200,
#                                      tau_init=None,
#                                      type_init='random',
#                                      tol=1e-6,
#                                      algo='vbem')
#
# c1 = np.argmax(tau, axis=1)
# print("ARI_SBM_init_random:", adjusted_rand_score(labels, c1))
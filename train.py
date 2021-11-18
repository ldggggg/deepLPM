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
import model
# from testB import *
# from testB2 import *
# from testC import *
import math
import pickle

############
import numpy as np
from scipy.spatial.distance import pdist, squareform
import args

# def create_simu(N,K):
#     delta = 0.5
#     mu1 = [0, 0]
#     mu2 = [delta * 1.5, delta * 1.5]
#     mu3 = [-1.5 * delta, delta * 1.5]
#     z_mu = np.concatenate((mu1,mu2,mu3), axis=0)
#     # mu4 = [3, 0]
#     # mu5 = [-3, 0]
#     sigma1 = [[0.1, 0],[0, 0.1]]
#     sigma2 = [[0.3, 0],[0, 0.3]]
#     sigma3 = [[0.1, 0],[0, 0.1]]
#     z_log_sigma = np.concatenate((sigma1,sigma2,sigma3), axis=0)
#     # sigma4 = [[0.1, 0],[0, 0.1]]
#     # sigma5 = [[0.2, 0],[0, 0.2]]
#     x1 = np.random.multivariate_normal(mu1, sigma1, N//K)
#     x2 = np.random.multivariate_normal(mu2, sigma2, N//K)
#     x3 = np.random.multivariate_normal(mu3, sigma3, N - 2*(N//K))
#     # x4 = np.random.multivariate_normal(mu4, sigma4, 120)
#     # x5 = np.random.multivariate_normal(mu5, sigma5, 120)
#
#     # import matplotlib.pyplot as plt
#     # f, ax = plt.subplots(1,figsize=(15,10))
#     # ax.scatter(x1[:,0], x1[:,1], color = 'r')
#     # ax.scatter(x2[:,0], x2[:,1], color = 'b')
#     # ax.scatter(x3[:,0], x3[:,1], color = 'g')
#     # # ax.scatter(x4[:,0], x4[:,1], color = 'y')
#     # # ax.scatter(x5[:,0], x5[:,1], color = 'purple')
#     # ax.set_title("This is the origin embeddings!")
#     # plt.show()
#
#     X = np.concatenate((x1,x2,x3), axis=0)
#     # np.savetxt('emb_3clusters.txt', X)
#     # np.savetxt('mu_3clusters.txt', z_mu)
#     # np.savetxt('cov_3clusters.txt', z_log_sigma)
#     Label1 = np.repeat(0, N//K)
#     Label2 = np.repeat(1, N//K)
#     Label3 = np.repeat(2, N- 2*(N//K))
#     # Label4 = np.repeat(3, 120)
#     # Label5 = np.repeat(4, 120)
#     Label = np.concatenate((Label1,Label2,Label3), axis=0)
#
#
#     dst = pdist(X, 'euclidean')
#     dst = squareform(dst)
#
#     alpha = 0.2
#     from scipy.special import expit
#     from scipy.stats import bernoulli
#     A = np.zeros((N, N))
#     for i in range(N-1):
#         for j in range(i+1, N):
#             prob = expit(alpha - dst[i,j])
#             A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)
#
#     # np.savetxt('adj_simu_3clusters.txt', A)
#     # np.savetxt('label_simu_3clusters.txt', Label)
#
#     return A, Label, X


#########################################################

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

print('Number of clusters:.................'+str(args.num_clusters))

# Load data
if args.dataset == 'simu':
    # simulated data
    # adj, labels = create_simuB(args.num_points, args.num_clusters, 1)  # , oX    args.num_points, args.num_clusters, args.hidden2_dim
    # load to test edge features
    # adj = np.loadtxt('/home/dliang/deepLsm/adj_simuA_3clusters.txt')  # load simu data
    # labels = np.loadtxt('/home/dliang/deepLsm/label_simuA_3clusters.txt')  # load simu labels
    adj = np.loadtxt('adj_simuB_3clusters.txt')  # load simu data
    labels = np.loadtxt('label_simuB_3clusters.txt')  # load simu labels
    adj = sp.csr_matrix(adj)
    features = np.zeros((adj.shape[0], args.input_dim))
    np.fill_diagonal(features, 1)
    features = sp.csr_matrix(features)

    # with open('edges_simu_3clusters_2texts_delta0.4', 'rb') as fp:C
    #     edges = pickle.load(fp)
    # edges = np.sum(edges, axis=1)  # N * V

    # features = sp.csr_matrix(edges)  # To test node features

    # load real labels
    labelC = []
    for idx in range(len(labels)):
        if labels[idx] == 0:
            labelC.append('#7294d4')
        elif labels[idx] == 1:
            labelC.append('#fdc765')
        # elif labels[idx] == 2:
        #     labelC.append('yellow')
        # elif labels[idx] == 3:
        #     labelC.append('purple')
        else:
            labelC.append('#869f82')
    # take a look
    # plt.scatter(oX[:, 0], oX[:, 1], color=labelC)

else:
    # adj, features, edges = load_data(args.dataset)
    features, labels, adj, edges = load_data(args.dataset)
    labels = labels.squeeze(1).tolist()
    adj = sp.coo_matrix(adj)
    # edges = torch.tensor(edges)

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

# from sparsebm import SBM
# number_of_clusters = args.num_clusters
# # A number of classes must be specify. Otherwise see model selection.
# model1 = SBM(number_of_clusters, n_init=100)
# model1.fit(adj.todense(), symmetric=True)
# # print("Labels:", model.labels)
# print("ARI_SBM:", adjusted_rand_score(labels, model1.labels))

from SBM_package.src import SBM
elbo, tau, tau_init, count, time_list = SBM.sbm(adj.todense(), args.num_clusters, algo='vbem', type_init='kmeans')
c = np.argmax(tau, axis=1)
print("ARI_SBM_init_kmeans:", adjusted_rand_score(labels, c))

# Some preprocessing
adj_norm = preprocess_graph(adj)  # used to train the encoder
features = sparse_to_tuple(features.tocoo())
# adj = sparse_to_tuple(adj)  # original adj
adj_label = adj + sp.eye(adj.shape[0])  # used to calculate the loss
adj_label = sparse_to_tuple(adj_label)


# Create Model
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].astype(float).T),
                            torch.FloatTensor(adj_norm[1].astype(float)),
                            torch.Size(adj_norm[2]))
# adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0].astype(float).T),
#                             torch.FloatTensor(adj[1].astype(float)),
#                             torch.Size(adj[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].astype(float).T),
                            torch.FloatTensor(adj_label[1]),
                            torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].astype(float).T),
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2]))


# init model and optimizer
model = getattr(model, args.model)(adj_norm)  # model_texts

model.pretrain(features, adj_label, labels, labelC)  # pretraining ari=0.4813

optimizer = Adam(model.parameters(), lr=args.learning_rate)  # , weight_decay=0.01
lr_s = StepLR(optimizer, step_size=10, gamma=0.9)


# train model
store_loss = torch.zeros(args.num_epoch)
store_loss1 = torch.zeros(args.num_epoch)
store_loss2 = torch.zeros(args.num_epoch)
store_loss3 = torch.zeros(args.num_epoch)
store_loss4 = torch.zeros(args.num_epoch)

store_A_pred = []
store_ari = torch.zeros(args.num_epoch)


def ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, P):
    # Loss1 = F.binary_cross_entropy(A_pred.view(-1), adj.to_dense().view(-1))
    # Loss1 = Loss1 * args.num_points
    # bce_loss = torch.nn.BCELoss(size_average=False) # calculate the sum
    # Loss1 = bce_loss(A_pred.view(-1), adj_label.to_dense().view(-1))

    OO = adj_label.to_dense()*(torch.log((A_pred/(1. - A_pred)) + 1e-16)) + torch.log((1. - A_pred) + 1e-16)
    ind = np.diag_indices(OO.shape[0])
    OO[ind[0], ind[1]] = torch.zeros(OO.shape[0])
    Loss1 = -torch.sum(OO)
    
    KL = torch.zeros((args.num_points, args.num_clusters))  # N * K
    for k in range(args.num_clusters):
        for i in range(args.num_points):
            KL[i, k] = 0.5 * (P*(log_cov_k[k] - log_cov_phi[i]) - P
                              + P*torch.exp(log_cov_phi)[i] / torch.exp(log_cov_k[k])
                              + torch.norm(mu_k[k] - mu_phi[i]) ** 2 / torch.exp(log_cov_k[k]))

    # for k in range(args.num_clusters):
    #     P = args.hidden2_dim
    #     KL[:, k] = 0.5 * torch.sum(P * (log_cov_k[k].unsqueeze(0) - log_cov_phi) - P
    #                                + torch.exp(log_cov_phi) / torch.exp(log_cov_k[k].unsqueeze(0))
    #                                + torch.sum((mu_k[k].unsqueeze(0) - mu_phi) ** 2, axis=1).unsqueeze(
    #         1) / torch.exp(log_cov_k[k].unsqueeze(0)), axis=1)

    Loss2 = torch.sum(gamma * KL)

    Loss3 = torch.sum(gamma * (torch.log(pi_k.unsqueeze(0)) - torch.log(gamma)))

    Loss = Loss1 + Loss2 - Loss3

    return Loss, Loss1, Loss2, -Loss3

# mu_phi, log_cov_phi, z = model.encoder(features)
for epoch in range(args.num_epoch):
    t = time.time()

    # if epoch == 0 or (epoch + 1) % 10 == 0:
        #     print('epoch.....................................:', epoch)
    # get mu_phi, log_cov_phi and embeddings
    mu_phi, log_cov_phi, z = model.encoder(features)
    # emb, y = model.encoder2(edges)
    # z = torch.FloatTensor(np.loadtxt('C:/Users/Dingge/Downloads/deepLsm/emb_3clusters.txt'))
    # mu_phi = z
    # A_pred, out_p = model.decoder(z, y, model.alpha, model.beta)
    A_pred = model.decoder(z, model.alpha)

    #if epoch == 0 or (epoch + 1) % 1 == 0:
    if epoch < 1 or (epoch + 1) % 1 == 0:
        # update gamma
        pi_k = model.pi_k
        log_cov_k = model.log_cov_k
        mu_k = model.mu_k
        model.update_gamma(mu_phi.detach().clone(),
                           log_cov_phi.detach().clone(),
                           pi_k, mu_k, log_cov_k, args.hidden2_dim)

        # update pi_k, mu_k and log_cov_k
        gamma = model.gamma
        model.update_others(mu_phi.detach().clone(),
                            log_cov_phi.detach().clone(),
                            gamma, args.hidden2_dim)

    pi_k = model.pi_k                    # pi_k should be a copy of model.pi_k
    log_cov_k = model.log_cov_k
    mu_k = model.mu_k
    gamma = model.gamma
    loss, loss1, loss2, loss3 = ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, args.hidden2_dim)
        
    #if epoch == 0 or (epoch + 1) % 100 == 0:
    if epoch > 1:    
        # calculate of ELBO loss
        optimizer.zero_grad()
        # update of GCN
        loss.backward()
        optimizer.step()
        # lr_s.step()

    if (epoch + 1) % 1 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_loss1=", "{:.5f}".format(loss1.item()), "train_loss2=", "{:.5f}".format(loss2.item()),
              "train_loss3=", "{:.5f}".format(loss3.item()),
              "time=", "{:.5f}".format(time.time() - t))

    # if (epoch + 1) % 100 == 0:
    #     f, ax = plt.subplots(1, figsize=(8, 8))
    #     ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
    #     ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
    #     ax.set_title("Embeddings after training!")
    #     plt.show()

    #     from sklearn.decomposition import PCA
    #     pca = PCA(n_components=2, svd_solver='full')
    #     out = pca.fit_transform(model.encoder.mean.cpu().data.numpy())
    #     mean = pca.fit_transform(model.mu_k.cpu().data.numpy())
    #     f, ax = plt.subplots(1, figsize=(15, 10))
    #     ax.scatter(out[:, 0], out[:, 1], color=labelC)
    #     ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
    #     ax.set_xlabel('PCA result of mean embedding of deepLSM')
    #     plt.show()

    store_loss[epoch] = torch.Tensor.item(loss)  # save train loss for visu
    store_loss1[epoch] = torch.Tensor.item(loss1)
    store_loss2[epoch] = torch.Tensor.item(loss2)
    store_loss3[epoch] = torch.Tensor.item(loss3)

    gamma = model.gamma.cpu().data.numpy()
    store_ari[epoch] = torch.tensor(adjusted_rand_score(labels, np.argmax(gamma, axis=1)))  # save ARI


# # plot train loss
# f, ax = plt.subplots(1, figsize=(15, 10))
# plt.subplot(231)
# plt.plot(store_loss1.cpu().data.numpy(), color='red')
# plt.title("Reconstruction loss1")
#
# plt.subplot(232)
# plt.plot(store_loss2.cpu().data.numpy(), color='red')
# plt.title("KL loss2")
#
# plt.subplot(233)
# plt.plot(store_loss3.cpu().data.numpy(), color='red')
# plt.title("Cluster loss3")
#
# plt.subplot(212)
# plt.plot(store_loss.cpu().data.numpy(), color='red')
# plt.title("Training loss in total")
#
# plt.show()

# # plot ARI
# f, ax = plt.subplots(1, figsize=(15, 10))
# ax.plot(store_ari.cpu().data.numpy(), color='blue')
# ax.set_title("ARI")
# plt.show()
#
# f, ax = plt.subplots(1, figsize=(8, 8))
# ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
# # ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
# ax.set_title("Latent Embeddings of DeepLPM", fontsize=18)
# plt.show()
# # f.savefig("C:/Users/Dingge/Desktop/results/emb_deepLPM_B.pdf", bbox_inches='tight')

# calculate ARI
gamma = model.gamma.cpu().data.numpy()
print("ARI_gamma:", adjusted_rand_score(labels, np.argmax(gamma, axis=1)))
print("Dim="+str(args.hidden2_dim))
print("Nb_of_clusters="+str(args.num_clusters))
print('Min loss:', np.min(store_loss.cpu().data.numpy()))

kmeans = KMeans(n_clusters=args.num_clusters).fit(model.encoder.mean.cpu().data.numpy())
labelk = kmeans.labels_
print("ARI_embedding:", adjusted_rand_score(labels, labelk))


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
import model_texts
# from testB import *
# from testB2 import *
# from testC import *
import math
import pickle

############
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import args

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
print('Number of clusters:.................'+str(args.num_clusters))

# Load data
if args.dataset == 'eveques':
    # simulated data
    # adj, labels = create_simuB(args.num_points, args.num_clusters)  # , oX    args.num_points, args.num_clusters, args.hidden2_dim
    # load to test edge features
    # adj = pd.read_csv('C:/Users/Dingge/Downloads/deepLsm/data/eveques_new/ResoEvequesClean2021-A.csv', header=0,
    #                   sep=';').to_numpy()  # load simu data
    # labels = np.loadtxt('/home/dliang/deepLsm/label_simu_3clusters.txt')  # load simu labels
    features, adj, edges = load_data(args.dataset)  # load cora data
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)

    # features = np.zeros((adj.shape[0], args.input_dim))
    # np.fill_diagonal(features, 1)
    # features = sp.csr_matrix(features)

    # edges1 = pd.read_csv('C:/Users/Dingge/Downloads/deepLsm/data/Eveques/ResoEvequesClean2021-Ydates.csv', header=0,
    #                      sep=';').to_numpy()
    # edges2 = pd.read_csv('C:/Users/Dingge/Downloads/deepLsm/data/Eveques/ResoEvequesClean2021-Yfonctions.csv', header=0,
    #                      sep=';').to_numpy()
    # edges3 = pd.read_csv('C:/Users/Dingge/Downloads/deepLsm/data/Eveques/ResoEvequesClean2021-Yregions.csv', header=0,
    #                      sep=';').to_numpy()
    # edges = np.array([edges1, edges2, edges3])

    # with open('edges_simu_3clusters_2texts_delta0.4', 'rb') as fp:
    #     edges = pickle.load(fp)
    # edges = np.sum(edges, axis=1)  # N * V
    # features = sp.csr_matrix(edges)  # To test node features

# else:
#     # adj, features, edges = load_data(args.dataset)
#     features, labels, adj, edges = load_data(args.dataset)  # load cora data
#     labels = labels.squeeze(1).tolist()
#     adj = sp.coo_matrix(adj)
#     edges = torch.tensor(edges)
#
#     labelC = []
#     for idx in range(len(labels)):
#         if labels[idx] == 0:
#             labelC.append('red')
#         elif labels[idx] == 1:
#             labelC.append('green')
#         elif labels[idx] == 2:
#             labelC.append('yellow')
#         elif labels[idx] == 3:
#             labelC.append('purple')
#         elif labels[idx] == 4:
#             labelC.append('grey')
#         elif labels[idx] == 5:
#             labelC.append('pink')
#         else:
#             labelC.append('blue')

# from sparsebm import SBM
# number_of_clusters = args.num_clusters
# # A number of classes must be specify. Otherwise see model selection.
# model1 = SBM(number_of_clusters, n_init=100)
# model1.fit(adj.todense(), symmetric=True)
# # print("Labels:", model.labels)
# print("ARI_SBM:", adjusted_rand_score(labels, model1.labels))

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
edges = torch.Tensor(edges)


# init model and optimizer
model = getattr(model_texts, args.model)(adj_norm)

model.pretrain(features, adj_label, edges)  # pretraining ari=0.4813

optimizer = Adam(model.parameters(), lr=args.learning_rate)  # , weight_decay=0.01
lr_s = StepLR(optimizer, step_size=10, gamma=0.95)


# train model
store_loss = torch.zeros(args.num_epoch)
store_loss1 = torch.zeros(args.num_epoch)
store_loss2 = torch.zeros(args.num_epoch)
store_loss3 = torch.zeros(args.num_epoch)
store_loss4 = torch.zeros(args.num_epoch)

store_A_pred = []
store_ari = torch.zeros(args.num_epoch)


def ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, P):  # , texts_pred
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

    # Loss4 = - torch.sum(edges * torch.log(texts_pred+1e-40))

    Loss = Loss1 + Loss2 - Loss3   #  + Loss4

    return Loss, Loss1, Loss2, -Loss3  #  , Loss4

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
    A_pred = model.decoder(z, edges, model.alpha, model.beta)

    #if epoch == 0 or (epoch + 1) % 1 == 0:
    if epoch < 1 or (epoch + 1) % 1 == 0:
        # update pi_k, mu_k and log_cov_k
        gamma = model.gamma
        model.update_others(mu_phi.detach().clone(),
                            log_cov_phi.detach().clone(),
                            gamma, args.hidden2_dim)

        # update gamma
        pi_k = model.pi_k
        log_cov_k = model.log_cov_k
        mu_k = model.mu_k
        model.update_gamma(mu_phi.detach().clone(),
                           log_cov_phi.detach().clone(), 
                           pi_k, mu_k, log_cov_k, args.hidden2_dim)

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

    # if (epoch + 1) % 600 == 0:
    #     f, ax = plt.subplots(1, figsize=(10, 15))
    #     ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
    #     ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
    #     ax.set_title("Embeddings after training!")
    #     plt.show()

    store_loss[epoch] = torch.Tensor.item(loss)  # save train loss for visu
    store_loss1[epoch] = torch.Tensor.item(loss1)
    store_loss2[epoch] = torch.Tensor.item(loss2)
    store_loss3[epoch] = torch.Tensor.item(loss3)
    # store_loss4[epoch] = torch.Tensor.item(loss4)

    gamma = model.gamma.cpu().data.numpy()
    # store_ari[epoch] = torch.tensor(adjusted_rand_score(labels, np.argmax(gamma, axis=1)))  # save ARI


# plot train loss
f, ax = plt.subplots(1, figsize=(15, 10))
plt.subplot(231)
plt.plot(store_loss1.cpu().data.numpy(), color='red')
plt.title("Reconstruction loss1")

plt.subplot(232)
plt.plot(store_loss2.cpu().data.numpy(), color='red')
plt.title("KL loss2")

plt.subplot(233)
plt.plot(store_loss3.cpu().data.numpy(), color='red')
plt.title("Cluster loss3")

plt.subplot(212)
plt.plot(store_loss.cpu().data.numpy(), color='red')
plt.title("Training loss in total")

plt.show()

print('Min loss:', np.min(store_loss.cpu().data.numpy()))

# f, ax = plt.subplots(1, figsize=(15, 10))
# plt.plot(store_loss4.cpu().data.numpy(), color='red')
# plt.title("Training loss of texts")
# plt.show()

# plot ARI
# f, ax = plt.subplots(1, figsize=(15, 10))
# ax.plot(store_ari.cpu().data.numpy(), color='blue')
# ax.set_title("ARI")
# plt.show()

labelC = []
labels = np.argmax(gamma, axis=1)
for idx in range(len(labels)):
    if labels[idx] == 0:
        labelC.append('lightblue')
    elif labels[idx] == 1:
        labelC.append('lightgreen')
    elif labels[idx] == 2:
        labelC.append('yellow')
    elif labels[idx] == 3:
        labelC.append('purple')
    elif labels[idx] == 4:
        labelC.append('blue')
    elif labels[idx] == 5:
        labelC.append('orange')
    elif labels[idx] == 6:
        labelC.append('cyan')
    elif labels[idx] == 7:
        labelC.append('red')
    elif labels[idx] == 8:
        labelC.append('green')
    elif labels[idx] == 9:
        labelC.append('pink')
    elif labels[idx] == 10:
        labelC.append('grey')
    elif labels[idx] == 11:
        labelC.append('cornflowerblue')
    elif labels[idx] == 12:
        labelC.append('darkkhaki')
    elif labels[idx] == 13:
        labelC.append('darksalmon')
    else:
        labelC.append('gold')

# f, ax = plt.subplots(1, figsize=(10, 15))
# ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
# ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
# ax.set_title("Embeddings after training!")
# plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
out = pca.fit_transform(model.encoder.mean.cpu().data.numpy())
mean = pca.fit_transform(model.mu_k.cpu().data.numpy())
f, ax = plt.subplots(1, figsize=(15, 10))
ax.scatter(out[:, 0], out[:, 1], color=labelC)
# ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
ax.set_xlabel('PCA result of embeddings of deepLSM (K=13)')
plt.show()

# calculate ARI
# gamma = model.gamma.cpu().data.numpy()
# print("ARI_gamma:", adjusted_rand_score(labels, np.argmax(gamma, axis=1)))

# kmeans = KMeans(n_clusters=args.num_clusters).fit(model.encoder.mean.cpu().data.numpy())
# labelk = kmeans.labels_
# print("ARI_embedding:", adjusted_rand_score(labels, labelk))

# import csv
# file = open("data_k=13_p=16.csv", "w")
# writer = csv.writer(file)
# mean = model.encoder.mean.cpu().data.numpy()
# for w in range(args.num_points):
#     writer.writerow([w, mean[w][0],mean[w][1],mean[w][2],mean[w][3],mean[w][4],mean[w][5],mean[w][6],mean[w][7],
#                      mean[w][8], mean[w][9], mean[w][10], mean[w][11], mean[w][12], mean[w][13], mean[w][14],
#                      mean[w][15], labels[w]])  # mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15]
# file.close()
#
# np.savetxt('pos_k=13_p=16.txt', out)
# np.savetxt('cl_k=13_p=16.txt', labels)
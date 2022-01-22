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
from testB import *
# from testB2 import *
from testC import *
import math
import pickle

############
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import args

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = torch.device('cuda:0')
# print(torch.cuda.is_available())
# print(device)
os.environ['CUDA_VISIBLE_DEVICES'] = ""
print('Number of clusters:.................'+str(args.num_clusters))

# import tracemalloc
# tracemalloc.start()

# Load data
if args.dataset == 'eveques':
    features, adj, edges = load_data(args.dataset)  # load data
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)

elif args.dataset == 'cora':
    features, adj, edges = load_data(args.dataset)  # load data
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)

    labels = sp.load_npz('./data/cora/Labels.npz')
    labels = labels.toarray()
    labels = labels.squeeze(0)

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

elif args.dataset == 'simuB':
    adj, labels = create_simuB(args.num_points, args.num_clusters, 0.3)
    features = np.zeros((adj.shape[0], args.input_dim))
    np.fill_diagonal(features, 1)
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)
    edges = 0

elif args.dataset == 'simuC':
    adj, labels = create_simuC(args.num_points, args.num_clusters)
    features = np.zeros((adj.shape[0], args.input_dim))
    np.fill_diagonal(features, 1)
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)
    edges = 0

    labelC = []
    for idx in range(len(labels)):
        if labels[idx] == 0:
            labelC.append('#7294d4')
        elif labels[idx] == 1:
            labelC.append('#fdc765')
        else:
            labelC.append('#869f82')

# else:
#     # adj, features, edges = load_data(args.dataset)
#     features, labels, adj, edges = load_data(args.dataset)  # load cora data
#     labels = labels.squeeze(1).tolist()
#     adj = sp.coo_matrix(adj)
#     edges = torch.tensor(edges)


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

# to GPU
# adj_norm = adj_norm.to(device)
# adj_label = adj_label.to(device)
# features = features.to(device)
# edges = edges.to(device)


# init model and optimizer
model = getattr(model_texts, args.model)(adj_norm)
# model.to(device)

model.pretrain(features, adj_label, edges)  # pretraining ari=0.4813

optimizer = Adam(model.parameters(), lr=args.learning_rate)  # , weight_decay=0.01
lr_s = StepLR(optimizer, step_size=10, gamma=0.95)


# train model
store_loss = torch.zeros(args.num_epoch)
store_loss1 = torch.zeros(args.num_epoch)
store_loss2 = torch.zeros(args.num_epoch)
store_loss3 = torch.zeros(args.num_epoch)

store_ari = torch.zeros(args.num_epoch)


def ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, P):

    OO = adj_label.to_dense()*(torch.log((A_pred/(1. - A_pred)) + 1e-16)) + torch.log((1. - A_pred) + 1e-16)
    OO = OO.to('cpu')
    ind = np.diag_indices(OO.shape[0])
    OO[ind[0], ind[1]] = torch.zeros(OO.shape[0])
    # OO = OO.to(device)
    Loss1 = -torch.sum(OO)
    
    KL = torch.zeros((args.num_points, args.num_clusters))  # N * K
    # KL = KL.to(device)
    for k in range(args.num_clusters):
        for i in range(args.num_points):
            KL[i, k] = 0.5 * (P*(log_cov_k[k] - log_cov_phi[i]) - P
                              + P*torch.exp(log_cov_phi)[i] / torch.exp(log_cov_k[k])
                              + torch.norm(mu_k[k] - mu_phi[i]) ** 2 / torch.exp(log_cov_k[k]))

    Loss2 = torch.sum(gamma * KL)

    Loss3 = torch.sum(gamma * (torch.log(pi_k.unsqueeze(0)) - torch.log(gamma)))

    Loss = Loss1 + Loss2 - Loss3

    return Loss, Loss1, Loss2, -Loss3


from sklearn.decomposition import PCA
def visu():
    pca = PCA(n_components=2, svd_solver='full')
    out = pca.fit_transform(model.encoder.mean.cpu().data.numpy())
    mean = pca.fit_transform(model.mu_k.cpu().data.numpy())
    f, ax = plt.subplots(1, figsize=(15, 10))
    ax.scatter(out[:, 0], out[:, 1], color=labelC)
    # ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
    ax.set_title('PCA result of embeddings of deepLSM (K='+str(args.num_clusters)+')', fontsize=18)
    plt.show()
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_ARVGA.pdf", bbox_inches='tight')

for epoch in range(args.num_epoch):
    t = time.time()

    # if epoch == 0 or (epoch + 1) % 10 == 0:
        #     print('epoch.....................................:', epoch)
    # get mu_phi, log_cov_phi and embeddings
    mu_phi, log_cov_phi, z = model.encoder(features)

    A_pred = model.decoder(z, edges, model.alpha, model.beta)

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

    if (epoch + 1) % 1000 == 0:
        visu()
        # f, ax = plt.subplots(1, figsize=(10, 15))
        # ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
        # ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
        # ax.set_title("Embeddings after training!")
        # plt.show()

    store_loss[epoch] = torch.Tensor.item(loss)  # save train loss for visu
    store_loss1[epoch] = torch.Tensor.item(loss1)
    store_loss2[epoch] = torch.Tensor.item(loss2)
    store_loss3[epoch] = torch.Tensor.item(loss3)

    gamma = model.gamma.cpu().data.numpy()
    store_ari[epoch] = torch.tensor(adjusted_rand_score(labels, np.argmax(gamma, axis=1)))  # save ARI


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

print('Min loss:', np.min(store_loss.cpu().data.numpy()), 'K='+str(args.num_clusters), str(args.use_nodes)+str(args.use_edges))

# f, ax = plt.subplots(1, figsize=(15, 10))
# plt.plot(store_loss4.cpu().data.numpy(), color='red')
# plt.title("Training loss of texts")
# plt.show()

# plot ARI
f, ax = plt.subplots(1, figsize=(15, 10))
ax.plot(store_ari.cpu().data.numpy(), color='blue')
ax.set_title("ARI")
plt.show()

# labelC = []
# # labels = np.argmax(gamma, axis=1)
# for idx in range(len(labels)):
#     if labels[idx] == 0:
#         labelC.append('lightblue')
#     elif labels[idx] == 1:
#         labelC.append('lightgreen')
#     elif labels[idx] == 2:
#         labelC.append('yellow')
#     elif labels[idx] == 3:
#         labelC.append('purple')
#     elif labels[idx] == 4:
#         labelC.append('blue')
#     elif labels[idx] == 5:
#         labelC.append('orange')
#     elif labels[idx] == 6:
#         labelC.append('cyan')
#     elif labels[idx] == 7:
#         labelC.append('red')
#     elif labels[idx] == 8:
#         labelC.append('green')
#     elif labels[idx] == 9:
#         labelC.append('pink')
#     elif labels[idx] == 10:
#         labelC.append('grey')
#     elif labels[idx] == 11:
#         labelC.append('cornflowerblue')
#     elif labels[idx] == 12:
#         labelC.append('darkkhaki')
#     elif labels[idx] == 13:
#         labelC.append('darksalmon')
#     else:
#         labelC.append('gold')

# f, ax = plt.subplots(1, figsize=(10, 15))
# ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
# ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
# ax.set_title("Embeddings after training!")
# plt.show()

# calculate ARI
# gamma = model.gamma.cpu().data.numpy()
print("ARI_gamma:", adjusted_rand_score(labels, np.argmax(gamma, axis=1)))
# np.savetxt('pred_K='+str(args.num_clusters), np.argmax(gamma, axis=1))

# kmeans = KMeans(n_clusters=args.num_clusters).fit(model.encoder.mean.cpu().data.numpy())
# labelk = kmeans.labels_
# print("ARI_embedding:", adjusted_rand_score(labels, labelk))

# import csv
# file = open('data_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.csv', "w")
# writer = csv.writer(file)
# mean = model.encoder.mean.cpu().data.numpy()
# for w in range(args.num_points):
#     writer.writerow([w, mean[w][0],mean[w][1],mean[w][2],mean[w][3],mean[w][4],mean[w][5],mean[w][6],mean[w][7],
#                      mean[w][8], labels[w]])  # mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15]
# file.close()
#
# np.savetxt('pos_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', out)
# np.savetxt('cl_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', labels)


# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')
#
# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat)
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
from test import create_simu
import math

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Load data
if args.dataset == 'simu':
    # simulated data
    adj, labels = create_simu(args.num_points, args.num_clusters, args.hidden2_dim)
    # adj = np.loadtxt('C:/Users/Dingge/Downloads/deepLsm/adj_simu_3clusters.txt')  # load simu data
    adj = sp.csr_matrix(adj)
    features = np.zeros((adj.shape[0], args.input_dim))
    np.fill_diagonal(features, 1)
    features = sp.csr_matrix(features)
    # edges = np.zeros((adj.shape[0], adj.shape[1], 2))

    # load real labels
    labelC = []
    # labels = np.loadtxt('C:/Users/Dingge/Downloads/deepLsm/label_simu_3clusters.txt')
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
number_of_clusters = 3
# A number of classes must be specify. Otherwise see model selection.
model1 = SBM(number_of_clusters, n_init=5)
model1.fit(adj.todense(), symmetric=True)
# print("Labels:", model.labels)
print("ARI_SBM:", adjusted_rand_score(labels, model1.labels))

# Some preprocessing
adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
adj = sparse_to_tuple(adj)


# Create Model
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].astype(float).T),
                            torch.FloatTensor(adj_norm[1].astype(float)),
                            torch.Size(adj_norm[2]))
adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0].astype(float).T),
                            torch.FloatTensor(adj[1].astype(float)),
                            torch.Size(adj[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].astype(float).T),
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2]))
# edges = torch.Tensor(edges)


# init model and optimizer
model = getattr(model, args.model)(adj_norm)

model.pretrain(features, adj, labels, labelC)  # pretraining ari=0.4813

optimizer = Adam(model.parameters(), lr=args.learning_rate)  # , weight_decay=0.01
lr_s = StepLR(optimizer, step_size=50, gamma=0.5)


# train model
store_loss = torch.zeros(args.num_epoch)
store_loss1 = torch.zeros(args.num_epoch)
store_loss2 = torch.zeros(args.num_epoch)
store_loss3 = torch.zeros(args.num_epoch)

store_A_pred = []
store_ari = torch.zeros(args.num_epoch)


def ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred):
    Loss1 = F.binary_cross_entropy(A_pred.view(-1), adj.to_dense().view(-1))
    # Loss1 = Loss1 * args.num_points

    KL = torch.zeros((args.num_points, args.num_clusters))  # N * K
    for k in range(args.num_clusters):
        for i in range(args.num_points):
            P = args.hidden2_dim
            KL[i, k] = 0.5 * (P * (log_cov_k[k] - log_cov_phi[i]) - P
                              + torch.exp(log_cov_phi)[i] / torch.exp(log_cov_k[k])
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
    # z = torch.FloatTensor(np.loadtxt('C:/Users/Dingge/Downloads/deepLsm/emb_3clusters.txt'))
    # mu_phi = z
    A_pred = model.decoder(z, model.alpha)

    if epoch == 0 or (epoch + 1) % 10 == 0:
        # update gamma
        pi_k = model.pi_k
        log_cov_k = model.log_cov_k
        mu_k = model.mu_k
        model.update_gamma(mu_phi, log_cov_phi, pi_k, mu_k, log_cov_k)

        # update pi_k, mu_k and log_cov_k
        gamma = model.gamma
        model.update_others(mu_phi, log_cov_phi, gamma)

    # calculate of ELBO loss
    pi_k = model.pi_k
    log_cov_k = model.log_cov_k
    mu_k = model.mu_k
    gamma = model.gamma
    loss, loss1, loss2, loss3 = ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred)

    # if epoch == 0 or (epoch + 1) % 10 == 0:
    # update of GCN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # lr_s.step()

    if (epoch + 1) % 10 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_loss1=", "{:.5f}".format(loss1.item()), "train_loss2=", "{:.5f}".format(loss2.item()),
              "train_loss3=", "{:.5f}".format(loss3.item()),
              "time=", "{:.5f}".format(time.time() - t))

    if (epoch + 1) % 50 == 0:
        f, ax = plt.subplots(1, figsize=(10, 15))
        ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
        ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
        ax.set_title("Embeddings after training!")
        plt.show()

    store_loss[epoch] = torch.Tensor.item(loss)  # save train loss for visu
    store_loss1[epoch] = torch.Tensor.item(loss1)
    store_loss2[epoch] = torch.Tensor.item(loss2)
    store_loss3[epoch] = torch.Tensor.item(loss3)

    gamma = model.gamma.cpu().data.numpy()
    store_ari[epoch] = torch.tensor(adjusted_rand_score(labels, np.argmax(gamma, axis=1)))  # save ARI


# plot train loss
# f, ax = plt.subplots(1, figsize=(15, 10))
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

# plot ARI
f, ax = plt.subplots(1, figsize=(15, 10))
ax.plot(store_ari.cpu().data.numpy(), color='blue')
ax.set_title("ARI")
plt.show()

# calculate ARI
gamma = model.gamma.cpu().data.numpy()
print("ARI_gamma:", adjusted_rand_score(labels, np.argmax(gamma, axis=1)))

kmeans = KMeans(n_clusters=args.num_clusters).fit(model.encoder.mean.cpu().data.numpy())
labelk = kmeans.labels_
print("ARI_embedding:", adjusted_rand_score(labels, labelk))


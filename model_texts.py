import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import itertools
import os
import numpy as np
import args
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from input_data import load_data
from sklearn.cluster import KMeans


# load real labels
# if args.dataset == 'simu':
# adj, labels = create_simu(args.num_points, args.num_clusters, args.hidden2_dim)
# labelC = []
# # labels = np.loadtxt('C:/Users/Dingge/Downloads/deepLsm/data/label_simu_3clusters.txt')
# for idx in range(len(labels)):
# 	if labels[idx] == 0:
# 		labelC.append('red')
# 	elif labels[idx] == 1:
# 		labelC.append('green')
# 	# elif labels[idx] == 2:
# 	# 	labelC.append('yellow')
# 	# elif labels[idx] == 3:
# 	# 	labelC.append('purple')
# 	else:
# 		labelC.append('blue')

# else:
# 	features, labels, adj, edges = load_data(args.dataset)
# 	labels = labels.squeeze(1).tolist()

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


# def cluster_acc(A_pred, A):
#     from sklearn.utils.linear_assignment_ import linear_assignment
#     assert A_pred.size == A.size
#     D = max(A_pred.max(), A.max())+1
#     w = np.zeros((D,D), dtype=np.int64)
#     for i in range(A_pred.size):
#         w[A_pred[i], A[i]] += 1
#     ind = linear_assignment(w.max() - w)
#     return sum([w[i,j] for i,j in ind])*1.0/A_pred.size, w

# Graph convolutional layers
class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs  # features: N * D
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)  # adj: N * N
        outputs = self.activation(x)  # outputs: N * D
        return outputs


class Encoder(nn.Module):
    def __init__(self, adj_norm):
        super(Encoder, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj_norm)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj_norm, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, 1, adj_norm, activation=lambda x: x)

        #  self.en_drop = nn.Dropout(0.2)
        # self.mu_bn = nn.BatchNorm1d(args.hidden2_dim)
        # self.logv_bn = nn.BatchNorm1d(1)

    def forward(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)  # N * P
        self.logstd = self.gcn_logstddev(hidden)  # N * 1
        # self.mean = self.mu_bn(self.gcn_mean(hidden))  # N * P
        # self.logstd = self.logv_bn(self.gcn_logstddev(hidden))  # N * 1
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        #*M*# 
        sampled_Z = gaussian_noise * torch.exp(self.logstd / 2) + self.mean  # embeddings: N * 
        return self.mean, self.logstd, sampled_Z


# class EncoderText(nn.Module):
#     def __init__(self):
#         super(EncoderText, self).__init__()
#         self.D1 = nn.Linear(args.num_words, args.hidden3_dim)
#         self.D2 = nn.Linear(args.hidden3_dim, args.nb_of_topics)
#
#     def forward(self, E):
#         hidden = self.D1(E)
#         # print(hidden.shape)
#         self.emb = self.D2(F.relu(hidden))  # theta in LDA
#         # print(self.emb)
#         #*M*#
#         sampled_Y = F.softmax(self.emb, dim=1)
#         # print(sampled_Y)
#         return self.emb, sampled_Y

class Decoder(nn.Module):
    # input: N * P.
    def __init__(self):  # beta, Edges,
        super(Decoder, self).__init__()
        # self.D3 = nn.Linear(args.nb_of_topics, args.num_words, bias=True)  # beta
        # self.bn = nn.BatchNorm1d(args.num_points)

    def forward(self, Z, Y, alpha, beta):
        inner_product = torch.matmul(Z, Z.T)
        tnp = torch.sum(Z ** 2, axis=1).reshape(-1, 1).expand(size=inner_product.shape)
        if args.use_edges == True:
            # out_p = F.softmax(self.D3(Y), dim=1)  # beta in LDA
            A_pred = torch.sigmoid(
                - (tnp - 2 * inner_product + tnp.T) + alpha + torch.matmul(Y, beta))
            # print(- (tnp - 2 * inner_product + tnp.T) + alpha + torch.matmul(Y, beta))
            # print(A_pred)
        else:
            A_pred = torch.sigmoid(- (tnp - 2 * inner_product + tnp.T) + alpha)
        # + torch.matmul(self.Edges, self.beta).squeeze(-1)
        # A_pred_b = torch.bernoulli(A_pred)

        return A_pred  # , out_p

class deepLPM(nn.Module):
    def __init__(self, adj_norm):
        super(deepLPM, self).__init__()
        self.adj_norm = adj_norm
        self.encoder = Encoder(adj_norm)
        # self.encoder2 = EncoderText()
        self.decoder = Decoder()

        self.alpha = nn.Parameter(torch.tensor(0.2), requires_grad=True)  #*M*#
        self.beta = nn.Parameter(torch.FloatTensor(args.nb_of_edges, ).fill_(1) / args.nb_of_edges, requires_grad=True)

        self.gamma = nn.Parameter(torch.FloatTensor(args.num_points, args.num_clusters).fill_(0.1),
                                  requires_grad=False)  # N * K
        self.pi_k = nn.Parameter(torch.FloatTensor(args.num_clusters, ).fill_(1) / args.num_clusters,
                                 requires_grad=False)  # K
        # self.mu_k = nn.Parameter(torch.FloatTensor(args.num_clusters, args.hidden2_dim).fill_(0.1), requires_grad=False)  # K * P
        self.mu_k = nn.Parameter(torch.FloatTensor(np.random.multivariate_normal(np.zeros(args.hidden2_dim),
                                                                                 np.eye(args.hidden2_dim),
                                                                                 args.num_clusters)),
                                 requires_grad=False)
        self.log_cov_k = nn.Parameter(torch.FloatTensor(args.num_clusters, 1).fill_(0.1), requires_grad=False)  # K

    # pre-train of graph embeddings Z to initialize parameters of cluster
    def pretrain(self, X, adj_label, edges):
        if not os.path.exists('./pretrain_model_stop.pk'):

            optimizer = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=args.pre_lr)

            store_pre_loss = torch.zeros(args.pre_epoch)
            for epoch in range(args.pre_epoch):
                z_mu, z_log_sigma, z = self.encoder(X)
                # emb, y = self.encoder2(edges)
                A_pred = self.decoder(z, edges, self.alpha, self.beta)
                # print(adj_label.to_dense())
                loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
                kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * z_log_sigma - z_mu ** 2 - torch.exp(z_log_sigma) ** 2).sum(1).mean()
                loss -= kl_divergence  # to train cora, we need to add the kl divergence

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 1 == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()))

                store_pre_loss[epoch] = torch.Tensor.item(loss)

            # self.encoder.gcn_logstddev.load_state_dict(self.encoder.gcn_mean.state_dict())
            #
            # Z = []
            # with torch.no_grad():
            # 		z1, z2, z = self.encoder(X)
            # 		assert F.mse_loss(z1, z2) == 0
            # 		Z.append(z1)
            #
            # Z = torch.cat(Z, 0).detach().cpu().numpy()

            Z = z.detach().cpu().numpy()

            # plot loss
            # f, ax = plt.subplots(1, figsize=(15, 10))
            # ax.plot(store_pre_loss, color='red')
            # ax.set_title("pre-training loss")
            # plt.show()

            kmeans = KMeans(n_clusters=args.num_clusters).fit(Z)
            labelk = kmeans.labels_
            print(labelk)
            # print("pretraining ARI embedding:", adjusted_rand_score(labels, labelk))
            # labelC = []
            # for idx in range(len(labelk)):
            #     if labelk[idx] == 0:
            #         labelC.append('lightblue')
            #     elif labelk[idx] == 1:
            #         labelC.append('lightgreen')
            #     elif labelk[idx] == 2:
            #         labelC.append('yellow')
            #     elif labelk[idx] == 3:
            #         labelC.append('purple')
            #     elif labelk[idx] == 4:
            #         labelC.append('blue')
            #     elif labelk[idx] == 5:
            #         labelC.append('orange')
            #     elif labelk[idx] == 6:
            #         labelC.append('cyan')
            #     elif labelk[idx] == 7:
            #         labelC.append('red')
            #     elif labelk[idx] == 8:
            #         labelC.append('green')
            #     else:
            #         labelC.append('pink')
            #
            # f, ax = plt.subplots(1, figsize=(10, 15))
            # ax.scatter(Z[:, 0], Z[:, 1], color=labelC)
            # # ax.scatter(self.mu_k[:, 0], self.mu_k[:, 1], color='black', s=50)
            # ax.set_title('Embeddings after pretraining')
            # plt.show()

            # Additional lines to initialize gamma based on the k-means on the latent embeddings
            self.gamma.fill_(1e-16)
            seq = np.arange(0, len(self.gamma))
            positions = np.vstack((seq, labelk))
            self.gamma[positions] = 1.
            print(self.gamma)

            # gmm = GaussianMixture(n_components=args.num_clusters, covariance_type='diag')
            # pre = gmm.fit_predict(Z)
            # print("pretraining ARI GMM:", adjusted_rand_score(labels, pre))
            # print('Acc={:.4f}%'.format(cluster_acc(pre, np.array(clc))[0] * 100))
            # pre = np.argmax(np.random.multinomial(1, [1 / args.num_clusters] * args.num_clusters, size=args.num_points),
            #                 axis=1)
            # print(pre)
            # print("pretraining ARI random:", adjusted_rand_score(labels, pre))

            # f, ax = plt.subplots(1, figsize=(15, 10))
            # ax.scatter(Z[:, 0], Z[:, 1], color=pre)
            # plt.show()

            # self.pi_k.data = torch.from_numpy(gmm.weights_).float()
            # self.mu_k.data = torch.from_numpy(gmm.means_).float()
            # self.log_cov_k.data = torch.log(torch.from_numpy(gmm.covariances_)).float()

            # if adjusted_rand_score(labels, labelk) < 0.009:
            #     deepLPM.pretrain(self, X, adj_label, edges, labels, labelC)

            # else:
            #     torch.save(self.state_dict(), './pretrain_model.pk')

            # if min(store_pre_loss) > 0.15:
            #     deepLPM.pretrain(self, X, adj_label, edges)

            print('Finish pretraining!')

        else:

            self.load_state_dict(torch.load('./pretrain_model.pk'))
            print('pi:', self.pi_k)
            print('mu:', self.mu_k)
            print('cov:', self.log_cov_k)

    # Functions for the initialization of cluster parameters
    def update_gamma(self, mu_phi, log_cov_phi, pi_k, mu_k, log_cov_k, P):
        # P = args.hidden2_dim
        # KL = 0.5 * (torch.sum(P * (log_cov_k - log_cov_phi.unsqueeze(1)) - P
        # 					  + torch.exp(log_cov_phi.unsqueeze(1)) / torch.exp(log_cov_k)
        # 					  + (mu_k - mu_phi.unsqueeze(1)) ** 2 / torch.exp(log_cov_k), axis=2))  # N * K

        det = 1e-16
        KL = torch.zeros((args.num_points, args.num_clusters))  # N * K
        for k in range(args.num_clusters):
            for i in range(args.num_points):
                KL[i, k] = 0.5 * (P*(log_cov_k[k] - log_cov_phi[i]) - P
                                  + P*torch.exp(log_cov_phi)[i] / torch.exp(log_cov_k[k])
                                  + torch.norm(mu_k[k] - mu_phi[i]) ** 2 / torch.exp(log_cov_k[k]))

        # for k in range(args.num_clusters):
        #     P = args.hidden2_dim
        #     KL[:, k] = 0.5 * (P * (log_cov_k[k] - log_cov_phi) - P
        #                       + (torch.exp(log_cov_phi) + torch.sum((mu_k[k].expand_as(mu_phi) - mu_phi) ** 2
        #
        #                                                             , axis=1)) / torch.exp(log_cov_k)[k])

        denominator = torch.sum(pi_k.unsqueeze(0) * torch.exp(-KL), axis=1)
        # print(denominator.shape)
        for k in range(args.num_clusters):
            self.gamma.data[:, k] = pi_k[k] * torch.exp(-KL[:, k]) / denominator + det

        ####################################################################################################
        # K = - 2 * (torch.log(pi_k.unsqueeze(0)) - KL)  # N * K
        #
        # for k in range(args.num_clusters):
        #     for l in range(args.num_clusters):
        #         denominator = 0
        #         denominator += torch.exp(0.5 * (K[:, k] - K[:, l])) + det
        #         print(l, denominator)
        #     self.gamma.data[:, k] = 1 / denominator

            # self.gamma.data[:, k] = 1 / torch.sum(torch.exp(0.5 * (K[:, k].unsqueeze(1) - K)), axis=1) + det

        # print('Update gamma!')
        # print('mu_phi:', mu_phi)
        # print('log_cov_phi', log_cov_phi)
        # print('KL:', KL)
        # print('K:', K)
        # print('gamma:', self.gamma)

    def update_others(self, mu_phi, log_cov_phi, gamma, P):
        N_k = torch.sum(gamma, axis=0)

        self.pi_k.data = N_k / args.num_points

        for k in range(args.num_clusters):
            gamma_k = gamma[:, k]  # N * 1
            self.mu_k.data[k] = torch.sum(mu_phi * gamma_k.unsqueeze(1), axis=0) / N_k[k]
            mu_k = self.mu_k

            diff = P*torch.exp(log_cov_phi) + torch.sum((mu_k[k].unsqueeze(0) - mu_phi) ** 2, axis=1).unsqueeze(1)
            cov_k = torch.sum(gamma_k.unsqueeze(1) * diff, axis=0) / (P * N_k[k])
            self.log_cov_k.data[k] = torch.log(cov_k)
            # print(self.log_cov_k)

    # print('Update pi_k, mu_k, log_cov_k!')
    # print('pi_k:', self.pi_k)
    # print('mu_k:', self.mu_k)
    # print('log_cov_k:', self.log_cov_k)
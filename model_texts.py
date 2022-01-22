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

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = torch.device('cuda:0')

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim, dtype = torch.float32) * 2 * init_range - init_range
    # initial = initial.to(device)
    return nn.Parameter(initial)


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

    def forward(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)  # N * P
        self.logstd = self.gcn_logstddev(hidden)  # N * 1
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        # gaussian_noise = gaussian_noise.to(device)
            # *M*#
        sampled_Z = gaussian_noise * torch.exp(self.logstd / 2) + self.mean  # embeddings: N * 
        return self.mean, self.logstd, sampled_Z


class Decoder(nn.Module):
    # input: N * P.
    def __init__(self):  # beta, Edges,
        super(Decoder, self).__init__()

    def forward(self, Z, Y, alpha, beta):
        inner_product = torch.matmul(Z, Z.T)
        tnp = torch.sum(Z ** 2, axis=1).reshape(-1, 1).expand(size=inner_product.shape)
        if args.use_edges == True:
            A_pred = torch.sigmoid(
                - (tnp - 2 * inner_product + tnp.T) + alpha + torch.matmul(Y, beta))
        else:
            A_pred = torch.sigmoid(- (tnp - 2 * inner_product + tnp.T) + alpha)

        return A_pred


class deepLPM(nn.Module):
    def __init__(self, adj_norm):
        super(deepLPM, self).__init__()
        self.adj_norm = adj_norm
        self.encoder = Encoder(adj_norm)
        # self.encoder2 = EncoderText()
        self.decoder = Decoder()

        self.alpha = nn.Parameter(torch.tensor(0.2, dtype = torch.float32), requires_grad=True)  # *M*#
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
                A_pred = self.decoder(z, edges, self.alpha, self.beta)
                loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
                kl_divergence = 0.5 / A_pred.size(0) * (
                            1 + 2 * z_log_sigma - z_mu ** 2 - torch.exp(z_log_sigma) ** 2).sum(1).mean()
                loss -= kl_divergence  # to train cora, we need to add the kl divergence

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 1 == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()))

                store_pre_loss[epoch] = torch.Tensor.item(loss)

            Z = z.detach().cpu().numpy()

            # plot loss
            # f, ax = plt.subplots(1, figsize=(15, 10))
            # ax.plot(store_pre_loss, color='red')
            # ax.set_title("pre-training loss")
            # plt.show()

            kmeans = KMeans(n_clusters=args.num_clusters).fit(Z)
            labelk = kmeans.labels_

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
        det = 1e-16
        KL = torch.zeros((args.num_points, args.num_clusters), dtype = torch.float32)  # N * K
        # KL = KL.to(device)
        for k in range(args.num_clusters):
            for i in range(args.num_points):
                KL[i, k] = 0.5 * (P * (log_cov_k[k] - log_cov_phi[i]) - P
                                  + P * torch.exp(log_cov_phi)[i] / torch.exp(log_cov_k[k])
                                  + torch.norm(mu_k[k] - mu_phi[i]) ** 2 / torch.exp(log_cov_k[k]))

        denominator = torch.sum(pi_k.unsqueeze(0) * torch.exp(-KL), axis=1, dtype = torch.float32)
        for k in range(args.num_clusters):
            self.gamma.data[:, k] = pi_k[k] * torch.exp(-KL[:, k]) / denominator + det

    def update_others(self, mu_phi, log_cov_phi, gamma, P):
        N_k = torch.sum(gamma, axis=0, dtype = torch.float32)

        self.pi_k.data = N_k / args.num_points

        for k in range(args.num_clusters):
            gamma_k = gamma[:, k]  # N * 1
            self.mu_k.data[k] = torch.sum(mu_phi * gamma_k.unsqueeze(1), axis=0, dtype = torch.float32) / N_k[k]
            mu_k = self.mu_k

            diff = P * torch.exp(log_cov_phi) + torch.sum((mu_k[k].unsqueeze(0) - mu_phi) ** 2, axis=1, dtype = torch.float32).unsqueeze(1)
            cov_k = torch.sum(gamma_k.unsqueeze(1) * diff, axis=0, dtype = torch.float32) / (P * N_k[k])
            self.log_cov_k.data[k] = torch.log(cov_k)
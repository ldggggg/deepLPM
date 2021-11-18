import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import args

class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_pi = GraphConvSparse(args.hidden1_dim, args.num_clusters, adj, activation=lambda x:x)

		# self.dropout = torch.nn.Dropout(0.2)
		self.alpha = torch.nn.Parameter(torch.tensor(0.3))
		# self.beta = torch.tensor(torch.zeros(2, 1), requires_grad=False)  # torch.nn.Parameter(torch.randn(2, 1))

		self.mean = torch.nn.Parameter(torch.randn(args.num_clusters, args.hidden2_dim))  # dim = K * D
		self.logstd = torch.nn.Parameter(torch.randn(args.num_clusters, args.hidden2_dim))  # dim = K * D

	def RT(self, mean, logstd):  # re-parameterization trick
		gaussian_noise = torch.randn(1, args.hidden2_dim)  # dim = 1 * D
		sampled_G = mean + gaussian_noise*torch.exp(logstd)
		return sampled_G

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.pi = self.gcn_pi(hidden)  # dim = N * K
		G = torch.zeros((args.num_clusters, args.hidden2_dim))  # dim = K * D
		sampled_z = torch.zeros((args.num_points, args.hidden2_dim))  # dim = N * D
		for k in range(args.num_clusters):
			G[k] = self.RT(self.mean[k], self.logstd[k])  # re-parameterization trick,  dim = 1 * D
		for i in range(args.num_points):
			sampled_z[i] = torch.matmul(torch.softmax(self.pi[i].unsqueeze(0), dim=0), G)  # dim = 1 * K * K * D = 1 * D
		return sampled_z

	def forward(self, X):  # , Edges
		Z = self.encode(X)
		A_pred = Graph_Construction(Z, self.alpha, type_of='distance').Middle()  # self.beta, Edges,
		# A_pred =  dot_product_decode(Z)
		return A_pred

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs   # features
		x = torch.mm(x, self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

class Graph_Construction:
    # Input: n * d.
    def __init__(self, X, alpha, type_of='distance'):  # beta, Edges,
        self.X = X
        self.type_of = type_of
        self.alpha = alpha
        # self.beta = beta
        # self.Edges = Edges

    def Middle(self):
            if self.type_of == 'projection':
                Inner_product = self.X.mm(self.X.T)
                # Graph_middle = torch.sigmoid(Inner_product)  # VGAE
                Graph_middle = torch.sigmoid(Inner_product + self.alpha)   # + torch.matmul(self.Edges, self.beta).squeeze(-1)
                # deepLSM: logit du produit scalaire normalisé, sans alpha et beta.

            else:
                Inner_product = self.X.mm(self.X.T)
                tnp = torch.sum(self.X**2, axis=1).reshape(-1,1).expand(size = Inner_product.shape)
                Graph_middle = torch.sigmoid(- (tnp - 2*Inner_product + tnp.T) + self.alpha)  # + torch.matmul(self.Edges, self.beta).squeeze(-1)

            return Graph_middle

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)
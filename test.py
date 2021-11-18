import numpy as np
from scipy.spatial.distance import pdist, squareform
import args
import pickle

def create_simu(N, K):
# N = args.num_points
# K = args.num_clusters
# D = args.hidden2_dim

    delta = 0.95
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
    x3 = np.random.multivariate_normal(mu3, sigma3, N-2*(N//K))
    # x4 = np.random.multivariate_normal(mu4, sigma4, 120)
    # x5 = np.random.multivariate_normal(mu5, sigma5, 120)

    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots(1,figsize=(8,8))
    # ax.scatter(x1[:,0], x1[:,1], color = '#7294d4')
    # ax.scatter(x2[:,0], x2[:,1], color = '#fdc765')
    # ax.scatter(x3[:,0], x3[:,1], color = '#869f82')
    # # ax.scatter(x4[:,0], x4[:,1], color = 'y')
    # # ax.scatter(x5[:,0], x5[:,1], color = 'purple')
    # ax.set_title("Original Embeddings of Scenario A (Delta=0.5)", fontsize=18)
    # plt.show()

    X = np.concatenate((x1,x2,x3), axis=0)
    # np.savetxt('emb_3clusters.txt', X)
    # np.savetxt('mu_3clusters.txt', z_mu)
    # np.savetxt('cov_3clusters.txt', z_log_sigma)
    Label1 = np.repeat(0, N//K)
    Label2 = np.repeat(1, N//K)
    Label3 = np.repeat(2, N-2*(N//K))
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

    np.savetxt('adj_simuA_3clusters.txt', A)
    np.savetxt('label_simuA_3clusters.txt', Label)
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_orig_A.pdf", bbox_inches='tight')

    return A, Label

A, Label = create_simu(args.num_points, args.num_clusters)

############## loading and manipulatind docs and vocabulary  ###############
# dct = pickle.load(open('dizionario_2texts.pkl', 'rb'))
# dctn = dct.token2id
# V = len(dctn)
#
# with open('sim_data_docs_deeplsm_2texts', 'rb') as fp:
#     docs = pickle.load(fp)
# # num version of docs
# ndocs = []
# for doc in range(len(docs)):
#     tmp = []
#     for word in docs[doc]:
#         tmp.append(dctn[word])
#     ndocs.append(tmp)
# # complete dtm for row
# cdtm = []
# for idx in range(len(ndocs)):
#     cdtm.append(np.bincount(ndocs[idx], minlength=V))
# cdtm = np.asarray(cdtm, dtype='float32')
#
# edges = np.zeros((args.num_points, args.num_points, V))
# clr = np.where(A == 1)[0]
# clc = np.where(A == 1)[1]
# for i in range(len(clr)):
#     edges[clr[i],clc[i],:] = edges[clc[i],clr[i],:] = cdtm[args.num_points*clr[i]+clc[i],:]
#
# with open('edges_simu_3clusters_2texts_delta0.4', 'wb') as fp:
#     pickle.dump(edges, fp)
#
# # with open('edges_simu_3clusters_2texts', 'rb') as fp:
# #     edges_test = pickle.load(fp)
# # sum = np.sum(edges_test, axis=1)

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
import args
import pickle

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D

def create_simuD(N, K):

    X, color = datasets.make_swiss_roll(n_samples=200)
    # Plot result
    # fig = plt.figure()
    # ax = fig.add_subplot(211, projection="3d")
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    out = pca.fit_transform(X)
    f, ax = plt.subplots(1, figsize=(10, 10))
    ax.scatter(out[:, 0], out[:, 1], color='blue')
    ax.scatter(-out[:, 0], -out[:, 1]-2, color='red')
    # ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
    ax.set_xlabel('PCA result of Swiss roll')
    plt.show()

    K2 = np.concatenate((np.asarray(-out[:, 0]).reshape(-1,1), np.asarray(-out[:, 1]-2).reshape(-1,1)), axis=1)
    C = np.concatenate((out, K2), axis=0)

    Label1 = np.repeat(0, N//K)
    Label2 = np.repeat(1, N-(N//K))
    Label = np.concatenate((Label1, Label2), axis=0)

    dst = pdist(C, 'euclidean')
    dst = squareform(dst)
    alpha = 0.2
    from scipy.special import expit
    from scipy.stats import bernoulli
    A = np.zeros((N, N))
    for i in range(N - 1):
        for j in range(i + 1, N):
            prob = expit(alpha - dst[i, j])
            A[i, j] = A[j, i] = bernoulli.rvs(prob, loc=0, size=1)

    # np.savetxt('adj_simuC_3clusters.txt', A)
    # np.savetxt('label_simuC_3clusters.txt', Label)

    return A, Label

A, Label = create_simuD(args.num_points, args.num_clusters)
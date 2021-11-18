import numpy as np
from scipy.spatial.distance import pdist, squareform
import args
import pickle

def create_simuC(N, K):

    x = np.random.uniform(-1,1,N//K)
    c = np.random.multinomial(1, [0.5,0.5], size=N//K)
    c = np.argmax(c, axis=1)
    y = np.sqrt(1 - x**2) + np.random.normal(0,0.1,N//K)
    y[c==1] = -y[c==1]

    x2 = np.random.uniform(-5,5,N//K)
    c = np.random.multinomial(1, [0.5,0.5], size=N//K)
    c = np.argmax(c, axis=1)
    y2 = np.sqrt(25-x2**2) + np.random.normal(0,0.1,N//K)
    y2[c==1] = -y2[c==1]

    x3 = np.random.uniform(-10,10,N-2*(N//K))
    c = np.random.multinomial(1, [0.5,0.5], size=N-2*(N//K))
    c = np.argmax(c, axis=1)
    y3 = np.sqrt(100-x3**2) + np.random.normal(0,0.1,N-2*(N//K))
    y3[c==1] = -y3[c==1]

    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots(1, figsize=(8, 8))
    # ax.scatter(x, y, color='#7294d4')
    # ax.scatter(x2, y2, color='#fdc765')
    # ax.scatter(x3, y3, color='#869f82')
    # ax.set_title("Original Embeddings of Scenario C", fontsize=18)
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_orig_C.pdf", bbox_inches='tight')

    K1 = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)
    K2 = np.concatenate((x2.reshape(-1,1),y2.reshape(-1,1)), axis=1)
    K3 = np.concatenate((x3.reshape(-1,1),y3.reshape(-1,1)), axis=1)

    C= np.concatenate((K1,K2,K3), axis=0)
    # np.savetxt('emb_3clusters.txt', K)

    Label1 = np.repeat(0, N//K)
    Label2 = np.repeat(1, N//K)
    Label3 = np.repeat(2, N-2*(N//K))
    Label = np.concatenate((Label1,Label2,Label3), axis=0)


    dst = pdist(C, 'euclidean')
    dst = squareform(dst)

    alpha = 0.2
    from scipy.special import expit
    from scipy.stats import bernoulli
    A = np.zeros((N, N))
    for i in range(N - 1):
        for j in range(i + 1, N):
            prob = expit(alpha - dst[i,j])
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    np.savetxt('adj_simuC_3clusters.txt', A)
    np.savetxt('label_simuC_3clusters.txt', Label)

###################### for LPCM #####################
    a = np.sum(A, axis=0)
    arr = np.delete(A, np.where(a == 0), axis=0)
    arr = np.delete(arr, np.where(a == 0), axis=1)
    lab = np.delete(Label, np.where(a == 0), axis=0)

    np.savetxt('adj_simuC_3clusters_LPCM.txt', arr)
    np.savetxt('label_simuC_3clusters_LPCM.txt', lab)

    return A, Label

A, Label = create_simuC(args.num_points, args.num_clusters)
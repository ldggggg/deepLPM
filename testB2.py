import numpy as np
import args
import pickle


def create_simuB2(N, K):
# N = args.num_points
# K = args.num_clusters
# D = args.hidden2_dim
# K = 3
    Pi = np.zeros((K, K))
    # delta = 0.7  # (0,1)
    b = 0.1
    a = 0.001

    Pi[0,0] = a
    Pi[0,1] = b
    Pi[1,0] = b
    Pi[1,1] = a

    Rho = [0.1, 0.9]
    # N=5
    c = np.random.multinomial(1, Rho, size=N)
    c = np.argmax(c, axis=1)


    from scipy.stats import bernoulli
    A = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            prob = Pi[c[i], c[j]]
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    label = []
    for idx in range(len(c)):
        if c[idx] == 0:
            label.append('red')
        elif c[idx] == 1:
            label.append('green')
        # elif labels[idx] == 2:
        #     labelC.append('yellow')
        # elif labels[idx] == 3:
        #     labelC.append('purple')
        else:
            label.append('blue')


    np.savetxt('adj_simuB2_2clusters.txt', A)
    np.savetxt('label_simuB2_2clusters.txt', c)

    return A, c

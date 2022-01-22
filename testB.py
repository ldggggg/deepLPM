import numpy as np
import args
import pickle

def create_simuB(N, K, delta):
# N = args.num_points
# K = args.num_clusters
# D = args.hidden2_dim
# K = 3
    Pi = np.zeros((K, K))
    # delta = 0.4  # (0,1)
    b = 0.25
    c = b
    a = 0.01 + (1-delta) * (b-0.01)

    Pi[0,0] = a
    Pi[0,1] = b
    Pi[0,2] = b
    Pi[1,0] = b
    Pi[2,0] = b
    Pi[1,1] = c
    Pi[1,2] = a
    Pi[2,1] = a
    Pi[2,2] = c

    Rho = [0.1, 0.45, 0.45]
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
            label.append('#7294d4')
        elif c[idx] == 1:
            label.append('#fdc765')
        # elif labels[idx] == 2:
        #     labelC.append('yellow')
        # elif labels[idx] == 3:
        #     labelC.append('purple')
        else:
            label.append('#869f82')


    np.savetxt('adj_simuB_3clusters.txt', A)
    np.savetxt('label_simuB_3clusters.txt', c)
    print('Delta is................: ', delta)
    print('Clusters='+str(K))

    return A, c

# A, Label = create_simuB(args.num_points, args.num_clusters, 0.5)
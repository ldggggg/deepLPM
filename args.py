### CONFIGS ###
dataset = 'cora'  # 'cora'
model = 'deepLPM'

if dataset == 'simuC':
    use_nodes = False
    use_edges = False
    nb_of_edges = 0

    num_points = 900  # N (2708 in cora)
    input_dim = 900  # D (1433 in cora)
    hidden1_dim = 64
    hidden2_dim = 16  # P
    num_clusters = 3  # K (7 in cora)

    num_epoch = 600  # 100
    learning_rate = 2e-3  # 2e-3 or 5e-3 (B: delta<0.4)
    pre_lr = 0.1  # 0.005 (Cora)  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 70  # 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50

elif dataset == 'eveques':
    use_nodes = False
    use_edges = True
    nb_of_edges = 3

    num_points = 1287  # N
    if use_nodes == True:
        input_dim = 10  # D
    else:
        input_dim = 1287  # D
    hidden1_dim = 64
    hidden2_dim = 16  # 10logK # P
    num_clusters = 3  # K

    num_epoch = 1000  # 100
    learning_rate = 2e-3  # 2e-3 (before) or 5e-3 (B: delta<0.4)
    pre_lr = 1e-3  # 0.01 (before)  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 100  # 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50

elif dataset == 'cora':
    use_nodes = True
    use_edges = True
    nb_of_edges = 2

    num_points = 2708  # N (2708 in cora)
    input_dim = 1433  # D (1433 in cora)
    hidden1_dim = 64
    hidden2_dim = 32  # P
    num_clusters = 7  # K (7 in cora)

    num_epoch = 2000  # 100
    learning_rate = 1e-2  # 2e-3 or 5e-3 (B: delta<0.4)
    pre_lr = 0.005  # 0.005 (Cora)  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 100  # 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50
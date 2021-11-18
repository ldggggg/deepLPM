### CONFIGS ###
dataset = 'eveques'  # 'cora'
model = 'deepLPM'

if dataset == 'simu':
    num_points = 300  # N (2708 in cora)
    input_dim = 300  # D (1433 in cora)
    hidden1_dim = 64
    hidden2_dim = 16  # P
    num_clusters = 3  # K (7 in cora)

    num_epoch = 600  # 100
    learning_rate = 2e-3  # 2e-3 or 5e-3 (B: delta<0.4)
    pre_lr = 0.1  # 0.005 (Cora)  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 70  # 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50

elif dataset == 'eveques':
    num_points = 1287  # N
    input_dim = 10  # D
    hidden1_dim = 64
    hidden2_dim = 16  # 10logK # P
    num_clusters = 18  # K

    use_edges = True
    nb_of_edges = 5

    num_epoch = 1000  # 100
    learning_rate = 2e-3  # 2e-3 (before) or 5e-3 (B: delta<0.4)
    pre_lr = 1e-3  # 0.01 (before)  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 70  # 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50

elif dataset == 'cora':
    num_points = 2708  # N (2708 in cora)
    input_dim = 1433  # D (1433 in cora)
    hidden1_dim = 64
    hidden2_dim = 16  # P
    num_clusters = 7  # K (7 in cora)

    num_epoch = 100  # 100
    learning_rate = 2e-3  # 2e-3 or 5e-3 (B: delta<0.4)
    pre_lr = 0.005  # 0.005 (Cora)  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 100  # 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50

# num_words = 761  # 1034
# hidden3_dim = 64  # P
# nb_of_topics = 3  # 3
# use_feature = True


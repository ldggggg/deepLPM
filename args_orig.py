### CONFIGS ###
dataset = 'simu'  # 'cora'
model = 'deepLPM'

num_points = 600  # N (2708 in cora)
input_dim = 600  # D (1433 in cora)
hidden1_dim = 64
hidden2_dim = 2  # P
num_clusters = 3  # K (7 in cora)
use_feature = True

num_epoch = 200
learning_rate = 1e-5
pre_lr = 0.2
pre_epoch = 100
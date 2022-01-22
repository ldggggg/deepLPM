import numpy as np
import networkx as nx
import pylab as pl
import pandas as pd

#%%
# A = np.loadtxt('adj_simuB_3clusters.txt').astype(np.int32)
# labels = np.loadtxt('label_simuB_3clusters.txt').astype(np.int32)
A = pd.read_csv('data/eveques_new/ResoEvequesClean2021-A.csv', header=0, sep=';').to_numpy()
labels = np.loadtxt('cl_k=9_p=16_TrueTrue.txt').astype(np.int32)

G = nx.from_numpy_array(A) # create graph
# add clustering assignment as nodes features to automatically get 
# positions depending on cluster assignments
for node in G.nodes():
    G.nodes[node]['block'] = labels[node]
pl.figure(figsize=(5,5))
scale_pos = 10. # attraction force by clusters for nodes position
seed=0 #fix seed for random nodes position
# x = np.loadtxt('x.txt')
# y = np.loadtxt('y.txt')
pos = np.loadtxt('pos_k=9_p=16_TrueTrue.txt')
pos1 = dict()
for i in range(1287):
    pos1[i] = pos[i]  # np.array([x[i],y[i]])  # pos[i]
# pos = nx.spring_layout(G,scale=scale_pos, seed=seed)
#pick edges features to draw like 
alpha_edge=0.1
width_edge=1.
edge_color ='grey'
nx.draw_networkx_edges(G,pos1,width=width_edge,alpha=alpha_edge,edge_color=edge_color)
#draw clusters with different colors
#chosen drawing features for nodes
node_size = 30
alpha_node = 1.

dict_color = {0:'#7294d4',1:'#fdc765', 2:'#869f82', 3:'lightblue', 4:'lightgreen', 5:'pink', 6:'blue',
              7:'purple', 8:'red', 9:'grey', 10:'gold3', 11:'orange', 12:'green', 13:'darkkhaki', 14:'darksalmon'}
for cluster in np.unique(labels):
    # cluster color chosen as default color C0, C1,C2
    nodelist = np.argwhere(labels==cluster)[:,0]
    nx.draw_networkx_nodes(G,pos1,nodelist=nodelist,node_size=node_size,alpha=alpha_node,node_color=dict_color[cluster])
pl.axis('off')
pl.tight_layout()
pl.title('Embedding_A_X_Y')
pl.show()    

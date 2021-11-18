import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import args
from test import *
adj, labels = create_simu(args.num_points, args.num_clusters, args.hidden2_dim)

# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')
utils.chooseCRANmirror(ind=1)

v = importr('VBLPCM')
adj = robjects.r.matrix(adj)
fit = v.vblpcmfit(v.vblpcmstart(adj, G=3, d=2))
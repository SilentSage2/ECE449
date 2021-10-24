import torch
import hw4_utils


def k_means(X=None, init_c=None, n_iters=3):
    """K-Means.
    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a cluster center.
    
    Return:
        c: shape [2, 2]. Each column is a cluster center.
    """

    # loading data and intiailzation of the cluster centers
    if X is None:
        X, c = hw4_utils.load_data()

    # your code below

    for k in range(n_iters):
        # first solve the assignment problem given the centers c
        
        # then solve the cluster center problem given the assignments
        
        # visulize the current clustering using hw4_utils.vis_cluster. 
        # with n_iters=3, there will be 3 figures. Put those figures in your written report. 
        pass
    
    return c

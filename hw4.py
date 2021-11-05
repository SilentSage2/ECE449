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
    else:
        c = init_c

    # your code below

    for k in range(n_iters):
        # first solve the assignment problem given the centers c
        
        # then solve the cluster center problem given the assignments
        
        # visulize the current clustering using hw4_utils.vis_cluster. 
        # with n_iters=3, there will be 3 figures. Put those figures in your written report. 
        N = X.size()[1]
        labels = []
        for i in range(N):
            label = classifier(c,X[:,i])
            labels.append(label)
        c1 = c[:,0].reshape(2,1) 
        c2 = c[:,1].reshape(2,1) 
        x1 = X[:,[i for i in range(len(labels)) if labels[i]==0]]
        x2 = X[:,[i for i in range(len(labels)) if labels[i]==1]]
        hw4_utils.vis_cluster(c1, x1, c2, x2)
        # pass
        c[0,0] = torch.mean(X[0,[i for i in range(len(labels)) if labels[i]==0]])
        c[1,0] = torch.mean(X[1,[i for i in range(len(labels)) if labels[i]==0]])
        c[0,1] = torch.mean(X[0,[i for i in range(len(labels)) if labels[i]==1]])
        c[1,1] = torch.mean(X[1,[i for i in range(len(labels)) if labels[i]==1]])
    
    return c
    
# help function
def classifier(c,point):
    return dist(c[:,0],point) < dist(c[:,1],point)
def dist(a, b):
    return torch.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

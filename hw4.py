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
    c  = init_c
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
        vis_cluster(c1, x1, c2, x2)
        # plt.plot(c[0,0],c[1,0],'r*')
        # plt.plot(c[0,1],c[1,1],'b*')
        # plt.plot(X[0, [i for i in range(len(labels)) if labels[i]==0]], X[1, [i for i in range(len(labels)) if labels[i]==0]], 'rx')
        # plt.plot(X[0, [i for i in range(len(labels)) if labels[i]==1]], X[1, [i for i in range(len(labels)) if labels[i]==1]], 'bo')
        # pass
        c[0,0] = torch.mean(X[0,[i for i in range(len(labels)) if labels[i]==0]])
        c[1,0] = torch.mean(X[1,[i for i in range(len(labels)) if labels[i]==0]])
        c[0,1] = torch.mean(X[0,[i for i in range(len(labels)) if labels[i]==1]])
        c[1,1] = torch.mean(X[1,[i for i in range(len(labels)) if labels[i]==1]])
    
    return c


def classifier(c,point):
    label = 0
    dis = dist(c[:,0],point)
    for i in range(c.size()[1]):
        if dis > dist(c[:,i],point):
            label = i
    return label
def dist(a, b):
    distance = torch.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return distance

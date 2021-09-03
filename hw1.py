import torch
import hw1_utils as utils


'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else (otherwise, your w will be ordered differently than the
    reference solution's in the autograder)!!!
'''

# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides!!!)
    '''
    N = X.shape[0]
    d = X.shape[1]
    w = torch.zeros(d+1, 1)
    X_new = torch.cat((torch.ones(N, 1), X), 1)
    for epoc in range(num_iter):
      grad_of_loss = tensor.matmul(X_new.T, tensor.matmul(X_new, w) - Y)
      w = w - lrate * grad_of_loss
    return w
    pass


def linear_normal(X, Y):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides!!!)
    '''
    N = X.shape[0]
    X_new = torch.cat((torch.ones(N, 1), X), 1)
    pseudoinverse = torch.matmul(torch.matmul(X_new, X_new).inverse, X_new.T)
    w = torch.matmul(pseudoinverse, Y)
    return w
    pass


def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X, Y = load_reg_data()
    N = X.shape[0]
    w = linear_normal(X, Y)
    X_new = torch.cat((torch.ones(N, 1), X), 1)
    y_hat = torch.matmul(X_new, w)
    plt.plot(X, y_hat)
    plt.scatter(X, Y)
    plt.show()
    pass


# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides) 
    '''
    pass


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass

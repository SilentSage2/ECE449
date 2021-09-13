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
        # gradient of R is: (1/N)*X.T*(Xw-Y)
        grad_of_loss =torch.matmul(X_new.T, torch.matmul(X_new, w) - Y) / N
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
    # pinverse: (X.T*X)^(-1)*X.T
    # pseudoinverse = torch.matmul(torch.inverse(torch.matmul(X_new.T, X_new)), X_new.T)
    pseudoinverse = torch.pinverse(X_new)
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

    plt.plot(X, y_hat,"r-")
    plt.scatter(X, Y)
    plt.xlabel("input X")
    plt.ylabel("output Y")
    plt.grid(linestyle='--')
    plt.legend(["Regression","Original"])
    plt.show()
    return plt.gcf()
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
    N = X.shape[0]
    d = X.shape[1]
    w = torch.zeros(d+1, 1)
    X_new = torch.cat((torch.ones(N, 1), X), 1)
    for epoc in range(num_iter):
        # gradient of R is: (1/N)*exp(-(Xw).T)*y)/(1+exp(-(Xw).T)*y))*(-X.T*y)
        grad_of_loss = logsitic_gradient(X_new, Y, w, N, d)
        w = w - lrate * grad_of_loss
    return w
    pass

def logsitic_gradient(X, Y, w, N, d):
    grad = torch.zeros(d+1,1)
    for i in range(N):
        exp_part = torch.exp((-1) * torch.matmul(X[i], w) * Y[i])
        grad += exp_part / (1 + exp_part) * (-1) * Y[i] * X[i].reshape(d+1,1)
    return grad / N 

def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    # X_reg, Y_reg = load_reg_data()
    # # print(X_reg.shape)
    # N = X_reg.shape[0]
    # X_reg_new = torch.cat((torch.ones(N, 1), X_reg), 1)
    # w_linear = linear_gd(X_reg, Y_reg)
    # y_hat_linear = torch.matmul(X_reg_new, w_linear)

    X, Y = load_logistic_data()

    x_begin = min(X.T[0])
    x_end = max(X.T[0])
    x = torch.linspace(x_begin, x_end, 1000)

    w_reg = linear_gd(X, Y)
    boundary_reg = -w_reg[1]/w_reg[2]*x-w_reg[0]/w_reg[2]

    w_log = logistic(X, Y)
    boundary_log = -w_log[1]/w_log[2]*x-w_log[0]/w_log[2]

    plt.plot(x, boundary_reg,"r-")
    plt.plot(x, boundary_log,"g-")
    plt.scatter(X.T[0], X.T[1])
    plt.xlabel("$X_{1}$")
    plt.ylabel("$X_{2}$")
    plt.grid(linestyle='--')
    plt.legend(["Linear","Logistic","Dataset"])
    plt.show()
    return plt.gcf()
    pass


import numpy as np
from scipy.stats import sem
from collections import deque
from Algorithm.PerGD_function import clip,beta,loss,performative_loss,grad1,approx_beta,approx_grad_beta,grad2
from sklearn.linear_model import LogisticRegression, Ridge

class PerGD:
    def __init__(self,H = 100, lr = 0.01):
        # self.s1 = 0.5
        self.H  = H
        self.lr = lr # learning rate
        # self.history = deque()
        # self.history.append(self.theta.copy())
        self.grad_fs = deque()
        # self.g2s     = deque()
        self.thetas = deque(maxlen = H + 1)
        self.means  = deque(maxlen = H + 1)
        self.betas  = deque(maxlen = H + 1)

    def train(self,X,y_ture):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        d = X_b.shape[1]
        Y = np.copy(y_ture)
        if len(self.thetas) == 0:
            self.theta = np.random.rand(d)
            self.thetas.append(self.theta.copy())
            self.betas.append(approx_beta(X, Y))
            grad = grad1(X_b, Y, self.theta)
            self.theta = clip(self.theta - self.lr * grad)
            self.thetas.append(self.theta.copy())
            self.betas.append(approx_beta(X_b, Y))
        else:
            g2, grad_beta = grad2(X_b, Y, self.betas, self.thetas)
            grad = grad1(X_b, Y, self.theta) + g2
            self.theta = clip(self.theta - self.lr * grad)

            self.thetas.append(self.theta)
            self.betas.append(approx_beta(X_b, Y))
        self.coef_ = self.theta[1:].reshape(-1,1).T
            
    def predict(self,X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        predictions = np.dot(X_b, self.theta)
        return predictions.reshape(-1, 1)
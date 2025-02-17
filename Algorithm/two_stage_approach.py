import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import time
import torch

class two_stage_algo:
    def __init__(self,X_base,y_base,tol = 1e-6):
        self.X_base = np.copy(np.c_[np.ones((X_base.shape[0], 1)), X_base] )
        self.y_base = np.copy(y_base)
        self.coef_ = None
        self.theta = None
        self.tol = tol
        self.mu_x = np.zeros((X_base.shape[1]+1,X_base.shape[1]+1))
        self.mu_y = np.zeros((X_base.shape[1]+1,1))
        self.X_shift = None 
        self.y_shift = None
        self.theta_list = np.zeros_like(np.c_[np.ones((X_base.shape[0], 1)), X_base])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def calculate_performative_effect(self):
        n = len(self.y_base)
        repeat_theta = self.theta_list

        theta_t_theta = repeat_theta.T @ repeat_theta
        theta_theta_inv = np.linalg.inv(theta_t_theta + 0.01*np.eye(theta_t_theta.shape[0]))
        self.mu_x = theta_theta_inv @ repeat_theta.T @ self.X_shift
        self.mu_y = theta_theta_inv @ repeat_theta.T @ self.y_shift 

    def train(self,X,y_ture):
        y = np.copy(y_ture)
        n = len(y)
        X = np.c_[np.ones((X.shape[0], 1)), X] 
        
        if self.X_shift is None:
            self.X_shift = X
            self.y_shift = y
        else:
            self.X_shift = np.concatenate((self.X_shift, X), axis=0)
            self.y_shift = np.concatenate((self.y_shift, y), axis=0)

        self.calculate_performative_effect()

        theta_final, cost_history = self.gradient_descent(self.X_base, self.y_base)
        self.theta = np.copy(theta_final) 

        self.coef_ = np.copy(theta_final[1:])[:, np.newaxis].T
        self.theta_list = np.concatenate((self.theta_list, np.tile(theta_final,(n, 1))), axis=0)
        return theta_final

    def predict(self,X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] 
        score = np.dot(X_b, self.theta)
        return score.reshape(-1, 1)

    def compute_cost(self,X, y, theta):
        m = len(y)
        mu_y = self.mu_y.cpu().numpy()
        mu_x = self.mu_x.cpu().numpy()
        theta_mu_y = np.repeat(theta @ mu_y, m, axis=0).reshape(-1, 1)
        theta_mu_x = np.tile(theta @ mu_x, (m, 1))
        cost = np.linalg.norm(y+theta_mu_y - ((X + theta_mu_x) @ theta).reshape(-1, 1))
        return cost

    def gradient_descent(self,X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.mu_x = torch.tensor(self.mu_x, dtype=torch.float32).to(self.device)
        self.mu_y = torch.tensor(self.mu_y, dtype=torch.float32).to(self.device)

        m = len(y)
        cost_history = []
        d = X.shape[1]
        theta = torch.randn(d, dtype=torch.float32, device=self.device)
        learning_rate = 0.1
        num_iterations = 1000
        max_grad_norm = d
        clip_value = 1

        XTY = (X.T@y).reshape(-1)
        y_muy = torch.sum(y[:, None] * self.mu_y, axis=0)
        for j in range(num_iterations):
            gradient = (y_muy+ theta @self.mu_y * self.mu_y * m -torch.sum(X* theta, axis=0).reshape(-1, 1) * self.mu_y \
                        + self.mu_x.mT@theta@theta*self.mu_y * m).reshape(-1)\
                        - XTY - torch.sum(theta @ self.mu_y*X, axis=0) + X @ theta @ X - torch.sum(self.mu_x.mT@theta@theta*X, axis=0)\
                        + torch.sum(self.mu_x.mT @ theta * y, axis=0) + theta @ self.mu_y*self.mu_x.mT @ theta *m \
                        - torch.sum(theta*self.mu_x.mT @ theta *X, axis=0) + self.mu_x.mT@theta@theta*self.mu_x.mT@theta * m\
                        + theta
            
            gradient = 2 * gradient / m
            grad_norm = torch.norm(gradient)
            if grad_norm > max_grad_norm:
                gradient = (gradient / grad_norm) * max_grad_norm

            theta -= learning_rate * gradient
            theta.clamp_(-clip_value*0.5, clip_value)
            cost = self.compute_cost(X.cpu().numpy(), y.cpu().numpy(), theta.cpu().numpy())
            cost_history.append(cost)
            if (j > 2) and (abs(cost_history[-1] - cost_history[-2]) < 1e-3):
                self.mu_x = self.mu_x.cpu().numpy()
                self.mu_y = self.mu_y.cpu().numpy()
                return theta.cpu().numpy(), cost_history
        self.mu_x = self.mu_x.cpu().numpy()
        self.mu_y = self.mu_y.cpu().numpy()
        return theta.cpu().numpy(), cost_history

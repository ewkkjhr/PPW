from collections import OrderedDict
import numpy as np
import math
import sys
from sklearn.linear_model import LogisticRegression, Ridge

class DFO_GD:
    def __init__(self,forgetting_factor = 0.5):
        self.iter = 0
        self.config = config
        self.metric = {'iter': [], 'gap': []}
        self.output_path = './res-new/'
        
        self.rho = 0.5
        self.forgetting_factor = forgetting_factor # corresponding to lambda in paper

        if self.forgetting_factor == 1:
            self.tau0 = 2/np.log(1/max(self.rho, self.forgetting_factor + 1e-8)) # 这里的1e-6可以防止出现log(0)的情况
        else:
            self.tau0 = 2/np.log(1/max(self.rho, self.forgetting_factor)) 
        self.outer_iter, self.inner_iter, self.sample_count = 0, 0, 0

        self.flag = False
        self.flag2 = True
    
    def train(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        d = X_b.shape[1]
        n = X_b.shape[0]

        if self.outer_iter == 0 and self.inner_iter == 0:
            # step size 相关的参数
            self.delta0 = d ** (1/6) * 100 #100 # 这里的50非常重要，决定了收敛的CPU time
            self.beta = 1/6
            self.eta0 = d ** (-2/3) * 0.3 #0.3
            self.alpha = -2/3
            self.theta, self.sample_z = self.initialization(X_b,y)
            self.pert_theta = np.copy(self.theta)
            self.coef_ = self.theta[1:].reshape(-1,1).T
        
        if self.inner_iter == 0:
            temp = self.tau0 * np.log(self.outer_iter + 1)+2
            self.tau_k = max(1, int(temp))
            self.new_uk = self.sample_unit_sphere(d) # direction
            self.delta_k = self.step_size('delta')
            
        if self.inner_iter < self.tau_k:
            self.sample_count += 1

            grd = d / self.delta_k * ((self.ell_loss(self.pert_theta, X_b,y)/n)/d) * (self.new_uk )
            # update theta
            rate = 0.01
            lr = (self.forgetting_factor ** (((self.tau_k - self.inner_iter))))
            self.theta = np.clip(self.theta - self.step_size('eta') * lr * grd * 0.1,-0.05,1) 
            self.pert_theta = self.theta + self.delta_k * self.new_uk * 0.02 
            self.inner_iter += 1

        if self.inner_iter == self.tau_k:
            self.inner_iter = 0
            self.outer_iter += 1
            self.pert_theta = self.theta
        
        self.coef_ = self.pert_theta[1:].reshape(-1,1).T


    def ell_loss(self, pert_theta, X,y):
        return np.linalg.norm(np.dot(X, pert_theta) - y) ** 2

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        predictions = np.dot(X_b, self.pert_theta)
        return predictions.reshape(-1, 1)

    def sample_unit_sphere(self,d):
        s = np.random.normal(0, 1, d)
        norm = math.sqrt(sum(s * s))
        return s / norm

    def initialization(self,X_b,y):
        d = X_b.shape[1]
        LR = Ridge(alpha = 1, fit_intercept=False)
        LR.fit(X_b, y)
        init_theta = LR.coef_.reshape(-1)
        sample_z = np.zeros((config.batch, d))
        return init_theta, sample_z
    
    def step_size(self, name='delta'):
        if name == 'delta':
            stepsize = self.delta0 / ((self.outer_iter+1) ** self.beta)
        elif name == 'eta':
            stepsize = self.eta0 * ((1+self.outer_iter) ** self.alpha)
        else:
            print('输入参数错误')
            sys.exit(1)
        return stepsize

class AlgoConfig(object):
    """定义一个关于算法的默认设置类"""

    def __init__(self):
        self.log_scale = True
        self.num_points = 2000 # 如果使用对数采样，图中的点数是self.num_points

        self.max_iter_log = 7
        self.max_iter_num = 10 ** int(self.max_iter_log)
        self.step = 300 # 隔self.step采样一次
        self.batch = 1

config = AlgoConfig()
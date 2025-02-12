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
        # self.sample_time = self.create_sampling_time()
        # self.max_record_time = self.sample_time[-1]
        
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
            # self.inner_iter = 1
            
        if self.inner_iter < self.tau_k:
            # self.new_uk = self.sample_unit_sphere(d) # direction
            # new_sample = self.problem.sample_from_AR(self.pert_theta)
            # new_sample = self.problem.sample_from_stationary_dist(pert_theta)
            self.sample_count += 1

            grd = d / self.delta_k * ((self.ell_loss(self.pert_theta, X_b,y)/n)/d) * (self.new_uk )
            # grd = self.problem.dim / delta_k * self.problem.expect_loss(pert_theta) * uk  #如果用真正的grd是可以收敛的

            # update theta
            rate = 0.01
            lr = (self.forgetting_factor ** (((self.tau_k - self.inner_iter))))
            self.theta = np.clip(self.theta - self.step_size('eta') * lr * grd * 0.1,-0.05,1) 
            self.pert_theta = self.theta + self.delta_k * self.new_uk * 0.02 #np.clip(self.theta + self.delta_k * self.new_uk*0.01,0,1)
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
        # init_theta = np.random.rand(d)
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
    # def fit_innner_loop(self, tau_k, uk, delta_k, bar):

    #     pert_theta = self.theta + delta_k * uk
    #     new_sample = self.problem.sample_from_AR(pert_theta)
    #     # new_sample = self.problem.sample_from_stationary_dist(pert_theta)
    #     self.sample_count += 1


    #     grd = self.problem.dim / delta_k * self.problem.ell_loss(pert_theta, new_sample) * uk
    #     # grd = self.problem.dim / delta_k * self.problem.expect_loss(pert_theta) * uk  #如果用真正的grd是可以收敛的

    #     # update theta
    #     self.theta = self.theta - self.step_size('eta') * (self.forgetting_factor ** (tau_k - self.inner_iter)) * grd

    #     if np.linalg.norm(self.theta) >= 1e8: # for debug use
    #         print('Error! There are some issue in programming')
    #         exit(0)

    #     if self.sample_time != [] and self.sample_count == self.sample_time[0]:
    #         self.sample_time.pop(0)
    #         self.record()
    #         # if self.rank == 0:
    #         #     self.info_bar(err=self.metric['gap'][-1])
    #         if self.sample_count == self.max_record_time:
    #             self.flag = True

    #     bar.update(1)
    #     if self.rank == 0:
    #         bar.set_description(f'rho = {self.forgetting_factor}, '
    #                             f'Current Sample Num {self.sample_count}, gap {self.metric["gap"][-1]:.5f}')

    # def create_sampling_time(self):
    #     """生成对数刻度或者正常刻度,sample_num记录metric运行的时间点"""
    #     if self.config.log_scale:
    #         L = np.logspace(0, self.config.max_iter_log, self.config.num_points, endpoint=False,
    #                         dtype=int).tolist()  # L stores the time point when we sample
    #         sample_num = list(OrderedDict.fromkeys(L))  # 去掉L中重复的元素(note: L中的元素都是non-decreasing的)
    #     else:
    #         sample_num = list(range(0, self.config.max_iter_num, self.config.step))  # 选取测算measurement的时间点
    #         # 这里原本时用range(0, max_iter)的，但是由于DFO执行时while-loop, self.iter=0的时候会出错
    #     return sample_num
    
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
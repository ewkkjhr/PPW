import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import numpy as np
from functions import est_varepsilon,data_distribution_map1,data_distribution_map2,remove_outliers_iqr,linear_data_generation
from datetime import datetime
import random
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import copy
from sklearn.linear_model import Ridge

# # problems parameters
# num_iters = 25
# d_list = [10,1000,10000]

class LinearRegression_one_iter_gd:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.theta = None
        self.coef_ = None  # 参数，包括截距项

    def fit(self, X, y,model = None):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在X的第一列添加1
        m = X_b.shape[0]  # 样本数量
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if model is None and self.theta is None:
            # LR = Ridge(alpha = 0, fit_intercept=False)
            # LR.fit(X_b, y)
            # self.theta = LR.theta.T.copy() #+ np.random.normal(0, 0.05, size=RR.theta.T.shape) #np.random.randn(X_b.shape[1], 1)  
            self.theta = np.random.randn(X_b.shape[1], 1) # 随机初始化参数
        gradients = (2 / m) * X_b.T.dot(X_b.dot(self.theta) - y)
        threshold = 100  # 设置阈值
        norm = np.linalg.norm(gradients)
        if norm > threshold:
            gradients = gradients * (threshold / norm)

        self.theta -= self.learning_rate * gradients  
        self.coef_ = self.theta[1:].T  # 参数，不包括截距项

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加截距项
        return X_b.dot(self.theta)  # 预测值
    

def RGD(X,y,num_iters = 25,d_list = [10,1000,10000],map = 1,strat_features = np.array([1, 6, 8])-1,\
        num_experiments = 10,seed_value = 42):

    method_name = 'RGD_Linear_Regression'
    num_d  = len(d_list)
    n = X.shape[0]
    d = X.shape[1]

    print('RGD Linear Regression')
    num_d  = len(d_list)
    n = X.shape[0]
    d = X.shape[1]
    RR_int = LinearRegression_one_iter_gd()
    RR_int.fit(X, y)

    model_gaps         = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    mse_list_start     = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    mse_list_end       = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    

    for i in range(num_experiments):
        print('  Running {} th times'.format(i))
        np.random.seed(seed_value + i)
        random.seed(seed_value  + i)
        for k, d in enumerate(d_list):
            print('    Running epsilon =  {}'.format(d))
            RR = copy.deepcopy(RR_int)

            for t in range(num_iters):
                # adjust distribution to current theta
                if map == 1:
                    X,y = linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map1(X, y,mu = d, model = RR, strat_features = strat_features)
                    # X_strat = preprocess_data_shift(X_strat, X, strat_features, n)
                
                if map == 2:
                    X_strat,y_strat = data_distribution_map2(X, y,mu = d, model = RR)

                # evaluate initial loss on the current distribution
                pred_label_old = RR.predict(X_strat)
                # mse = mean_squared_error(y_strat, pred_label_old)
                mse = np.sqrt(mean_absolute_error(y_strat, pred_label_old))
                mse_list_start[i,k,t] = mse

                # learn on induced distribution
                theta_old = RR.coef_.copy()
                RR.fit(X_strat, y_strat)
                theta_new = RR.coef_.copy()
                model_gaps[i,k,t] = np.linalg.norm(theta_new-theta_old)

                # evaluate final loss on the current distribution
                pred_label = RR.predict(X_strat)
                # mse = mean_squared_error(y_strat, pred_label)
                mse = np.sqrt(mean_absolute_error(y_strat, pred_label))
                mse_list_end[i,k,t] = mse

        print('-'*50)

    for k, d in enumerate(d_list):
        model_gaps_avg = np.mean(model_gaps, axis=0)
        model_gaps_std = np.std(model_gaps, axis=0)
        mse_list_start_avg = np.mean(mse_list_start, axis=0)
        mse_list_start_std = np.std(mse_list_start, axis=0)
        mse_list_end_avg = np.mean(mse_list_end, axis=0)
        mse_list_end_std = np.std(mse_list_end, axis=0)

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Completion Time:", current_time_str)

    return model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name

import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from functions import est_varepsilon,data_distribution_map1,data_distribution_map2,remove_outliers_iqr,linear_data_generation
from datetime import datetime
import random
from Algorithm.PerGD_alg import PerGD

def PerformativeGD(X,y,num_iters = 25,d_list = [10,1000,10000],map = 1,strat_features = np.array([1, 6, 8])-1,\
        num_experiments = 10,seed_value = 42):
    
    method_name = 'PerGD'
    num_d  = len(d_list)
    n = X.shape[0]
    d = X.shape[1]

    print('PerGD:')
    model_gaps         = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    mse_list_start     = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    mse_list_end       = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]

    for i in range(num_experiments):
        print('  Running {} th times'.format(i))
        np.random.seed(seed_value + i)
        random.seed(seed_value  + i)
        for k, d in enumerate(d_list):
            # initial model
            print('    Running epsilon =  {}'.format(d))
            model = PerGD()
            model.train(X, y)
            theta = np.copy(model.coef_)
            
            for t in range(num_iters):
                # adjust distribution to current theta
                if map == 1:
                    X,y = linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map1(X, y,mu = d, model = model, strat_features = strat_features)
                    
                if map == 2:
                    X_strat,y_strat = data_distribution_map2(X, y,mu = d, model = model)

                # evaluate initial loss on the current distribution
                pred_label = model.predict(X_strat)
                mse = np.sqrt(mean_squared_error(y_strat, pred_label))
                mse_list_start[i,k,t] = mse
                
                # learn on induced distribution
                model.train(X_strat, y_strat)
                theta_new = np.copy(model.coef_)
                if np.linalg.norm(theta_new) == 0:
                    theta_new = theta_new + 1e-5
                if np.linalg.norm(theta) == 0:
                    theta = theta + 1e-5
                model_gaps[i,k,t] = np.linalg.norm(theta_new-theta)
                theta = np.copy(theta_new)

                # evaluate final loss on the current distribution
                pred_label_new = model.predict(X_strat)
                mse = np.sqrt(mean_squared_error(y_strat, pred_label_new))
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

import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import numpy as np
from sklearn.metrics import mean_squared_error
from functions import est_varepsilon,data_distribution_map1,data_distribution_map2,remove_outliers_iqr,linear_data_generation
from datetime import datetime
import random
from Algorithm.PPNN import PerPreNN
import copy

def PPNN(X,y,num_iters = 25,d_list = [10,1000,10000],map = 1,strat_features = np.array([1, 6, 8])-1,\
        num_experiments = 10,seed_value = 42):
    
    method_name = 'RRM with Neural Networks'
    num_d  = len(d_list)
    n = X.shape[0]
    d = X.shape[1]
    
    print('Performative Prediction with Neural Network:')
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
            model = PerPreNN()
            model.train(X, y)
            theta = copy.deepcopy(model.theta)
            
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
                model = PerPreNN()
                model.train(X_strat, y_strat)
                gap = 0
                theta_new = copy.deepcopy(model.theta)
                theta_full_new = np.concatenate([matrix.flatten() for matrix in theta_new])
                theta_full = np.concatenate([matrix.flatten() for matrix in theta])
                model_gaps[i,k,t] = np.linalg.norm(theta_full_new-theta_full)
                theta = copy.deepcopy(theta_new)

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

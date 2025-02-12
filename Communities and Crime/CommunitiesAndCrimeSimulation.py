import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from Algorithm.plot import plot_fig
from Algorithm.alg_method_1 import method_1
from Algorithm.alg_RRM import RRM
from Algorithm.alg_RGD import RGD
from Algorithm.alg_PPNN import PPNN
from Algorithm.alg_Outside import TSA
from Algorithm.alg_DFO import DFO
from Algorithm.alg_PerGD import PerformativeGD

# problems parameters
seed_value = 42
num_iters = 100
d_list = [2,4,8,16] #2,4,6,8,
num_experiments = 10
map = 2
np.random.seed(seed_value)
random.seed(seed_value)
folder_path = 'Communities and Crime/result/'

initial=pd.read_csv('Communities and Crime/communities-crime-clean.csv')
initial = initial.drop('communityname', axis=1)
initial = initial.drop('fold', axis=1)
initial = initial.drop('state', axis=1)
y = initial['ViolentCrimesPerPop'].values.reshape(-1, 1)
# scaler = StandardScaler()
# y = scaler.fit_transform(y)
initial = initial.drop('ViolentCrimesPerPop', axis=1)
X = initial.values

methods = ['RRM Linear Regression','RGD Linear Regression','Two-Stage Approach','PerfGD','RRM Neural Networks','DFO','PPW']

# method 1
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    method_1(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
print(file_name_npy)
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# RRM
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    RRM(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# RGD
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    RGD(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# ppnn
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    PPNN(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# outside the echo chamber
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    TSA(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# PerformativeGD
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    PerformativeGD(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# DFO (Two-timescale Derivative Free Optimization for Performative Prediction with Markovian Data)
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    DFO(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

plot_fig(num_iters,d_list,folder_path,methods)

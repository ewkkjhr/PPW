
import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
import pandas as pd
from Algorithm.alg_method_1 import method_1
from Algorithm.plot_par_sen import plot_fig_par
import warnings

warnings.filterwarnings("ignore")
# problems parameters
seed_value = 42
num_iters = 100
d_list = [2,4,8,16]
num_experiments = 10
map = 2
np.random.seed(seed_value)
random.seed(seed_value)
folder_path = 'Communities and Crime/par_result/'

initial=pd.read_csv('Communities and Crime/communities-crime-clean.csv')
initial = initial.drop('communityname', axis=1)
initial = initial.drop('fold', axis=1)
initial = initial.drop('state', axis=1)
y = initial['ViolentCrimesPerPop'].values.reshape(-1, 1)
initial = initial.drop('ViolentCrimesPerPop', axis=1)
X = initial.values
alphas = [2.1,4,8,16]

for alpha in alphas:
    print('alpha = ',alpha)
    # method 1
    model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    method_1(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value,alpha_1 = alpha)
    file_name_npy = f"{folder_path}{alpha}.npz"
    np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
                mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
                mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
    print(f"Data saved to {file_name_npy}")

plot_fig_par(num_iters,d_list,folder_path,alphas)
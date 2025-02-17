import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import os
import numpy as np
from functions import plot_model_gap,plot_step,plot_mse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import pandas as pd

# problems parameters
num_iters = 100
d_list = [10,1000,10000]
def format_number(number):
    formatted_number = f"{number:.3e}"
    coefficient, exponent = formatted_number.split('e')
    coefficient = str(float(coefficient))
    exponent = exponent.lstrip('+0') if exponent[0] != '-' else '-' + exponent[1:].lstrip('0')
    formatted_string = f"{coefficient} \\times 10^{{{exponent}}}"
    return formatted_string

def plot_fig(num_iters = 25,d_list = [10,1000,10000],folder_path = 'result/',methods = []):
    num_d  = len(d_list)
    data_dict = {}

    for methods_name in methods:
        data = np.load(folder_path+methods_name + '.npz')
    
        inner_dict = {key: data[key] for key in data.files}
        data_dict[methods_name] = inner_dict
    
        data.close()

    if 'DFO' in data_dict:
        data_dict[r'$' +'\mathrm{DFO(\lambda)}'+ r'$'] = data_dict['DFO']
        del data_dict['DFO']

    methods = methods[:-2] + [r'$' +'\mathrm{DFO(\lambda)}'+ r'$'] + [methods[-1]]

    if len(methods) == 7:
        colors = ['y', 'g', 'r', 'c', 'k', 'm', 'b']
        markers = ['*', 'D', '^', 's', 'p','v', 'o']
        linestyles = ['-.', '--', '-.', ':', '--','-', '-']
    else:
        colors = ['y', 'g', 'r', 'c', 'm', 'b']
        markers = ['*', 'D', '^', 's', 'v', 'o']
        linestyles = ['-.', '--', '-.', ':', '-', '-']


    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        ax = fig.gca()
        offset = 0.8
        max_element = 0
        min_element = 100
        for i,(file_key, inner_dict) in enumerate(data_dict.items()):
            for j in range(1,num_iters):
                mse_list_start_avg = inner_dict.get('mse_list_start_avg')
                mse_list_end_avg = inner_dict.get('mse_list_end_avg')
                plot_step(j, offset, mse_list_start_avg, mse_list_end_avg, file_key,colors[i],markers[i],num_iters,c)
                arrays = [mse_list_start_avg,mse_list_end_avg]
                max_element = max(max_element,np.max(np.maximum.reduce(arrays)))
                min_element = min(min_element,np.min(np.minimum.reduce(arrays)))
        plt.xlabel('Iteration', fontsize = 20)
        plt.ylabel('RMSE', fontsize = 20)
        plt.tick_params(labelsize=18)
        plt.yscale('log')
        plt.legend(loc='upper right',fontsize = 24)
        file_name = f'mse_d = {d_list[c]}.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')
    
    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        max_element = 0
        min_element = 100
        for i, methods_name in enumerate(methods):
            inner_dict = data_dict[methods_name]
            mse_list_start_avg = inner_dict.get('mse_list_start_avg')
            mse_list_start_std = inner_dict.get('mse_list_start_std')
            plot_mse(mse_list_start_avg[c],mse_list_start_std[c],colors[i],markers[i],linestyles[i],methods_name)
            max_element = max(max_element,np.max(np.maximum.reduce(mse_list_start_avg[c])))
            min_element = min(min_element,np.min(np.minimum.reduce(mse_list_start_avg[c])))
        plt.xlabel('Iteration', fontsize = 20)
        plt.ylabel('RMSE', fontsize = 20)
        plt.tick_params(labelsize=18)
        plt.yscale('log')
        plt.legend(loc='upper right',fontsize = 24)
        file_name = f'mse_d = {d_list[c]}_start.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        max_element = 0
        min_element = 100
        for i, methods_name in enumerate(methods):
            inner_dict = data_dict[methods_name]
            mse_list_start_avg = inner_dict.get('mse_list_start_avg')
            mse_list_start_std = inner_dict.get('mse_list_start_std')
            plot_mse(mse_list_start_avg[c],mse_list_start_std[c],colors[i],markers[i],linestyles[i],methods_name,std = 2)
            max_element = max(max_element,np.max(np.maximum.reduce(mse_list_start_avg[c])))
            min_element = min(min_element,np.min(np.minimum.reduce(mse_list_start_avg[c])))
        plt.xlabel('Iteration', fontsize = 20)
        plt.ylabel('RMSE', fontsize = 20)
        plt.tick_params(labelsize=18)
        plt.yscale('log')
        plt.legend(loc='upper right',fontsize = 24)
        file_name = f'mse_d = {d_list[c]}_start_no_std.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        for i, methods_name in enumerate(methods):
            inner_dict = data_dict[methods_name]
            model_gaps_avg = inner_dict.get('model_gaps_avg')
            model_gaps_std = inner_dict.get('model_gaps_std')
            plot_model_gap(model_gaps_avg[c],model_gaps_std[c],colors[i],markers[i],linestyles[i],methods_name)
        plt.xlabel('Iteration', fontsize = 20)
        plt.ylabel(r'$\|\theta_t - \theta_{t-1}\|$', fontsize = 20)
        plt.tick_params(labelsize=18)
        plt.legend(loc='upper right',fontsize = 24)
        plt.yscale('log')
        file_name = f'Model_gap_d = {d_list[c]}.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        for i, methods_name in enumerate(methods):
            inner_dict = data_dict[methods_name]
            model_gaps_avg = inner_dict.get('model_gaps_avg')
            model_gaps_std = inner_dict.get('model_gaps_std')
            plot_model_gap(model_gaps_avg[c],model_gaps_std[c],colors[i],markers[i],linestyles[i],methods_name,std = 2)
        plt.xlabel('Iteration', fontsize = 20)
        plt.ylabel(r'$\|\theta_t - \theta_{t-1}\|$', fontsize = 20)
        plt.tick_params(labelsize=18)
        plt.legend(loc='upper right',fontsize = 24)
        plt.yscale('log')
        file_name = f'Model_gap_d = {d_list[c]}_no_std.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    plt.close('all')
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Plot Completion Time:", current_time_str)

    df = pd.DataFrame()
    descriptions = []
    for i in range(num_d):
        descriptions.append(f'd = {d_list[i]}')
    for i, methods_name in enumerate(methods):
        inner_dict = data_dict[methods_name]
        model_gaps_avg = inner_dict.get('model_gaps_avg')
        means = np.mean(model_gaps_avg[:,5:], axis=1)
        stds = np.std(model_gaps_avg[:,5:], axis=1)
        data = [f"{np.round(mean, decimals=3)} $\\pm$ {np.round(std, decimals=3)}" for mean, std in zip(means, stds)]
        df[methods_name] = data
    df.insert(0, ' ', descriptions)
    df_1 = df.T
    df_1.columns = df_1.iloc[0]  
    df_1 = df_1.drop(df_1.index[0])  
    df_1.to_csv(folder_path+'gap_mean_std_after_5.csv')

    df = pd.DataFrame()
    descriptions = []
    for i in range(num_d):
        descriptions.append(f'd = {d_list[i]}')
    for i, methods_name in enumerate(methods):
        inner_dict = data_dict[methods_name]
        mse_list_start_avg = inner_dict.get('mse_list_start_avg')
        means = np.mean(mse_list_start_avg[:,5:], axis=1)
        stds = np.std(mse_list_start_avg[:,5:], axis=1)
        data = [f"{np.round(mean, decimals=3)} $\\pm$ {np.round(std, decimals=3)}" for mean, std in zip(means, stds)]
        df[methods_name] = data
    df.insert(0, ' ', descriptions)

    df_1 = df.T
    df_1.columns = df_1.iloc[0]  
    df_1 = df_1.drop(df_1.index[0])  
    df_1.to_csv(folder_path+'mse_mean_std_after_5.csv') 

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Table Completion Time:", current_time_str)

if __name__ == "__main__":
    plot_fig(num_iters,d_list)

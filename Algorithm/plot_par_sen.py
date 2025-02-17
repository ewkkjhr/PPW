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

fontsize1 = 16
fontsize2 = 14
fontsize3 = 16

def plot_fig_par(num_iters = 25,d_list = [10,1000,10000],folder_path = 'result/',alphas=[],latex_text = r"$\alpha$"):
    num_d  = len(d_list)
    data_dict = {}

    for alpha in alphas:
        data = np.load(folder_path+f'{alpha}' + '.npz')
    
        inner_dict = {key: data[key] for key in data.files}
        data_dict[alpha] = inner_dict
    
        data.close()

    linewidth = 2
    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y', 'orange','purple']
    markers = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',' ']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':','-']

    for c in range(num_d):
        fig = plt.figure(figsize=(7.5,3.5))
        ax = fig.gca()
        offset = 0.8
        max_element = 0
        min_element = 100
        for i,(file_key, inner_dict) in enumerate(data_dict.items()):
            for j in range(1,num_iters):
                mse_list_start_avg = inner_dict.get('mse_list_start_avg')
                mse_list_end_avg = inner_dict.get('mse_list_end_avg')
                plot_step(j, offset, mse_list_start_avg, mse_list_end_avg, f'{latex_text} = {file_key}',colors[i],markers[i],num_iters,c,linewidth = linewidth)
                arrays = [mse_list_start_avg,mse_list_end_avg]
                max_element = max(max_element,np.max(np.maximum.reduce(arrays)))
                min_element = min(min_element,np.min(np.minimum.reduce(arrays)))
        plt.xlabel('Iteration', fontsize = fontsize1)
        plt.ylabel('RMSE', fontsize = fontsize1)
        plt.tick_params(labelsize=fontsize2)
        plt.ylim(0.18, 1.4) 
        plt.yscale('log')
        plt.legend(loc='upper right', fontsize = 21)
        file_name = f'mse_d = {d_list[c]}.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')
    
    for c in range(num_d):
        fig = plt.figure(figsize=(7.5,3.5))
        max_element = 0
        min_element = 100
        for i, alpha in enumerate(alphas):
            inner_dict = data_dict[alpha]
            mse_list_start_avg = inner_dict.get('mse_list_start_avg')
            mse_list_start_std = inner_dict.get('mse_list_start_std')
            plot_mse(mse_list_start_avg[c],mse_list_start_std[c],colors[i],markers[i],linestyles[i],f'{latex_text} = {alpha}',linewidth = linewidth)
            max_element = max(max_element,np.max(np.maximum.reduce(mse_list_start_avg[c])))
            min_element = min(min_element,np.min(np.minimum.reduce(mse_list_start_avg[c])))
        plt.xlabel('Iteration', fontsize = fontsize1)
        plt.ylabel('RMSE', fontsize = fontsize1)
        plt.tick_params(labelsize=fontsize2)
        plt.ylim(0.18, 1.4) 
        plt.yscale('log')
        plt.legend(loc='upper right', fontsize = fontsize3)
        file_name = f'mse_d = {d_list[c]}_start.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(7.5,3.5))
        max_element = 0
        min_element = 100
        for i, alpha in enumerate(alphas):
            inner_dict = data_dict[alpha]
            mse_list_start_avg = inner_dict.get('mse_list_start_avg')
            mse_list_start_std = inner_dict.get('mse_list_start_std')
            plot_mse(mse_list_start_avg[c],mse_list_start_std[c],colors[i],markers[i],linestyles[i],f'{latex_text} = {alpha}',std = 2,linewidth = linewidth)
            max_element = max(max_element,np.max(np.maximum.reduce(mse_list_start_avg[c])))
            min_element = min(min_element,np.min(np.minimum.reduce(mse_list_start_avg[c])))
        plt.xlabel('Iteration', fontsize = fontsize1)
        plt.ylabel('RMSE', fontsize = fontsize1)
        plt.tick_params(labelsize=fontsize2)
        plt.ylim(0.18, 1.4) 
        plt.yscale('log')
        plt.legend(loc='upper right', fontsize = fontsize3)
        file_name = f'mse_d = {d_list[c]}_start_no_std.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(7.5,3.5))
        for i, alpha in enumerate(alphas):
            inner_dict = data_dict[alpha]
            model_gaps_avg = inner_dict.get('model_gaps_avg')
            model_gaps_std = inner_dict.get('model_gaps_std')
            plot_model_gap(model_gaps_avg[c],model_gaps_std[c],colors[i],markers[i],linestyles[i],f'{latex_text} = {alpha}',linewidth = linewidth)
        plt.xlabel('Iteration', fontsize = fontsize1)
        plt.ylabel(r'$\|\theta_t - \theta_{t-1}\|$', fontsize = fontsize1)
        plt.tick_params(labelsize=fontsize2)
        plt.legend(loc='upper right',fontsize = fontsize3)
        plt.ylim(1e-14, 10) 
        plt.yscale('log')
        file_name = f'Model_gap_d = {d_list[c]}.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(7.5,3.5))
        for i, alpha in enumerate(alphas):
            inner_dict = data_dict[alpha]
            model_gaps_avg = inner_dict.get('model_gaps_avg')
            model_gaps_std = inner_dict.get('model_gaps_std')
            plot_model_gap(model_gaps_avg[c],model_gaps_std[c],colors[i],markers[i],linestyles[i],f'{latex_text} = {alpha}',std = 2,linewidth = linewidth)
        plt.xlabel('Iteration', fontsize = fontsize1)
        plt.ylabel(r'$\|\theta_t - \theta_{t-1}\|$', fontsize = fontsize1)
        plt.tick_params(labelsize=fontsize2)
        plt.legend(loc='upper right',fontsize = fontsize3)
        plt.ylim(1e-14, 10) 
        plt.yscale('log')
        file_name = f'Model_gap_d = {d_list[c]}_no_std.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    plt.close('all')
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Plot Completion Time:", current_time_str)


if __name__ == "__main__":
    plot_fig_par(num_iters,d_list)

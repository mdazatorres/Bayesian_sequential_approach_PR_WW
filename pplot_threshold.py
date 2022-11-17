# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:20:52 2022

@author: crice
"""

from run_mcmc import mcmc_main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from matplotlib.pyplot import subplots
# from scipy.stats import gamma,binom
# import matplotlib.dates as mdates
# import datetime

plt.rcParams['font.size'] = 20
font_xylabel = 24

#city = 'Davis (sludge)'
city = 'UCDavis (sludge)'
#city = "Modesto"
workdir = "./"
dpi = 100 # fig quality
ind_var = {'Delta':1,'Omicron':2,'B245':3}

mcmc = mcmc_main(city=city)
output_mcmc = pd.read_csv('%soutput/mcmc_%s' %(workdir,city) , index_col=0)
#city_data = mcmc.city_data
city_data = mcmc.Data_mask

Output_all = output_mcmc.values
output_theta = mcmc.summary(Output_all)    

def plot_segments(x,y,col,ls="--"):
    plt.plot([x, x], [0, y], color=col, linestyle=ls)
    plt.plot([0,x],[y,y], color=col, linestyle=ls)


def FindClosestIndex(tt, d):
    """
    Find the index of the closest item (time, array) in tt to d[i] (obs. time).

    Parameters
    ----------
    tt: array
    d: array

    Returns
    -------
    index: array
         List of indexes.
    """
    index = np.array([0] * len(d))
    for j, t in enumerate(d):
        mn = np.min(np.abs(tt - t))
        for i, df in enumerate(abs(tt - t)):
            if (df == mn):
                index[j] = i
    return index

col1,col2,col3 = ["black","navy","orange"]
def Conce_levels(wave, xmax = 0.0008,ymax=0.36,ww="sludge"): 
    data_per = city_data[city_data.Variants==wave] #city_data[(city_data['SampleDate'] >= mcmc.training_date[wave][0]) & (city_data['SampleDate'] <= mcmc.training_date[wave][1])]
    output_theta_var = output_theta[:,[0, ind_var[wave]]]
    
    y = data_per["pos_rate"]
    x_con = data_per["NormalizedConc_s"]
    xmax = max([xmax,x_con.max()])
    x_con_crude = data_per["NormalizedConc_crude"]
    
    color = ["green","orange","red"]
    levels = np.array([5,8,10])/100

    concentration = np.linspace(1e-16,xmax,400)
    xx = concentration#(concentration - mcmc.sc.mean_[0])/mcmc.sc.scale_[0] # Scaler
    ones = np.ones((len(xx), 1))
    X = np.hstack((ones, xx.reshape(-1,1)))   
     
    Output_trace = mcmc.eval_predictive(output_theta_var, X)
    
    #mean_theta = np.mean(output_theta_var,0)
    #mean_theta_p = mcmc.rate(mean_theta, X)
    Q500 = np.quantile(Output_trace, 0.5, axis=0)
    Q025 = np.quantile(Output_trace, 0.025, axis=0)
    Q975 = np.quantile(Output_trace, 0.975, axis=0)
    Q250 = np.quantile(Output_trace, 0.25, axis=0)
    Q750 = np.quantile(Output_trace, 0.75, axis=0)
    #Out_df = pd.DataFrame({'Q500': Q500, 'Q025': Q025, 'Q975':Q975, 'Q250':Q250, 'Q750':Q750, 'mean_theta': mean_theta_p})

    # Find the level of concetration with P(pr<level) <= 0.8
    index_95 = FindClosestIndex(Q975,levels)
    index_75 = FindClosestIndex(Q750,levels)
    index = FindClosestIndex(Q500,levels)
    
    plt.figure(figsize=(12, 9))
    plt.plot(concentration,Q500)
    plt.fill_between(concentration, Q025,Q975, color=col2, alpha=0.15)
    plt.fill_between(concentration, Q250, Q750, color=col2, alpha=0.15)
    plt.ylabel("Positivity rate")

    [plot_segments(concentration[index][i],levels[i],col=color[i],ls="--") for i in range(len(levels))]
    [plot_segments(concentration[index_75][i],levels[i],col=color[i],ls="--") for i in range(len(levels))]
    [plot_segments(concentration[index_95][i],levels[i],col=color[i],ls="--") for i in range(len(levels))]
    # Real data
    plt.plot(x_con.values,y,".",color="black")
    plt.plot(x_con_crude.values,y,".", )

    plt.xlim([0,xmax]); plt.ylim([0,ymax])#1.05*max(Q975)]); 
    plt.xlabel("N/PMMoV")
    #plt.xticks(ticks= concentration[index], labels= np.round(concentration[index],6), rotation=0)
     
    xleve = np.round(concentration[index],5)
    xleve75 = np.round(concentration[index_75],5)
    xleve95 = np.round(concentration[index_95],5)
               
    pr_lev = pd.DataFrame({'Con95': xleve95, 'Con75': xleve75, 'Con50':xleve})
    print(pr_lev)#.to_latex())

    plt.savefig("figures/Levels_%s_%s.png"%(city,wave),dpi=dpi)
    return #True#print_con
# Davis
if city =='Davis (sludge)':
    xmax_d = 0.001; ymax_d = 0.126
    Conce_levels(wave = "Omicron", xmax =xmax_d,ymax=ymax_d,ww="sludge")
    Conce_levels(wave = "B245", xmax = xmax_d,ymax=ymax_d,ww="sludge")
    # Conce_levels(wave = "Delta", xmax = 0.0015,ymax=0.21,ww="sludge")
else: #city = 'UCDavis (sludge)'
    #xmax_uc = 0.000801; ymax_uc = 0.121
    xmax_uc = 0.001; ymax_uc = 0.126
    Conce_levels(wave = "Omicron", xmax = xmax_uc, ymax=ymax_uc,ww="sludge")
    Conce_levels(wave = "B245", xmax = xmax_uc, ymax=ymax_uc,ww="sludge")
    # Conce_levels(wave = "Delta", xmax = xmax_uc, ymax=ymax_uc,ww="sludge")

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:25:30 2022

@author: crice
"""

from run_mcmc import mcmc_main
import pickle
from datetime import date, timedelta,datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.size'] = 14
font_xylabel = 18
workdir = "./"

n_per = {'UCDavis (sludge)':28,'Davis (sludge)':28,'Modesto':18}

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


def find_thresholds(city = 'UCDavis (sludge)', alpha = 0.75,levels = np.array([5,8,10])/100):
    """
    We find the value of WW concentration, $c$, such that with probability $\alpha$
    the positivity rate (PR) is less than or equal to the CDC threshold $y$.
    That is find the levels of concetration C_1,C_2,C_3, such that
    P(Y < l_i | C_i) = alpha, for i = 1,2,3
    $\alpha$ should be set in the perspective of the precision needed to estimate
    the thresholds. CDC thresholds for the PR corresponding to low (Y > 0.05),
    moderate (0.05 < Y < 0.08), substantial (0.08 < Y < 0.1), and high Y > 0.1  transmission 
    
    Parameters
    ----------
    city: string
        City to analilze; city in {'UCDavis (sludge)'}
    alpha: float
        The precision we set to estimate the threshold alpha in (0,1).
    levels: array
        CDC thresholds for the positivity rate.

    Returns
    -------
    index: array
        Thresholds for the levels of concetration.
    """
    # Save the thresholds
    outname = "output/%s_thresholds_%s.csv"%(city,alpha)
    output = np.empty((n_per[city],len(levels)));
    dates = [""]*n_per[city]
    
    # Levels of concentration for predictions
    xmax_all = {'UCDavis (sludge)':0.002,'Davis (sludge)':0.002,'Modesto':0.0001} # Maximun concentration
    concentration = np.linspace(1e-16,xmax_all[city],500) # Grid of concentrations
    ones = np.ones((len(concentration), 1))
    X = np.hstack((ones, concentration.reshape(-1,1))) # Desing matrix
    

    for per in range(n_per[city]):
        # Class with the ouput of the posterior simulations and data     
        mcmc = mcmc_main(city=city, per=per)
        init_per = mcmc.init0 + timedelta(days=per * (mcmc.size_window - 1))
        dates[per] = init_per.strftime('%Y-%m-%d')
        # Load the MCMC output
        output_mcmc = pickle.load(open(workdir + 'output/' + city + 'per_' + str(per) + '_samples.pkl', 'rb'))

        output_theta = output_mcmc[:, :-1] # MCMC outpus for parameter beta_0 and meta_1    
        Output_trace = mcmc.eval_predictive(output_theta, X) # Predicted positivity rate
    
        # Compute the quantiles of the predictive positivity rate
        Q_alpha = np.quantile(Output_trace, alpha, axis=0)
        # Find the levels of concetration with P(pr<level|C) = alpha
        index_alpha = FindClosestIndex(Q_alpha,levels) # Index
        xleve_alpha = concentration[index_alpha]       # Concentration
        # Save the thresholds
        output[per] = xleve_alpha
        print("Period %s: %s" %(per,init_per))
    output = pd.DataFrame(output,columns=["PR_%s" %i for i in levels],index=pd.DatetimeIndex(dates))
    output.to_csv(outname,index=True,index_label="Dates")
    return output

def plot_thresholds(city,alpha,ymax= 0.003):
    lw_thresholds = 2.5
    alp = 1
    color = ["#88CCEE","#DDCC77","#882241"]#["#009E73","#E69F00","#D55E00"]
    outname = "output/%s_thresholds_%s.csv"%(city,alpha)
    mcmc = mcmc_main(city=city, per=0)
    city_data = mcmc.city_data
    city_data_all=city_data[(city_data['SampleDate'] >= mcmc.init0)]
    #y = city_data_all["pos_rate"]
    x_con = city_data_all["NormalizedConc_s"]
    x_con_crude = city_data_all["NormalizedConc_crude"]
    date_full = city_data_all.SampleDate#.values
    
    thresholds = pd.read_csv(outname,index_col="Dates")  
    dates = pd.DatetimeIndex(thresholds.index)#.strftime('%Y-%m-%d')
    shift = 13

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for i in range(n_per[city]):
        init = dates[i]
        if (init >= datetime(2021, 8, 22)) & (init < datetime(2021, 12, 1)):
            continue
        con_i = thresholds.iloc[i]
        days = mdates.drange(init, init + timedelta(shift), timedelta(days=1))  # how often do de plot
        pr_005, pr_008, pr_01 = np.ones(shift)*con_i[0], np.ones(shift)*con_i[1], np.ones(shift)*con_i[2]
        
        #fig, ax = subplots(num=1, figsize=(9, 5))
        ax.plot(days,pr_005, '-', linewidth=lw_thresholds, color=color[0],alpha=alp)
        ax.plot(days,pr_008, '-', linewidth=lw_thresholds, color=color[1],alpha=alp)
        ax.plot(days,pr_01, '-', linewidth=lw_thresholds, color=color[2],alpha=alp)
    # To put the labels
    ax.plot(days,pr_005, '-', linewidth=2, color=color[0],alpha=alp,label="Low transmission")
    ax.plot(days,pr_008, '-', linewidth=2, color=color[1],alpha=alp,label="Moderate transmission")
    ax.plot(days,pr_01, '-', linewidth=2, color=color[2],alpha=alp, label="High transmission")
    # Real data
    ax.scatter(date_full,x_con_crude,marker="<",s=10,color="black",label="Raw N/PMMoV")
    ax.scatter(date_full,x_con,marker="o",s=10,color="grey",label="Smoothed N/PMMoV")
    ax.set_ylabel("Thresholds for N/PMMoV")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_ylim(0,ymax)
    ax.set_xlim(mcmc.init0 - timedelta(days=1), mcmc.end)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021                                                                              2022', loc='left')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    ax.grid( linestyle='--', color='gray', alpha=0.2)
    ax.legend()
    
    #ax_p = ax.twinx()
    #p2, = ax_p.plot(date_full, y,"-", marker="<",markersize=3,color='green', label='Positivity rate')

    plt.savefig("figures/Levels_%s_%s.png"%(city,alpha),dpi=300)


# =============================================================================
#  
# =============================================================================
city = 'Davis (sludge)'
#thresholds = find_thresholds(city, alpha = 0.95,levels = np.array([5,8,10])/100)
plot_thresholds(city,alpha = 0.95,ymax= 0.0014)

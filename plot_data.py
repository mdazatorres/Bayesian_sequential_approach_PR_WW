from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_mcmc import mcmc_main
from datetime import date, timedelta


plt.rcParams['font.size'] = 13
workdir = "./"


def plot_conc_pos_rate(city):
    mcmc = mcmc_main(city=city, per=0)

    city_data = mcmc.city_data
    city_data = city_data[(city_data['SampleDate'] >= mcmc.init0) & (city_data['SampleDate'] <= mcmc.end)]
    Date = pd.DatetimeIndex(city_data['SampleDate'])
    fig, ax = subplots(num=1, figsize=(12, 5)) #(10, 6)
    ax_p = ax.twinx()
    p1, = ax.plot(pd.DatetimeIndex(city_data['SampleDate']),  city_data['NormalizedConc_s'], linewidth=2, color='k',    label='Smoothed N/PMMoV')
    p2, = ax_p.plot(pd.DatetimeIndex(city_data['SampleDate']), city_data['pos_rate'], 'o', linewidth=1,color='green', label='Positivity rate')

    ax.legend(handles=[p1, p2], loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_xlim(Date[0] - timedelta(days=2), Date[-1])
    ax.set_ylabel('N/PMMoV')
    ax_p.set_ylabel('Positivity rate')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021                                                                               2022', loc='left')
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    #plt.yticks([])
    fig.tight_layout()

    fig.savefig(workdir + 'figures/'+city + '_conc_vs_pos_rate.png')


def plot_data(city, save, workdir):
    mcmc = mcmc_main(city=city, per=0)
    fontsize = 14
    city_data = mcmc.city_data
    city_data = city_data[(city_data['SampleDate'] >= mcmc.init0) & (city_data['SampleDate'] <= mcmc.end)]
    Date = pd.DatetimeIndex(city_data['SampleDate'])

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines.right.set_position(("axes", 1.09))
    Y = city_data['NormalizedConc_crude']; lab1= 'N/PMMoV'
    Y1 = city_data['positives']; lab2='Cases'
    Y2 = city_data['Testing']; lab3='Tests'
    p2 = twin1.bar(Date, Y1, color='green', alpha=0.4, label=lab2)
    p3 = twin2.bar(Date, Y2, color='b',alpha=0.2, label=lab3)
    p1, = ax.plot(Date, Y, ".", color='k', label=lab1)

    ax.set_xlim(Date[0]-timedelta(days=2), Date[-1])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    # ax.tick_params(which='major', axis='x')
    yl = 1.02
    ax.set_ylim(0, np.nanmax(Y) * yl)
    twin1.set_ylim(0, np.nanmax(Y1) * yl)
    twin2.set_ylim(0, np.nanmax(Y2) * yl)

    ax.set_ylabel(lab1, fontsize=fontsize)
    twin1.set_ylabel(lab2, fontsize=fontsize)
    twin2.set_ylabel(lab3, fontsize=fontsize)

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='x', **tkw)
    ax.legend(handles=[p1, p2, p3], loc=1)

    ax.set_xlabel('2021                                                                                2022', loc='left')
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    fig.tight_layout()

    if save:
        fig.savefig(workdir + 'figures/' + city + '_cases_vs_tests.png')





city='UCDavis (sludge)'
#city='Davis (sludge)'

plot_data(city, save=True, workdir=workdir)



#plot_conc_pos_rate(city)







from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import trim_mean, pearsonr
from sklearn.linear_model import LinearRegression
from run_mcmc import mcmc_main
from datetime import date, timedelta
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from loess import loess_1d
import epyestim.covid19 as covid19

plt.rcParams['font.size'] = 13
workdir = "./"

wave_={'Delta':(pd.to_datetime('2021-07-01'), pd.to_datetime('2022-01-20')),
          'Omicron':(pd.to_datetime('2021-11-29'), pd.to_datetime('2022-04-28')),
          'B245':(pd.to_datetime('2022-02-11'), pd.to_datetime('2022-07-01'))}

wave={'Delta':(pd.to_datetime('2021-08-01'), pd.to_datetime('2021-12-10')),
          'Omicron':(pd.to_datetime('2021-12-24'), pd.to_datetime('2022-02-28')),
          'B245':(pd.to_datetime('2022-03-10'), pd.to_datetime('2022-06-15'))}
variants=['Delta', 'Omicron', 'B245']
colors=['r','cyan','blue']
def plot_conc_pos_rate(city):
    mcmc = mcmc_main(city=city, per=0)

    city_data = mcmc.city_data
    city_data = city_data[(city_data['SampleDate'] >= mcmc.init0) & (city_data['SampleDate'] <= mcmc.end)]
    #city_data = city_data[(city_data['SampleDate'] >= mcmc.training_date['Omicron'][0]) & (city_data['SampleDate'] <= mcmc.training_date['Omicron'][1])]

    Date = pd.DatetimeIndex(city_data['SampleDate'])
    fig, ax = subplots(num=1, figsize=(12, 5)) #(10, 6)
    ax_p = ax.twinx()


    p1, =ax.plot(pd.DatetimeIndex(city_data['SampleDate']),  city_data['NormalizedConc_s'], linewidth=2, color='k',    label='Smoothed N/PMMoV')
    p2, =ax_p.plot(pd.DatetimeIndex(city_data['SampleDate']), city_data['pos_rate'], 'o', linewidth=1,color='green', label='Positivity rate')

    #ax.plot(city_data['pos_rate'], city_data['NormalizedConc_crude'], 'o', color='orange', linewidth=2)
    #ax.plot(mcmc.Data_ana['SampleDate'], mcmc.Data_ana['NormalizedConc_loess'], linewidth=2, color='b', label='Smoothed N/PMMoV (LOESS)')

    #ax.set_ylim(-0.0001,0.0008)
    #ax_p.set_ylim(-0.0001, 0.08)
    ax.legend(handles=[p1, p2],loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_xlim(Date[0] - timedelta(days=2), Date[-1])
    ax.set_ylabel('N/PMMoV')
    ax_p.set_ylabel('Positivity rate')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021                                                                               2022', loc='left')
    ax.grid(color='gray', linestyle='--',alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    #plt.yticks([])
    fig.tight_layout()

    fig.savefig(workdir + 'figures/'+city + '_conc_vs_pos_rate.png')


def plot_data(city,  save, workdir):
    mcmc = mcmc_main(city=city, per=0)
    fontsize=14

    city_data = mcmc.city_data
    city_data = city_data[(city_data['SampleDate'] >= mcmc.init0) & (city_data['SampleDate'] <= mcmc.end)]
    Date = pd.DatetimeIndex(city_data['SampleDate'])

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("axes", 1.09))
    Y = city_data['NormalizedConc_crude']; lab1= 'N/PMMoV'
    Y1 = city_data['positives']; lab2='Cases'
    Y2 = city_data['Testing']; lab3='Tests'



    p2 = twin1.bar(Date, Y1, color='green', alpha=0.4, label=lab2)
    p3 = twin2.bar(Date, Y2, color='b',alpha=0.2, label=lab3)
    p1, = ax.plot(Date, Y, ".", color='k', label=lab1)

    #for i in range(3):
    #    init_var, end_var = wave[variants[i]]
    #    ax.axvline(x=init_var,ymin=0, ymax=1000, color=colors[i])
    #    ax.axvline(x=end_var, ymin=0, ymax=1000, color=colors[i])

    ax.set_xlim(Date[0]-timedelta(days=2), Date[-1])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    # ax.tick_params(which='major', axis='x')
    yl = 1.02
    ax.set_ylim(0, np.nanmax(Y) * yl)
    twin1.set_ylim(0, np.nanmax(Y1) * yl)
    twin2.set_ylim(0, np.nanmax(Y2) * yl)

    #ax.set_xlabel("Date", fontsize=fontsize)
    ax.set_ylabel(lab1, fontsize=fontsize)
    twin1.set_ylabel(lab2, fontsize=fontsize)
    twin2.set_ylabel(lab3, fontsize=fontsize)

    #ax.yaxis.label.set_color(p1.get_color())
    #twin1.yaxis.label.set_color(p2.get_color())
    #twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    #ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    #twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    #twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    #twin2.tick_params(axis='y', colors='b', **tkw)
    ax.tick_params(axis='x', **tkw)
    ax.legend(handles=[p1, p2, p3], loc=1)

    ax.set_xlabel('2021                                                                                2022', loc='left')
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    fig.tight_layout()

    if save:
        fig.savefig(workdir + 'figures/' + city + '_cases_vs_tests.png')







city='Davis (sludge)'
#city='Modesto'

plot_data(city, save=True, workdir=workdir)



#plot_conc_pos_rate(city)







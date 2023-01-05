from run_mcmc import mcmc_main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
from datetime import date, timedelta
import pickle


plt.rcParams['font.size'] = 14
font_xylabel = 18
workdir = "./"


def plot_beta(city, pers, ax):
    mcmc = mcmc_main(city=city, per=0)
    length = mcmc.num_data
    shift = length + mcmc.forecast

    for i in range(pers):
        init = mcmc.init0 + timedelta(days=i * (mcmc.size_window - 1))
        output_mcmc = pickle.load(open(workdir + 'output/' + city + 'per_' + str(i) + '_samples.pkl', 'rb'))
        beta = output_mcmc[:, 1]
        q095, q500, q050, q750, q250 = np.quantile(beta, q=[0.95, 0.5, 0.05, 0.75, 0.25])
        q095_v, q500_v, q050_v, q750_v, q250_v = np.ones(shift)*q095, np.ones(shift)*q500, np.ones(shift)*q050, np.ones(shift)*q750, np.ones(shift)*q250
        days = mdates.drange(init, init + timedelta(shift), timedelta(days=1))  # how often do de plot
        if i < pers-1:
            ax.plot(days[:mcmc.size_window], q500_v[:mcmc.size_window], '-', linewidth=2, color='r')
            ax.fill_between(days[:mcmc.size_window], q050_v[:mcmc.size_window], q095_v[:mcmc.size_window], color='blue', alpha=0.2)
            ax.fill_between(days[:mcmc.size_window], q250_v[:mcmc.size_window], q750_v[:mcmc.size_window], color='blue', alpha=0.3)
        else:
            ax.plot(days[:shift], q500_v[:shift], '-', linewidth=2, color='r')
            ax.fill_between(days[:shift], q050_v[:shift], q095_v[:shift], color='blue', alpha=0.2)
            ax.fill_between(days[:shift], q250_v[:shift], q750_v[:shift], color='blue', alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_xlim(mcmc.init0, mcmc.end)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlim(mcmc.init0 - timedelta(days=3), mcmc.end + timedelta(days=35))
    ax.tick_params(which='major', axis='x')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")



def plot_post(city, per, ax, color):
    mcmc = mcmc_main(city=city, per=per)

    output_mcmc = pickle.load(open(workdir + 'output/' + city + 'per_' + str(per) + '_samples.pkl', 'rb'))

    city_data = mcmc.city_data
    init_per = mcmc.init0 + timedelta(days=per * (mcmc.size_window - 1))
    init_for = mcmc.init0 + timedelta(days=per * (mcmc.size_window - 1)+mcmc.num_data)
    end_per = init_per + timedelta(days=mcmc.num_data+mcmc.forecast)

    data_per = city_data[(city_data['SampleDate'] >= init_per) & (city_data['SampleDate'] <= end_per)]
    Xtest = mcmc.getX(init_per,  end_per)


    output_theta = output_mcmc[:,:-1]


    Output_trace = mcmc.eval_predictive(output_theta, Xtest)

    Q500 = np.quantile(Output_trace, 0.5, axis=0)

    Q025 = np.quantile(Output_trace, 0.025, axis=0)
    Q975 = np.quantile(Output_trace, 0.975, axis=0)

    Q250 = np.quantile(Output_trace, 0.15, axis=0) # 0.15 0.85
    Q750 = np.quantile(Output_trace, 0.85, axis=0)

    Out_df = pd.DataFrame({'Q500': Q500, 'Q025': Q025, 'Q975':Q975, 'Q250':Q250, 'Q750':Q750})
    Out_df['SampleDate'] = pd.DatetimeIndex(data_per['SampleDate'])

    ax.plot(Out_df['SampleDate'], Out_df['Q500'], markersize=2, linewidth=2, color='r')
    ax.fill_between(Out_df['SampleDate'], Out_df['Q025'], Out_df['Q975'], color=color, alpha=0.3)
    ax.plot(pd.DatetimeIndex(data_per['SampleDate']), data_per['pos_rate'],'o', markersize=2, color='k')
    ax.set_xlim(mcmc.init0-timedelta(days=3), mcmc.end+timedelta(days=35))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))

    ax.tick_params(which='major', axis='x')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")


def plot_post_wave(city, wave, all_wave, color):
    fig, ax = subplots(num=1, figsize=(9, 5))
    plot_post(city, wave, all_wave,  ax, color)
    fig.tight_layout()

    fig.savefig(workdir + 'figures/' + city +'_'+wave+ '_post_pos_rate.png')


def plot_all_wave(city, pers):
    fig, ax = subplots(num=1, figsize=(12, 5))
    for i in range(pers):
        plot_post(city, per=i, ax=ax, color='b')
    ax.set_ylabel('Posivity rate', fontsize=14)
    ax.set_xlabel('2021                                                                    2022', loc='left')
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(workdir + 'figures/' + city +'_post_pos_rate.png')



def plot_post_wave_beta(city, pers):

    fig, _axs = plt.subplots(nrows=2, ncols=1, figsize=(13, 7), sharex=True)

    axs = _axs.flatten()

    for i in range(pers):
        plot_post(city, per=i, ax=axs[1], color='b')
    axs[1].set_ylabel('Positivity rate', fontsize=font_xylabel)
    plot_beta(city, pers=pers, ax=axs[0])

    axs[0].set_ylabel(r'$\beta_1$', fontsize=font_xylabel)
    axs[0].grid( linestyle='--', color='gray', alpha=0.2)
    axs[1].grid( linestyle='--', color='gray', alpha=0.2)

    axs[1].set_xlabel('2021                                                                         2022', loc='left')
    plt.setp(axs[1].get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    fig.savefig(workdir + 'figures/' + city +'_post_pos_rate.png')





#ax.legend(frameon=False)
#plot_beta(city,ax)
#plot_params()
#city='Davis (sludge)'
city='UCDavis (sludge)'
#plot_beta(city,ax=ax)
plot_all_wave(city, pers=28)
#plot_post_wave_beta(city, pers=28)



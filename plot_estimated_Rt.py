from run_mcmc import mcmc_main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
from datetime import timedelta
import pickle
from scipy.stats import erlang
import matplotlib as mpl
import epyestim.covid19 as covid19



plt.rcParams['font.size'] = 14
font_xylabel = 18


workdir = "./"

def plot_Rt(city, per, cases, ax):
    mcmc = mcmc_main(city=city, per=per)
    output_mcmc = pickle.load(open(workdir + 'output/' + city + 'per_' + str(per) + '_samples.pkl', 'rb'))

    Tmean= mcmc.Tmean[city]

    city_data = mcmc.city_data
    init_per = mcmc.init0 + timedelta(days=per * (mcmc.size_window - 1))
    init_for = mcmc.init0 + timedelta(days=per * (mcmc.size_window - 1) + mcmc.num_data)
    end_per = init_per + timedelta(days=mcmc.num_data + mcmc.forecast)

    data_per = city_data[(city_data['SampleDate'] >= init_per) & (city_data['SampleDate'] <= end_per)]
    city_data_all= mcmc.data_city_all_per#city_data[(city_data['SampleDate'] >= mcmc.init0)]
    Xtest = mcmc.getX(init_per, end_per)

    output_theta = output_mcmc[:, :-1]

    Output_trace = mcmc.eval_predictive(output_theta, Xtest)

    Q500 = np.quantile(Output_trace, 0.5, axis=0)

    Q025 = np.quantile(Output_trace, 0.025, axis=0)
    Q975 = np.quantile(Output_trace, 0.975, axis=0)

    Q250 = np.quantile(Output_trace, 0.15, axis=0)  # 0.15 0.85
    Q750 = np.quantile(Output_trace, 0.85, axis=0)

    Out_df = pd.DataFrame({'Q500': Q500, 'Q025': Q025, 'Q975': Q975, 'Q250': Q250, 'Q750': Q750})
    Out_df['SampleDate'] = pd.DatetimeIndex(data_per['SampleDate'])

    #davisdf_s = pd.Series(data=data_per['positives_crude'].values, index=Out_df['SampleDate'])
    davisdf_pr = pd.Series(data=Out_df['Q500'].values*Tmean, index=Out_df['SampleDate'])

    #ch_time_varying_r = covid19.r_covid(davisdf_s)
    ch_time_varying_pr = covid19.r_covid(davisdf_pr)

    n_pr = ch_time_varying_pr.shape[0]
    init_date = Out_df['SampleDate'][0]
    delay = 21 #11
    days = pd.date_range(start=init_date + pd.Timedelta(days=delay), end=init_date+pd.Timedelta(days=delay+n_pr-1))
    date_pred = days
    if cases:
        dl = 10
        city_data_all = city_data_all.reset_index()
        davis_all = pd.Series(data=city_data_all['positives'].values[:-45], index=city_data_all['SampleDate'][:-45])
        ch_time_varying_r_all = covid19.r_covid(davis_all)
        n_all = ch_time_varying_r_all.shape[0]
        date_all = pd.date_range(start=mcmc.init0 + pd.Timedelta(days=21+dl), end=mcmc.init0 + pd.Timedelta(days=21+dl + n_all - 1))

        ax.plot(date_pred, ch_time_varying_pr['Q0.5'], color='b')
        ax.fill_between(date_pred, ch_time_varying_pr['Q0.025'], ch_time_varying_pr['Q0.975'], facecolor='b', alpha=0.2, hatch= '/', edgecolor='b',label=r'$R_e(t)$ with estimated PR')
        ax.plot(date_all, ch_time_varying_r_all['Q0.5'], color='k')
        ax.fill_between(date_all, ch_time_varying_r_all['Q0.025'], ch_time_varying_r_all['Q0.975'], facecolor='k', alpha=0.2, hatch= r'$\$', edgecolor='k', label=r'$R_e(t)$ with observed cases')
        ax.set_xlim([mcmc.init0 + pd.Timedelta(days=21+dl), mcmc.init0 + pd.Timedelta(days=21 +dl+ n_all - 1 - 8)])
    else:
        ax.plot(date_pred, ch_time_varying_pr['Q0.5'], color='b')
        ax.fill_between(date_pred, ch_time_varying_pr['Q0.025'], ch_time_varying_pr['Q0.975'], facecolor='b', alpha=0.2, hatch= '/', edgecolor='b')

    ax.set_ylabel(r'$R_e(t)$ with $95\%$ CI')
    ax.set_xlabel('2021                                                                           2022', loc='left')
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(which='major', axis='x')


#def Rt(x):
#    IP_dist = erlang(a=3, scale=8 / 3)
#    nn = 10
#    w = np.diff(IP_dist.cdf(np.arange(0, nn + 1)))
#    w /= sum(w)
#    w = np.flip(w)
#    x = x.values
#    rt = x[-1]/np.sum(x*w)
#    return rt


def plot_all_Rt(city):
    fig, ax = subplots(num=1, figsize=(12, 5))
    for i in range(25):
        plot_Rt(city, cases=False, per=i, ax=ax)
    plot_Rt(city, cases=True, per=25, ax=ax)
    mpl.rcParams['axes.spines.right'] = False
    ax.axhline(y=1, color="red")
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    fig.savefig(workdir + 'figures/' + city + '_Rt.png')




city='Davis (sludge)'
#city='UCDavis (sludge)'
plot_all_Rt(city)




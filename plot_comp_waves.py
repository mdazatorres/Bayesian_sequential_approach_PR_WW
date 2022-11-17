from run_mcmc import mcmc_main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from scipy.stats import gamma,binom
import matplotlib.dates as mdates
from datetime import date, timedelta
import datetime
import pickle
from scipy.stats import erlang, gamma
import matplotlib as mpl
import epyestim.covid19 as covid19
from Rts_AR import Rts_AR,Rts_P


plt.rcParams['font.size'] = 14
font_xylabel = 18


#city = 'Davis'
#wave = 'all'
#wave = 'Omicron'
#wave = 'B45'
workdir = "./"


wave={'Delta':(pd.to_datetime('2021-08-01'), pd.to_datetime('2021-12-10')),
          'Omicron':(pd.to_datetime('2021-12-24'), pd.to_datetime('2022-02-28')),
          'B245':(pd.to_datetime('2022-03-10'), pd.to_datetime('2022-06-15'))}

wave_={'Delta':(pd.to_datetime('2021-07-01'), pd.to_datetime('2022-01-20')),
          'Omicron':(pd.to_datetime('2021-11-29'), pd.to_datetime('2022-04-28')),
          'B245':(pd.to_datetime('2022-02-11'), pd.to_datetime('2022-07-01'))}
variants=['Delta', 'Omicron', 'B245']
colors=['r','cyan','blue']


def plot_params(city, per, ax):
    output_mcmc = pickle.load(open(workdir + 'output/' + city + 'per_' + str(per) + '_samples.pkl', 'rb'))
    ax.hist(output_mcmc[:, 1], lw=2, density=True, histtype=u'step',label=per)

def plot_all_beta_hist(city):
    fig, ax = subplots(num=1, figsize=(9, 5))
    for i in range(30):
        plot_params(city, per=i, ax=ax)
    ax.legend()



def plot_beta(city,pers,ax):
    mcmc = mcmc_main(city=city, per=0)
    length = mcmc.num_data
    shift = length + mcmc.forecast
    #fig, ax = subplots(num=1, figsize=(12, 5))

    for i in range (pers):

        init = mcmc.init0 + timedelta(days=i * (mcmc.size_window - 1))
        output_mcmc = pickle.load(open(workdir + 'output/' + city + 'per_' + str(i) + '_samples.pkl', 'rb'))
        beta = output_mcmc[:, 1]

        q095, q500, q050, q750, q250 = np.quantile(beta, q=[0.95, 0.5, 0.05, 0.75, 0.25])
        q095_v, q500_v, q050_v, q750_v, q250_v = np.ones(shift)*q095, np.ones(shift)*q500, np.ones(shift)*q050, np.ones(shift)*q750, np.ones(shift)*q250
        days = mdates.drange(init, init + timedelta(shift), timedelta(days=1))  # how often do de plot
        if i<pers-1:
            ax.plot(days[:mcmc.size_window], q500_v[:mcmc.size_window], '-', linewidth=2, color='r')
            ax.fill_between(days[:mcmc.size_window], q050_v[:mcmc.size_window], q095_v[:mcmc.size_window], color='blue', alpha=0.2)
            ax.fill_between(days[:mcmc.size_window], q250_v[:mcmc.size_window], q750_v[:mcmc.size_window], color='blue', alpha=0.3)
        else:
            ax.plot(days[:shift], q500_v[:shift], '-', linewidth=2, color='r')
            ax.fill_between(days[:shift], q050_v[:shift], q095_v[:shift], color='blue', alpha=0.2)
            ax.fill_between(days[:shift], q250_v[:shift], q750_v[:shift], color='blue', alpha=0.3)

    #for i in range(3):
    #    init_var, end_var = wave[variants[i]]
    #    ax.axvline(x=init_var, ymin=0, ymax=1000, color=colors[i])
    #    ax.axvline(x=end_var, ymin=0, ymax=1000, color=colors[i])
    #ax.set_ylabel(r'$\beta_1$')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_xlim(mcmc.init0, mcmc.end)
    #ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    #ax.set_xlabel('2021         ' + '                              ' + '     2022')
    ax.set_xlim(mcmc.init0 - timedelta(days=3), mcmc.end + timedelta(days=35))
    ax.tick_params(which='major', axis='x')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    #fig.tight_layout()

    #fig.savefig(workdir + 'figures/' + city +'_beta_pos.png')

    #ax.set_xlim(init0 - dt.timedelta(1), init + dt.timedelta(shift))
    #plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")





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
    #ax.fill_between(Out_df['SampleDate'], Out_df['Q250'], Out_df['Q750'], color=color, alpha=0.3)



    ax.plot(pd.DatetimeIndex(data_per['SampleDate']), data_per['pos_rate'],'o', markersize=2, color='k')
    ax.set_xlim(mcmc.init0-timedelta(days=3), mcmc.end+timedelta(days=35))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))

    ax.tick_params(which='major', axis='x')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")


def plot_post_wave(city, wave, all_wave, color):
    fig, ax = subplots(num=1, figsize=(9, 5))
    plot_post(city, wave, all_wave,  ax, color)
    fig.tight_layout()

    fig.savefig(workdir + 'figures/' + city +'_'+wave+ '_post_pos_rate.png')


# pers=28

def plot_all_wave(city, pers):
    fig, ax = subplots(num=1, figsize=(12, 5))

    for i in range(pers):
        plot_post(city, per=i, ax=ax, color='b')
    ax.set_ylabel('Posivity rate', fontsize=14)
    #ax.set_xlabel('2021         ' + '                              ' + '     2022')

    ax.set_xlabel('2021                                                                    2022', loc='left')
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig(workdir + 'figures/' + city +'_post_pos_rate.png')
    #ax.set_ylim(0, 0.1)
    #fig.savefig(workdir + 'figures/' + city +'_all_wave'+ '_post_pos_rate.png')

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



def plot_Rt(city, per, cases, ax):
    #city_data_all.Testing[:-45].mean()
    # 1191.5434173669469
    mcmc = mcmc_main(city=city, per=per)

    output_mcmc = pickle.load(open(workdir + 'output/' + city + 'per_' + str(per) + '_samples.pkl', 'rb'))

    city_data = mcmc.city_data
    init_per = mcmc.init0 + timedelta(days=per * (mcmc.size_window - 1))
    init_for = mcmc.init0 + timedelta(days=per * (mcmc.size_window - 1) + mcmc.num_data)
    end_per = init_per + timedelta(days=mcmc.num_data + mcmc.forecast)

    data_per = city_data[(city_data['SampleDate'] >= init_per) & (city_data['SampleDate'] <= end_per)]
    city_data_all=city_data[(city_data['SampleDate'] >= mcmc.init0)]
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

    davisdf_s = pd.Series(data=data_per['positives_crude'].values, index=Out_df['SampleDate'])
    #davisdf_pr2 = pd.Series(data=Out_df['Q500'].values*800, index=Out_df['SampleDate'])
    davisdf_pr = pd.Series(data=Out_df['Q500'].values*1200, index=Out_df['SampleDate'])

    ch_time_varying_r = covid19.r_covid(davisdf_s)
    ch_time_varying_pr = covid19.r_covid(davisdf_pr)
    #ch_time_varying_pr2 = covid19.r_covid(davisdf_pr2)
    nn=ch_time_varying_pr.shape[0]
    #mm=ch_time_varying_pr2.shape[0]
    #if ch_time_varying_pr.shape[0]< ch_time_varying_pr2.shape[0]:
    #    ch_time_varying_pr2 = ch_time_varying_pr2[mm-nn:]


    n_pr= ch_time_varying_pr.shape[0]
    init_date= Out_df['SampleDate'][0]
    delay=10 #11
    days = pd.date_range(start=init_date+pd.Timedelta(days=delay), end=init_date+pd.Timedelta(days=delay+n_pr-1))
    #fig, ax = subplots(num=1, figsize=(12, 5))

    #date_pred = pd.DatetimeIndex(ch_time_varying_pr.index)

    #date_cases = pd.DatetimeIndex(ch_time_varying_r.index)
    #date_pred = ch_time_varying_pr.index

    date_pred = days#=ch_time_varying_pr.index.strftime("%b-%d-Y%")
    #ax.plot(date_cases, ch_time_varying_r['Q0.5'], color='k')
    #ax.fill_between(date_cases, ch_time_varying_r['Q0.025'], ch_time_varying_r['Q0.975'], color='k', alpha=0.1)
    if cases:
        dl=9
        city_data_all = city_data_all.reset_index()
        # fig, ax = subplots(num=1, figsize=(12, 5))
        davis_all = pd.Series(data=city_data_all['positives_crude'].values[:-45], index=city_data_all['SampleDate'][:-45])
        ch_time_varying_r_all = covid19.r_covid(davis_all)
        # date_all1 = ch_time_varying_r_all.index.strftime("%b-%d-%Y")
        n_all = ch_time_varying_r_all.shape[0]
        date_all = pd.date_range(start=mcmc.init0 + pd.Timedelta(days=21+dl), end=mcmc.init0 + pd.Timedelta(days=21+dl + n_all - 1))

        ax.plot(date_pred, ch_time_varying_pr['Q0.5'], color='b')
        ax.fill_between(date_pred, ch_time_varying_pr['Q0.025'], ch_time_varying_pr['Q0.975'], facecolor='b', alpha=0.2, hatch= '/', edgecolor='b',label=r'$R_e(t)$ with estimated PR')

        #ax.plot(date_pred, ch_time_varying_pr2['Q0.5'], color='g', label='Predicted PR with T=1000')
        #ax.fill_between(date_pred, ch_time_varying_pr2['Q0.025'], ch_time_varying_pr2['Q0.975'], facecolor='g', alpha=0.1, hatch= r'$\$', edgecolor='g')

        ax.plot(date_all, ch_time_varying_r_all['Q0.5'], color='k')
        ax.fill_between(date_all, ch_time_varying_r_all['Q0.025'], ch_time_varying_r_all['Q0.975'], facecolor='k', alpha=0.2, hatch= r'$\$', edgecolor='k', label=r'$R_e(t)$ with observed cases')
        ax.set_xlim([mcmc.init0 + pd.Timedelta(days=21), mcmc.init0 + pd.Timedelta(days=21 + n_all - 1 - 8)])
    else:
        ax.plot(date_pred, ch_time_varying_pr['Q0.5'], color='b')
        ax.fill_between(date_pred, ch_time_varying_pr['Q0.025'], ch_time_varying_pr['Q0.975'], facecolor='b', alpha=0.2, hatch= '/', edgecolor='b')

        #ax.plot(date_pred, ch_time_varying_pr2['Q0.5'], color='g')
        #ax.fill_between(date_pred, ch_time_varying_pr2['Q0.025'], ch_time_varying_pr2['Q0.975'], facecolor='g', alpha=0.1, hatch= r'$\$', edgecolor='g')

    ax.set_ylabel(r'$R_e(t)$ with $95\%$ CI')
    ax.set_xlabel('     2021                                                                       2022', loc='left')
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(which='major', axis='x')





def Rt(x):
    IP_dist = erlang(a=3, scale=8 / 3)
    nn = 10
    w = np.diff(IP_dist.cdf(np.arange(0, nn + 1)))
    w /= sum(w)
    w = np.flip(w)
    x=x.values
    rt=x[-1]/np.sum(x*w)
    #m = len(pr)  ##Data size
    #pr=Out_df['Q500']
    #re=pr.rolling(window=nn).apply(lambda x: Rt(x))
    return rt

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



#plot_beta(city,ax)
#plot_params()

#plot1(Sgene=True,mcmc=mcmc_Sg)

#city = 'Modesto'
#city='Davis'
#city='Davis (sludge)'
city='Davis (sludge)'
#plot_beta(city,ax=ax)
#plot_all_wave(city, pers=28)
#plot_post_wave_beta(city, pers=28)
#plot_params(city, par=-1)
#plot_comp_params(city)

plot_all_Rt(city)
#plt.rcParams['font.size'] = 10

#fig, ax = subplots(num=1, figsize=(12, 5))
#plot_Rt(city=city,per=12,cases=True,ax=ax)


#mcmc = mcmc_main(city=city, per=0)

# city_data_all=mcmc.city_data[(mcmc.city_data['SampleDate'] >= mcmc.init0)]
#
# date=pd.DatetimeIndex(city_data_all['SampleDate'][2:-45]).strftime("%d-%m-%Y")
# cases_av=city_data_all['positives_moving_average'][2:-45]
# date_=pd.to_datetime(date, format="%d-%m-%Y",exact=True)
#
#
# davisdf_s = pd.Series(data=cases_av.values, index=date_)
# ch_time_varying_r_all = covid19.r_covid(davisdf_s)

from datetime import date
"""
tst = Rts_AR(np.array(cases_av.values), init_date=date_[0], n=len(cases_av.values))
tst.CalculateRts()
end_date=date_[-1]
days= pd.date_range(start=end_date-pd.Timedelta(days=tst.n-1), end=end_date)

fig, ax = subplots(num=1, figsize=(12, 5))
ax.plot(days, tst.rts[1], color='b', lw=2)
ax.fill_between(days, tst.rts[0], tst.rts[2], facecolor='b',alpha=0.3, hatch='/', edgecolor='b', lw=1, label='ta')
ax.plot(ch_time_varying_r_all.index[12:], ch_time_varying_r_all['Q0.5'][:-12], color='k')
ax.fill_between(ch_time_varying_r_all.index[12:], ch_time_varying_r_all['Q0.025'][:-12], ch_time_varying_r_all['Q0.975'][:-12], color='k', alpha=0.1)
"""


#spainRt = Rts_AR(spain_data.Cases_average.values, init_date=date[0], n=5)


#ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))



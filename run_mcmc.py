
import numpy as np
import pandas as pd
import os,sys, pickle
from pytwalk import pytwalk
from scipy import stats
from datetime import timedelta
from scipy.stats import multivariate_normal

pd.options.mode.chained_assignment = None


class mcmc_main:
    def __init__(self, city, per):
        self.data_ww = pd.read_csv('data/data_ww_cases.csv')
        self.end = pd.to_datetime('2022-07-01')
        self.init0 = pd.to_datetime('2021-07-01')
        self.per = per
        self.size_window = 14

        self.init = self.init0 + timedelta(days=self.per * (self.size_window - 1))
        self.num_data = 30
        self.forecast = 10

        self.city = city
        self.city_data = self.read_data()
        self.y_data, self.X, self.city_data = self.read_data()
        self.n = len(self.y_data)
        self.n_sample = 3000

        self.Phi = {'UCDavis (sludge)': 287, 'Davis (sludge)': 237, 'Modesto': 41, 'Davis': 260, 'UCDavis': 260,'Merced': 50}
        self.Mu_b0 = {'UCDavis (sludge)':-4, 'Davis (sludge)':-4, 'Modesto':-4, 'Merced':-4, 'Davis':-4, 'UCDavis':-4, 'all':-4}
        self.Sig_b0 = {'UCDavis (sludge)':1, 'Davis (sludge)':1, 'Modesto':1, 'Merced':1, 'Davis':1, 'UCDavis':1, 'all':1}
        self.Mu_b1 = {'UCDavis (sludge)':3000, 'Davis (sludge)':3000, 'Modesto':3000, 'Merced':4500,'Davis':500, 'UCDavis':500, 'all':4500} # Omicron
        self.Sig_b1 = {'UCDavis (sludge)':1000**2, 'Davis (sludge)':1000**2, 'Modesto':1000**2,'Merced':1000**2, 'Davis':150, 'UCDavis':150}

        self.mu = np.array([self.Mu_b0[self.city], self.Mu_b1[self.city]])
        self.sig = np.array([self.Sig_b0[self.city], self.Sig_b1[self.city]])
        self.phi = self.Phi[self.city]

        self.d = len(self.mu)     # number of parameters to estimate
        self.burnin = 5000        # burnin size
        self.thini = 10           # integration autocorrelation time

        if os.path.isfile('output/' + self.city + '_post_params.pkl'):
            self.post_params = pickle.load(open('output/' + self.city+ '_post_params.pkl', 'rb'))
        else:
            self.post_params={}

    
    def trim_fun(self, x):
        x = x.dropna()
        x1 = x.sort_values().ravel()
        return np.mean(x1[1:-1])


    def read_data(self):
        city_data = self.data_ww[self.data_ww['City'] == self.city]
        city_data = city_data.reset_index()
        city_data['SampleDate'] = pd.to_datetime(city_data['SampleDate'])
        self.data_city_all_per = city_data[(city_data['SampleDate'] >= self.init0) & (city_data['SampleDate'] <= self.end)]

        city_data['positives_moving_average'] = np.copy(city_data['positives'].rolling(window=7, center=False, min_periods=3).mean())
        city_data['NormalizedConc_s'] = city_data['NormalizedConc_crude'].rolling(window=7, min_periods=3, center=True).apply(lambda x: self.trim_fun(x))

        city_data['NormalizedConc_s'] = city_data['NormalizedConc_s'].interpolate()
        city_data['positives'][city_data.Testing.isna()] = np.nan
        city_data['pos_rate'] = city_data['positives']/city_data['Testing']
        city_data['pos_rate_average'] = city_data['pos_rate'].rolling(window=7, center=True, min_periods=3).mean()

        self.Data_ana = city_data[(city_data['SampleDate'] >= self.init) & (city_data['SampleDate'] <= self.init+ timedelta(days=self.num_data))]

        mask = self.Data_ana['pos_rate'] > 0
        self.Data_mask = self.Data_ana.loc[mask, :]
        self.weight = (self.Data_mask['Testing']/self.Data_mask['Testing'].mean())

        x = self.Data_mask[['NormalizedConc_s']]
        ones = np.ones((x.shape[0], 1))
        X = np.hstack((ones, x))

        y_data = self.Data_mask['pos_rate']

        return y_data.values, X, city_data

    def getX(self, init, end):
        Data_per = self.city_data[(self.city_data['SampleDate'] >= init) & (self.city_data['SampleDate'] <= end)]
        x = Data_per[['NormalizedConc_s']]
        ones = np.ones((x.shape[0], 1))
        X = np.hstack((ones, x))
        return X

    def rate(self, x, X):
        # x: pars,
        # X: covariable matriz (ww data)
        eta = X @ x
        e_eta = np.exp(eta)
        mu = e_eta/(1+e_eta)
        return mu

    def eval_rate(self, Out, X):
        xfunc = lambda x:self.rate(x, X)
        return np.apply_along_axis(xfunc, 1, Out)

    def predictive(self, x, X):
        mu = self.rate(x, X)
        a = mu * self.phi
        b = self.phi - a
        Output_trace = stats.beta.rvs(a, b)
        return Output_trace

    def eval_predictive(self, Out, conc):
        xfunc = lambda x:self.predictive(x, conc)
        return np.apply_along_axis(xfunc, 1, Out)

    def mean_predictive(self, x, X):
        mu = self.rate(x, X)
        return mu

    def eval_mean_predictive(self, Out, conc):
        xfunc = lambda x:self.mean_predictive(x, conc)
        return np.apply_along_axis(xfunc, 1, Out)

    def loglikelihood(self, x):
        mu = self.rate(x, self.X)
        a = mu * self.phi
        b = self.phi - a
        log_likelihood = np.sum(self.weight.values*stats.beta.logpdf(self.y_data, a, b))
        #log_likelihood = np.sum(stats.beta.logpdf(self.y_data, a, b))
        return log_likelihood


    def logprior(self, x):
        """
        Logarithm of a normal distribution
        """
        if self.per == 0:
            cov = np.diag(self.sig)
            log_prior = multivariate_normal.logpdf(x, mean=self.mu, cov=cov)
        else:
            mu_b0, std_b0 = self.post_params['beta0_' + str(self.per - 1)]
            mu_b1, std_b1 = self.post_params['beta1_' + str(self.per - 1)]
            log_prior = multivariate_normal.logpdf(x, mean=np.array([mu_b0, mu_b1]), cov=np.diag(np.array([std_b0**2, std_b1**2])))
        return log_prior


    def Energy(self, x):
        """
        -log of the posterior distribution
        """
        return -1*(self.loglikelihood(x) + self.logprior(x))

    def Supp(self, x):
        """
        Support of the parameters to be estimated
        """
        return True

    def LG_Init(self):
        """
        Initial condition
        """
        if self.per==0:
            cov= np.diag(self.sig)
            sim = multivariate_normal.rvs(mean=self.mu, cov=cov)
        else:
            mu_b0, std_b0 = self.post_params['beta0_' + str(self.per - 1)]
            mu_b1, std_b1 = self.post_params['beta1_' + str(self.per - 1)]
            sim = multivariate_normal.rvs(mean=np.array([mu_b0, mu_b1]), cov=np.diag(np.array([std_b0**2, std_b1**2])))

        return sim.ravel()

    def fit_posterior(self):
        scl = 1.2
        beta0 = self.samples[:, 0]
        mu = beta0.mean()
        std = beta0.std() * scl
        self.post_params['beta0_' + str(self.per)] = (mu, std)

        beta1 = self.samples[:, 1]
        mu = beta1.mean()
        #std = beta1.mean() * scl
        std = beta1.std() * scl
        self.post_params['beta1_' + str(self.per)] = (mu, std)

    def RunMCMC(self, T=50000, burnin=1000):

        self.twalk = pytwalk(n=self.d, U=self.Energy, Supp=self.Supp)
        self.twalk.Run(T=T, x0=self.LG_Init(), xp0=self.LG_Init())

        self.iat = int(self.twalk.IAT(start=burnin)[0, 0])
        self.burnin = burnin
        #print("\nEffective sample size: %d" % ((T-burnin)/self.iat,))
        self.samples = self.twalk.Output[burnin::(self.iat), :]  # Burn in and thining, output t-wal
        self.fit_posterior()
        print("\nSaving files in ", 'output/' + self.city + '_*.pkl')
        pickle.dump(self.samples, open( 'output/' + self.city + 'per_' + str(self.per)+'_samples.pkl', 'wb'))

        outname_var =  'output/'+ self.city + '_post_params.pkl'

        with open(outname_var, 'wb') as outfile:
            pickle.dump(self.post_params, outfile)

    def summary(self, Output_all):
        Output = Output_all[self.burnin::self.thini, :]
        Output_theta = Output[:, :self.d]
        Energy = Output[self.burnin:, -1]
        return Output_theta
"""
for i in range(29):
    mcmc = mcmc_main(city="UCDavis (sludge)", per=i)
    mcmc.RunMCMC()
"""

#mcmc = mcmc_main(city='UCDavis (sludge)', per=0)
#city_data = mcmc.data_city_all_per
#print(city_data.Testing.sum())

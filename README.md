### Simulation codes for the paper:
## Bayesian sequential approach to monitor COVID-19 variants through positivity rate from wastewater

##### J. Cricelio Montesinos-López, Maria L. Daza–Torres, Yury E. García, César Herrera, C. Winston Bess, Marlene K. Wolfe, Alexandria B. Boehm, Heather N. Bischel, Miriam Nuño
This module consists in the following files:

Data
data_ww_cases.csv 

Codes
Example_Estimation of COVID-19  PR from ww data.ipynb
Notebook tutorial to estimate of PR  from ww data

run_mcmc.py
Main code:
	•	Set all the parameters for the models
	•	Load and processed data
	•	Likelihood, priors for mcmc are defined
	•	Compute Priors from the posterior distributions

plot_data.py
- plot_data()
To plot Normalized wastewater data (N/PMMoV), number of COVID-19 tests conducted, and positive cases.
- plot_conc_pos_rate()
To plot 7 day- trimmed average of WW data (Smoothed N/PMMoV) and daily PR.

save_mcmc.py
save_output()
This function is for running and saving the mcmc for each period. We just save the output.

plot_PR_predictions.py
- plot_beta()
To plot the posterior distribution of the model’s parameters for a forecast period.
- plot_post()
To plot the estimated PR for a forecast period.
- plot_post_wave_beta()
To plot the posterior distribution of the model’s parameters for all periods.
- plot_post_wave()
To plot the estimated PR for all periods.

plot_estimated_Rt.py
- plot_Rt()	
To plot the Rt from estimated PR using the algorithm of Cori et al 2013.
- plot_all_Rt()

To plot  the Rt for all periods.

threshold.py
- FindClosestIndex()
Find the index of the closest item (time, array) in tt to d[i] (ops. time)
- find_thresholds()
Find the value of WW concentration, c, such that with probability alpha the positivity rate is less than or equal to the CDC threshold.
plot_thresholds()
To plot computed thresholds for all periods.

Auxiliary program:
pytwalk.py
Library for the t-walk MCMC algorithm. For more details about this library see https://www.cimat.mx/~jac/twalk/
Cori, A., Ferguson, N. M., Fraser, C., & Cauchemez, S. (2013). A new framework and software to estimate time-varying reproduction numbers during epidemics. American journal of epidemiology, 178(9), 1505-1512.


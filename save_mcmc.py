from run_mcmc import mcmc_main
from pytwalk import pytwalk
import pandas as pd



workdir = "./"

def save_output(city,per):
  workdir = "./"
  mcmc= mcmc_main(city=city)
  LG_twalk = pytwalk(n=mcmc.d, U=mcmc.Energy, Supp=mcmc.Supp)
  LG_twalk.Run(T=50000, x0=mcmc.LG_Init(), xp0=mcmc.LG_Init())
  Output_df = pd.DataFrame(LG_twalk.Output, columns=['b0', 'b1','b2', 'Energy'])
  Output_df.to_csv(workdir + 'output/' + 'mcmc_' + mcmc.city+ 'per_' + str(per))
  mcmc.summary(LG_twalk.Output)

#city = 'UCDavis (sludge)'
city = 'Davis (sludge)'


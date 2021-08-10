import pandas as pd
import numpy as np

print('Start!')
'''
1. Load the dataset in R (Filename: JOBSI HR.dta)
'''
data = pd.read_csv('JOBSII_HR.csv', index_col=None)
data.pop('income')
'''
 2. For each variable, calculate the mean for the whole sample and within
each treatment group. For continuous covariates, also report the me-
dians, standard deviation and ranges within each treatment group.
Record your results in a table. In a few sentences, comment on what
you see and whether it is expected.
'''
df = pd.DataFrame(data = {}, index = ['mean','median','std','range'])
for d in data.columns:

    df[d] = float('nan')
    df.loc['mean', d] = data[d].mean()
    df.loc['median', d] = data[d].median()
    df.loc['std', d] = data[d].std()
    df.loc['range', d] = data[d].max()-data[d].min()
    for treat in [0,1]:
        col_name = '{}_{}'.format(d,treat)
        df[col_name] = float('nan')
        data_sel = data.loc[data['Z'].values==treat,:]
        df.loc['mean', col_name] = data_sel[d].mean()
        df.loc['median', col_name] = data_sel[d].median()
        df.loc['std', col_name] = data_sel[d].std()
        df.loc['range', col_name] = data_sel[d].max()-data[d].min()
df.to_csv('point_2.csv')

'''
3. Bayesian model-based analysis. For each scenario described below derive the posterior distributions of the finite sample average causal effect
and the super-population average causal effect. Plot the resulting posterior distributions in a histogram and report the following summary
statistics of the resulting posterior distributions: mean, standard deviation, median, 2.5% and 97.5% percentiles.
'''

from utils import *
print('Start!')
data_path = r'C:\Users\peppu\Documents\MyPythonProject\BayesianInference\Assignment_Mattei\data/'
filename = data_path + '/JOBSII_HR.csv'
data = pd.read_csv(filename)

treat_var = 'Z'
causal_var = 'depress6'
par_prior = dict(nu_c=0, omega2_c=100^2, nu_t=0, omega2_t=100^2, a_c=2, b2_c=0.01, a_t=2, b2_t=0.01)

chain_mA = model_3_a_i(niter=5000, nburn=100, thin=1,  par_prior=par_prior, Yobs=data[causal_var], W=data[treat_var],
                     seed=2021, theta_start=None, save_results=True)
print(chain_mA['Estimands'].describe())
print(chain_mA['Theta'].describe())
print('End!')



print('End!')
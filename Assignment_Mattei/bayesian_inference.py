import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from utils import *
import copy
from numpy.random import beta as beta
from numpy.random import binomial

def bern(p, size=None):
    return binomial(n=1,p=p,size=size)

print('Start!')
'''
1. Load the dataset in R (Filename: JOBSI HR.dta)
'''
data_path = 'C:/Users/peppu/Documents/MyPythonProject/BayesianInference/Assignment_Mattei'
filename = data_path + '/data/JOBSII_HR.csv'
data_orig = pd.read_csv(filename)
data_orig.pop('income')
'''
 2. For each variable, calculate the mean for the whole sample and within
each treatment group. For continuous covariates, also report the me-
dians, standard deviation and ranges within each treatment group.
Record your results in a table. In a few sentences, comment on what
you see and whether it is expected.
'''
print('Start 2!')
data = data_orig.copy(deep=True)
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
df.to_csv(data_path + '/SavedOutput/point_2.csv')
print('End 2!')

'''
3. Bayesian model-based analysis. For each scenario described below derive the posterior distributions of the finite sample average causal effect
and the super-population average causal effect. Plot the resulting posterior distributions in a histogram and report the following summary
statistics of the resulting posterior distributions: mean, standard deviation, median, 2.5% and 97.5% percentiles.
'''

######################################################################################################
# 3 a i #
######################################################################################################
load_saved_model = True
save_load_folder = 'SavedOutput'
data = data_orig.copy(deep=True)
print('Start 3 - a - i!')
treat_var = 'Z'
causal_var = 'depress6'
par_prior = dict(nu_c=0, omega2_c=100^2, nu_t=0, omega2_t=100^2, a_c=2, b2_c=0.01, a_t=2, b2_t=0.01)
if load_saved_model:
    with open('{}/{}/model_3_a_i.json'.format(data_path,save_load_folder),'r') as fid:
        chain_m_3_a_i = js.load(fid)
    chain_m_3_a_i['Estimands'] = pd.DataFrame(chain_m_3_a_i['Estimands'])
    chain_m_3_a_i['Theta'] = pd.DataFrame(chain_m_3_a_i['Theta'])
else:
    chain_m_3_a_i = model_3_a_i_ii(niter=50000, nburn=5000, thin=1,  par_prior=par_prior, Yobs=data[causal_var],W=data[treat_var],
                                    seed=2021, theta_start=None, save_results=True)
    with open('{}/{}/model_3_a_i.json'.format(data_path,save_load_folder),'w') as fid:
        out_json = {}
        out_json['Estimands'] = chain_m_3_a_i['Estimands'].to_dict()
        out_json['Theta'] = chain_m_3_a_i['Theta'].to_dict()
        js.dump(out_json, fid, indent=4)
print(chain_m_3_a_i['Estimands'].describe())
print(chain_m_3_a_i['Theta'].describe())
chain_m_3_a_i['Estimands'].describe([0.025,0.5,0.975]).to_csv('{}/{}/point_3_a_i.csv'.format(data_path,save_load_folder))

f, ax = plt.subplots(2,1,sharex=True,figsize=(9,6))
merged_dist = np.hstack((chain_m_3_a_i['Estimands']['ate_fs'],chain_m_3_a_i['Estimands']['ate_sp'])); merged_dist.sort()
bin_count,bins = np.histogram(merged_dist,40)
ax[0].hist(chain_m_3_a_i['Estimands']['ate_fs'], bins=bins)
ax[0].axvline(x=chain_m_3_a_i['Estimands']['ate_fs'].mean(), ymax=bin_count.max(), ymin=0, color='r')
ax[0].set_ylabel('Bin count ATE_FS')
ax[1].hist(chain_m_3_a_i['Estimands']['ate_sp'], bins=bins)
ax[1].axvline(x=chain_m_3_a_i['Estimands']['ate_fs'].mean(), ymax=bin_count.max(), ymin=0, color='r')
ax[1].set_ylabel('Bin count ATE_SP')
ax[1].set_xlabel('Average Treatment Effect')
for a in ax: a.grid();
plt.show(block=False)
f.savefig('{}/{}/point_3_a_i.png'.format(data_path,save_load_folder))
print('End 3 - a - i!')
######################################################################################################
######################################################################################################

######################################################################################################
# 3 a ii #
######################################################################################################
load_saved_model = True
save_load_folder = 'SavedOutput'
print('Start 3 - a - ii!')
df = data_orig.copy(deep=True)
causal_var = 'depress6'
df['y_log'] = np.log(data[causal_var])
treat_var = 'Z'
causal_var = 'y_log'
par_prior = dict(nu_c=0, omega2_c=100^2, nu_t=0, omega2_t=100^2, a_c=2, b2_c=0.01, a_t=2, b2_t=0.01)
if load_saved_model:
    with open('{}/{}/model_3_a_ii.json'.format(data_path,save_load_folder),'r') as fid:
        chain_m_3_a_ii = js.load(fid)
    chain_m_3_a_ii['Estimands'] = pd.DataFrame(chain_m_3_a_ii['Estimands'])
    chain_m_3_a_ii['Theta'] = pd.DataFrame(chain_m_3_a_ii['Theta'])
else:
    chain_m_3_a_ii = model_3_a_i_ii(niter=50000, nburn=5000, thin=1,  par_prior=par_prior, Yobs=df[causal_var],W=df[treat_var],
                                 y_log=True,
                                 seed=2021, theta_start=None, save_results=True)
    with open('{}/{}/model_3_a_ii.json'.format(data_path,save_load_folder),'w') as fid:
        out_json = {}
        out_json['Estimands'] = chain_m_3_a_ii['Estimands'].to_dict()
        out_json['Theta'] = chain_m_3_a_ii['Theta'].to_dict()
        js.dump(out_json, fid, indent=4)

print(chain_m_3_a_ii['Estimands'].describe())
print(chain_m_3_a_ii['Theta'].describe())
chain_m_3_a_ii['Estimands'].describe([0.025,0.5,0.975]).to_csv('{}/{}/point_3_a_ii.csv'.format(data_path,save_load_folder))

f, ax = plt.subplots(2,1,sharex=True,figsize=(9,6))
merged_dist = np.hstack((chain_m_3_a_ii['Estimands']['ate_fs'],chain_m_3_a_ii['Estimands']['ate_sp'])); merged_dist.sort()
bin_count,bins = np.histogram(merged_dist,40)
ax[0].hist(chain_m_3_a_ii['Estimands']['ate_fs'], bins=bins)
ax[0].axvline(x=chain_m_3_a_ii['Estimands']['ate_fs'].mean(), ymax=bin_count.max(), ymin=0, color='r')
ax[0].set_ylabel('Bin count ATE_FS')
ax[1].hist(chain_m_3_a_ii['Estimands']['ate_sp'], bins=bins)
ax[1].axvline(x=chain_m_3_a_ii['Estimands']['ate_fs'].mean(), ymax=bin_count.max(), ymin=0, color='r')
ax[1].set_ylabel('Bin count ATE_SP')
ax[1].set_xlabel('Average Treatment Effect')
for a in ax: a.grid();
plt.show(block=False)
f.savefig('{}/{}/point_3_a_ii.png'.format(data_path,save_load_folder))
print('End 3 - a - ii!')
######################################################################################################
######################################################################################################

######################################################################################################
# 3 a ii #
######################################################################################################
print('Start 3 - b!')
data = data_orig.copy(deep=True)
treat_var = 'Z'
causal_var = 'employ6'
save_load_folder = 'SavedOutput'

Nc = np.sum(data[treat_var].values == 0)
Nt = np.sum(data[treat_var].values == 1)

# Calculate the starting probability of the Bernoulli distribution
pi_c = data.loc[data[treat_var].values==0,causal_var].sum()/np.sum(data[treat_var].values==0)
pi_t = data.loc[data[treat_var].values==1,causal_var].sum()/np.sum(data[treat_var].values==1)

# Prior distribution of pi_c and pi_t
a_c, a_t = 1, 1
b_c, b_t = 1, 1

# Posterior distribution of the model parameters
a_c_obs = a_c + data.loc[data[treat_var].values==0,causal_var].sum()
b_c_obs = b_c + (1-data.loc[data[treat_var].values==0,causal_var]).sum()
a_t_obs = a_t + data.loc[data[treat_var].values==1,causal_var].sum()
b_t_obs = b_t + (1-data.loc[data[treat_var].values==1,causal_var]).sum()

# Simulate
np.random.seed(2021)
n_iter = 50000
ate_fs = []  # ATE of finite sample
ate_sp = []  # ATE of super population
for i in range(0,n_iter):

    pi_c_sim = beta(a_c_obs, b_c_obs)  # Sample from psoterioe distribution of the controls
    pi_t_sim = beta(a_t_obs, b_t_obs)  # Sample from psoterioe distribution of the treated

    #Imputation of the missing potential outcomes
    y1 , y0 = np.zeros(data.shape[0])*float('nan'), np.zeros(data.shape[0])*float('nan')

    # Imputation of missing data for treated
    y1[data[treat_var].values == 1] = data.loc[data[treat_var].values==1,causal_var]  # y1 treated are observed
    y1[data[treat_var].values == 0] = bern(pi_t_sim, size = Nc)  # y1 non-treated are imputed

    # Imputation of missing data for nontreated
    y0[data[treat_var].values == 0] = data.loc[data[treat_var].values==0,causal_var]  # y0 non-treated are observed
    y0[data[treat_var].values == 1] = bern(pi_c_sim, size = Nt)  # y0 treated are imputed

    # Compute the ATEs
    ate_fs.append(np.mean(y1)-np.mean(y0))  # average treatment effect of finite sample
    ate_sp.append(pi_t_sim - pi_c_sim)  # average treatment effect of super population

out_ate = pd.DataFrame(data={'ate_fs':ate_fs,'ate_sp':ate_sp})
out_ate.describe([0.025,0.5,0.975]).to_csv('{}/{}/point_3_b.csv'.format(data_path,save_load_folder))
print(out_ate.describe())

f, ax = plt.subplots(2,1,sharex=True,figsize=(9,6))
merged_dist = np.hstack((out_ate['ate_fs'],out_ate['ate_sp'])); merged_dist.sort()
bin_count,bins = np.histogram(merged_dist,40)
ax[0].hist(out_ate['ate_fs'], bins=bins)
ax[0].axvline(x=out_ate['ate_fs'].mean(), ymax=bin_count.max(), ymin=0, color='r')
ax[0].set_ylabel('Bin count ATE_FS')
ax[1].hist(out_ate['ate_sp'], bins=bins)
ax[1].axvline(x=out_ate['ate_fs'].mean(), ymax=bin_count.max(), ymin=0, color='r')
ax[1].set_ylabel('Bin count ATE_SP')
ax[1].set_xlabel('Average Treatment Effect')
for a in ax: a.grid();
f.savefig('{}/{}/point_3_b.png'.format(data_path,save_load_folder))
print('End 3 - b!')
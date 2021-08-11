import pymc3 as pm
import pandas as pd
data_path = 'C:/Users/peppu/Documents/MyPythonProject/BayesianInference/Assignment_Mattei'
filename = data_path + '/data/JOBSII_HR.csv'
data_orig = pd.read_csv(filename)

Y = data_orig.loc[data_orig['Z']==1,'depress6'].values

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    mu_c = pm.Normal("mu_c", mu=0, sigma=100)
    sig_c = pm.InverseGamma("sig_c", alpha=2, beta=0.01)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Lognormal("Y_obs", mu=mu_c, sigma=sig_c, observed=Y)

map_estimate_t = pm.find_MAP(model=basic_model)
########################################################################################################################
data = data_orig.copy(deep=True)
# Prior distribution of pi_c and pi_t
a_c, a_t = 114, 128
b_c, b_t = 57, 152

# Posterior distribution of the model parameters
a_c_obs = a_c + data.loc[data[treat_var].values==0,causal_var].sum()
b_c_obs = b_c + (1-data.loc[data[treat_var].values==0,causal_var]).sum()
a_t_obs = a_t + data.loc[data[treat_var].values==1,causal_var].sum()
b_t_obs = b_t + (1-data.loc[data[treat_var].values==1,causal_var]).sum()

# Simulate
np.random.seed(2021)
n_iter = 20000
ate_fs = []  # ATE of finite sample
ate_sp = []  # ATE of super population
theta = {'pi_t':[], 'pi_c':[]}
for i in range(0,n_iter):

    pi_c_sim = beta(a_c_obs, b_c_obs)  # Sample from posterior distribution of the controls
    pi_t_sim = beta(a_t_obs, b_t_obs)  # Sample from posterioe distribution of the treated
    # Store probability of the binomial into a dictionary
    theta['pi_c'].append(pi_c_sim)
    theta['pi_t'].append(pi_t_sim)

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

Yc = data.loc[data[treat_var].values==0,'employ6'].values
Yt = data.loc[data[treat_var].values==1,'employ6'].values

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    pi = pm.Beta("pi", alpha=a_c, beta =b_c)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Binomial("Y_obs", p=pi, observed=Yc, n=1)

map_estimate_c = pm.find_MAP(model=basic_model)
basic_model_2 = pm.Model()

with basic_model_2:

    # Priors for unknown model parameters
    pi = pm.Beta("pi", alpha=a_t, beta =b_t)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Binomial("Y_obs", p=pi, observed=Yt, n=1)

map_estimate_t = pm.find_MAP(model=basic_model_2)

print(map_estimate_t['pi'] , pd.DataFrame(theta)['pi_t'].mean())
print(map_estimate_c['pi'] , pd.DataFrame(theta)['pi_c'].mean())
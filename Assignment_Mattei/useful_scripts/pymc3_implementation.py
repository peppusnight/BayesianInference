import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az
data_path = 'C:/Users/peppu/Documents/MyPythonProject/BayesianInference/Assignment_Mattei'
filename = data_path + '/data/JOBSII_HR.csv'
data_orig = pd.read_csv(filename)

#
# ######################################################################################################
# # 3 a i
# ######################################################################################################
# Y1 = data_orig.loc[data_orig['Z']==1,'depress6'].values
# Y0 = data_orig.loc[data_orig['Z']==0,'depress6'].values
#
#
# basic_model = pm.Model()
#
# with basic_model:
#
#     # Priors for unknown model parameters
#     mu_t = pm.Normal("mu_t", mu=0, sigma=100)
#     sig_t = pm.InverseGamma("sig_t", alpha=2, beta=0.01)
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs = pm.Normal("Y_obs", mu=mu_t, sigma=sig_t, observed=Y1)
#
#     # Priors for unknown model parameters
#     mu_c = pm.Normal("mu_c", mu=0, sigma=100)
#     sig_c = pm.InverseGamma("sig_c", alpha=2, beta=0.01)
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs_2 = pm.Normal("Y_obs_2", mu=mu_c, sigma=sig_c, observed=Y0)
#
#     start = pm.find_MAP()
#     trace = pm.sample(5000, cores=1, tune=500, start=start, progressbar=False, return_inferencedata=True)
#     # trace = pm.sample(5000, tune=1000, progressbar=False, return_inferencedata=True)
#     # trace = pm.sample()
#
# # map_estimate_t = pm.find_MAP(model=basic_model)
#
# az.summary(trace).to_csv('3_a_i.csv')
# ######################################################################################################
# ######################################################################################################
#
# ######################################################################################################
# # 3 a ii
# ######################################################################################################
# Y1 = data_orig.loc[data_orig['Z']==1,'depress6'].values
# Y0 = data_orig.loc[data_orig['Z']==0,'depress6'].values
#
# basic_model = pm.Model()
#
# with basic_model:
#
#     # Priors for unknown model parameters
#     mu_t = pm.Normal("mu_t", mu=0, sigma=100)
#     sig_t = pm.InverseGamma("sig_t", alpha=2, beta=0.01)
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs = pm.Lognormal("Y_obs", mu=mu_t, sigma=sig_t, observed=Y1)
#
#     # Priors for unknown model parameters
#     mu_c = pm.Normal("mu_c", mu=0, sigma=100)
#     sig_c = pm.InverseGamma("sig_c", alpha=2, beta=0.01)
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs_2 = pm.Lognormal("Y_obs_2", mu=mu_c, sigma=sig_c, observed=Y0)
#
#     start = pm.find_MAP()
#     trace = pm.sample(5000, cores=1, tune=500, start=start, progressbar=False, return_inferencedata=True)
#
# az.summary(trace).to_csv('3_a_ii.csv')
######################################################################################################
######################################################################################################

# ######################################################################################################
# # 3 b
# ######################################################################################################
# data = data_orig.copy(deep=True)
# treat_var = 'Z'
# causal_var = 'employ6'
# # Prior distribution of pi_c and pi_t
# a_c, a_t = 1, 1
# b_c, b_t = 1, 1
#
# Y0 = data.loc[data[treat_var].values==0,causal_var].values
# Y1 = data.loc[data[treat_var].values==1,causal_var].values
#
# basic_model = pm.Model()
#
# with basic_model:
#
#     # Priors for unknown model parameters
#     pi_c = pm.Beta("pi_c", alpha=a_c, beta = b_c)
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs = pm.Binomial("Y_obs", p=pi_c, observed=Y0, n=1)
#
#     # Priors for unknown model parameters
#     pi_t = pm.Beta("pi_t", alpha=a_t, beta =b_t)
#
#     # Likelihood (sampling distribution) of observations
#     Y_obs_2 = pm.Binomial("Y_obs_2", p=pi_t, observed=Y1, n=1)
#
#     start = pm.find_MAP()
#     trace = pm.sample(5000, cores=1, tune=500, start=start, progressbar=False, return_inferencedata=True)
#
# az.summary(trace).to_csv('3_b.csv')

######################################################################################################
# 3 c i
######################################################################################################
# Y = data_orig.loc[data_orig['Z']==1,'depress6'].values
#
# basic_model = pm.Model()
# ["sex","age","race","nonmarried","educ","EconHard","assertive","motivation"]
# with basic_model:
#
#     ## Define weakly informative Normal priors to give Ridge regression
#     b0 = pm.Normal("b0_intercept", mu=0, sigma=10)
#     b1 = pm.Normal("b1_sex", mu=0, sigma=10)
#     b2 = pm.Normal("b2_age", mu=0, sigma=10)
#     b3 = pm.Normal("b3_race", mu=0, sigma=10)
#     b4 = pm.Normal("b4_nonmarried", mu=0, sigma=10)
#     b5 = pm.Normal("b5_educ", mu=0, sigma=10)
#     b6 = pm.Normal("b6_EconHard", mu=0, sigma=10)
#     b7 = pm.Normal("b7_assertive", mu=0, sigma=10)
#     b8 = pm.Normal("b8_motivation", mu=0, sigma=10)
#
#     ## Define linear model
#     y_est = b0 + b1 * dfhoggs["x"]
#
#     start = pm.find_MAP()
#     trace = pm.sample(5000, cores=1, tune=500, start=start, progressbar=False, return_inferencedata=True)

# az.summary(trace).to_csv('3_c.csv')

X_obs_t = data_orig.loc[data_orig['Z']==1,["sex","age","race","nonmarried","educ","EconHard","assertive","motivation"]].values
nxx = X_obs_t.shape[1]  # Number of columns
Y_obs_t = data_orig.loc[data_orig['Z']==1,'depress6'].values

X_obs_c = data_orig.loc[data_orig['Z']==0,["sex","age","race","nonmarried","educ","EconHard","assertive","motivation"]].values
nxx = X_obs_c.shape[1]  # Number of columns
Y_obs_c = data_orig.loc[data_orig['Z']==0,'depress6'].values
with pm.Model() as linear_model:
    beta_t = pm.MvNormal("beta_t", mu=np.zeros(nxx), cov=np.eye(nxx)*100**2, shape=nxx)
    sigma_t = pm.InverseGamma("sigma_t", alpha=2, beta=0.01)
    y_observed_t = pm.Lognormal(
        "y_observed_t",
        mu=X_obs_t @ beta_t,
        sigma=sigma_t,
        observed=Y_obs_t,
    )

    beta_c = pm.MvNormal("beta_c", mu=np.zeros(nxx), cov=np.eye(nxx)*100**2, shape=nxx)
    sigma_c = pm.InverseGamma("sigma_c", alpha=2, beta=0.01)
    y_observed_c = pm.Lognormal(
        "y_observed_c",
        mu=X_obs_c @ beta_c,
        sigma=sigma_c,
        observed=Y_obs_c,
    )

    # prior = pm.sample_prior_predictive()
    # posterior = pm.sample(10, cores=1)
    # posterior_pred = pm.sample_posterior_predictive(posterior)
    start = pm.find_MAP()
    trace = pm.sample(200, cores=1, tune=20, start=start, progressbar=False, return_inferencedata=True)

az.summary(trace).to_csv('3_c.csv')

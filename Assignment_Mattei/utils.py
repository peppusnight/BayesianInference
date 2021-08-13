import pandas as pd
import numpy as np
import scipy
import simplejson as js
from numpy.random import normal as rnorm
from numpy.random import chisquare as rchisq
from numpy.random import multivariate_normal as rmvnorm
from sklearn.linear_model import LinearRegression
mat_inv = np.linalg.inv

def resiudual_standard_error(X, y, model):
    '''
    # lecture 2 Marchetti - "Distribution theory"
    https://gist.github.com/grisaitis/cf481034bb413a14d3ea851dab201d31
    :param y:
    :param X_with_intercept:
    :return:
    '''
    N = len(X)
    p = X.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = X

    y_hat = model.predict(X)
    residuals = y - y_hat
    residual_sum_of_squares = np.sum(residuals**2)

    sigma_squared_hat = residual_sum_of_squares / (N - p) # s2_res
    # var_beta_hat = mat_inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat # cov(beta)
    # standard_error_beta = np.sqrt(np.diag(var_beta_hat))
    return sigma_squared_hat

def model_3_a_i_ii(niter,  par_prior, Yobs, W, nburn=None, thin=1, seed=None, theta_start=None, y_log=False, save_results=False):
    '''

    :param niter:
    :param par_prior:
    :param Yobs:
    :param W:
    :param nburn:
    :param thin:
    :param seed:
    :param theta_start:
    :param y_log:
    :param save_results:
    :return:
    '''

    # Set the seed for the simulation
    if seed is not None:
        np.random.seed(seed)

    Nc = sum(1 - W) # Number of controls
    Nt = sum(W) # Number of treated
    N = Nt + Nc # Total number of cases

    yobs_c = np.mean(Yobs[W == 0]) # Average of the controls
    yobs_t = np.mean(Yobs[W == 1]) # Average of the treated

    ##Setup - MCMC
    nburn = niter/2 if nburn is None else nburn # Number of burn in

    draws = range((nburn + 1), niter+1, thin) # Number of draws

    if (max(draws) < niter):
        print("The number of iterations will be changed from {} to {}".format(niter, max(draws)+thin))
        niter = max(draws)+thin
        draws = range((nburn + 1), niter+1, thin)

    ndraws = len(draws) # Number of iterations
    j = -1  ##Counter j=1...ndraws

    # Parameter initialization
    if theta_start is None:
        theta = {'mu_c' : np.mean(Yobs[W == 0]) + rnorm(0, 0.1), # Y_c is normal with avg = mu_c
                 'mu_t' :  np.mean(Yobs[W == 1]) + rnorm(0, 0.1), # Y_t is normal with avg = mu_t
                 'sigma2_c' : np.var(Yobs[W == 0]) + rnorm(0, 0.1), # Y_c is normal with var = sigma2_c
                 'sigma2_t' : np.var(Yobs[W == 1]) + rnorm(0, 0.1) # Y_t is normal with var = sigma2_t
                 }
    else:
        theta = theta_start

    # Store posterior distribution of Y
    Theta = pd.DataFrame(data={'mu_c':[float('nan')]*ndraws,'mu_t':[float('nan')]*ndraws,
                               'sigma2_c':[float('nan')]*ndraws,'sigma2_t':[float('nan')]*ndraws})
    # Store ate_fs and ate-sp
    Estimands = pd.DataFrame(data={'ate_fs':[float('nan')]*ndraws,'ate_sp':[float('nan')]*ndraws})

    for ell in range(1,niter+1):

        # Gibbs resampling applied

        ##Update mu.c # Full conditional
        omega2_c_obs = 1 / (Nc / theta['sigma2_c'] + 1 / par_prior['omega2_c']) # Slide 34 - lesson short
        nu_c_obs = omega2_c_obs * ((yobs_c * Nc) / theta['sigma2_c'] +  par_prior['nu_c'] / par_prior['omega2_c']) # Slide 34 - lesson short
        theta['mu_c'] = rnorm(nu_c_obs, np.sqrt(omega2_c_obs))

        ##Update mu.t # Full conditional
        omega2_t_obs = 1 / (Nt / theta['sigma2_t'] + 1 / par_prior['omega2_t']) # Slide 34 - lesson short
        nu_t_obs = omega2_t_obs * ((yobs_t * Nt) / theta['sigma2_t'] +  par_prior['nu_t'] / par_prior['omega2_t']) # Slide 34 - lesson short
        theta['mu_t'] = rnorm(nu_t_obs, np.sqrt(omega2_t_obs))

        ##Update sigma2.c # Full conditional
        a_c_obs = Nc + par_prior['a_c'] # Slide 34 - lesson short
        b2_c_obs =  (par_prior['a_c'] * par_prior['b2_c'] + np.sum((Yobs[W == 0]-theta['mu_c']) ** 2)) / a_c_obs # Slide 34 - lesson short
        theta['sigma2_c'] =  (a_c_obs * b2_c_obs) / rchisq(a_c_obs)

        ##Update sigma2.t
        a_t_obs = Nt + par_prior['a_t'] # Slide 34 - lesson short
        b2_t_obs =  (par_prior['a_t'] * par_prior['b2_t'] + np.sum((Yobs[W == 1]-theta['mu_t']) ** 2)) / a_t_obs # Slide 34 - lesson short
        theta['sigma2_t'] =  (a_t_obs * b2_t_obs) / rchisq(a_t_obs)

        # rm(omega2.c.obs, nu.c.obs, omega2.t.obs, nu.t.obs, a.c.obs, b2.c.obs, a.t.obs, b2.t.obs)

        # Save only specified iterations
        if (ell in draws):
            j = j+1

            for k,v in theta.items():
                Theta.loc[j, k] = v


            # Impute the missing potential outcomes using Ymis | Yobs, W, X, theta
            Y0, Y1 = np.zeros(Yobs.shape[0])*float('nan'), np.zeros(Yobs.shape[0])*float('nan')

            Y0[W == 0] = Yobs[W == 0]
            Y0[W == 1] = rnorm(theta['mu_c'], np.sqrt(theta['sigma2_c']), Nt)

            Y1[W == 0] = rnorm(theta['mu_t'], np.sqrt(theta['sigma2_t']), Nc)
            Y1[W == 1] = Yobs[W == 1]

            if y_log==False:
                ##FINITE SAMPLE ATE
                Estimands.loc[j, "ate_fs"] = np.mean(Y1)-np.mean(Y0)

                ##SUPER-POPULATION ATE
                Estimands.loc[j, "ate_sp"] = theta['mu_t']-theta['mu_c']
            else:
                ##FINITE SAMPLE ATE
                Estimands.loc[j, "ate_fs"] = np.mean(np.exp(Y1))-np.mean(np.exp(Y0))

                ##SUPER-POPULATION ATE
                Estimands.loc[j, "ate_sp"] = np.exp(theta['mu_t']+0.5*theta['sigma2_t']**2)-np.exp(theta['mu_c']+0.5*theta['sigma2_c']**2)

    if save_results:
        model = dict(Theta=Theta.to_dict(), Estimands=Estimands.to_dict())
        with open('model_3_a_i.json','w') as fid:
            js.dump(model, fid, indent=4)

    return dict(Theta = Theta, Estimands = Estimands)

def model_3_c_i(niter,  par_prior, Yobs, W, X, nburn=None, thin=1, seed=None, theta_start=None, y_log=False):
    '''

    :param niter:
    :param par_prior:
    :param Yobs:
    :param W:
    :param nburn:
    :param thin:
    :param seed:
    :param theta_start:
    :param y_log:
    :return:
    '''

    # Set the seed for the simulation
    if seed is not None:
        np.random.seed(seed)

    Nc = sum(1 - W) # Number of controls
    Nt = sum(W) # Number of treated
    N = Nt + Nc # Total number of cases

    XX = np.empty(shape=(N, X.shape[1]+1), dtype=np.float)
    XX[:, 0] = 1
    XX[:, 1:X.shape[1]+1] = X # Design matrix with intercept
    nxx = XX.shape[1] # Number of columns

    X_c = X[W == 0, :] # Data for controls
    XX_c = XX[W==0, :] # Data for treated
    X_t = X[W == 1, :] # Data for controls
    XX_t = XX[W==1, :] # Data for treated

    ##Setup - MCMC
    nburn = niter/2 if nburn is None else nburn # Number of burn in

    draws = range((nburn + 1), niter+1, thin) # Number of draws

    if (max(draws) < niter):
        print("The number of iterations will be changed from {} to {}".format(niter, max(draws)+thin))
        niter = max(draws)+thin
        draws = range((nburn + 1), niter+1, thin)

    ndraws = len(draws) # Number of iterations
    j = -1  ##Counter j=1...ndraws

    # Parameter initialization
    if theta_start is None:
        lm_c = LinearRegression().fit(X_c, Yobs[W==0])
        lm_t = LinearRegression().fit(X_t, Yobs[W==1])
        theta = {'beta_c' : np.hstack((lm_c.intercept_, lm_c.coef_)) + rnorm(0, 0.1, nxx), # Y_c is normal with avg = beta_c * X (beta_c contains the intercept)
                 'beta_t' : np.hstack((lm_t.intercept_, lm_t.coef_)) + rnorm(0, 0.1, nxx), # Y_t is normal with avg = beta_t * X (beta_t contains the intercept)
                 'sigma2_c' : resiudual_standard_error(X=X_c, y= Yobs[W==0],model=lm_c)  + rnorm(0, 1), # Y_c is normal with var = sigma2_c
                 'sigma2_t': resiudual_standard_error(X=X_t, y=Yobs[W==1], model=lm_t) + rnorm(0, 1) # Y_t is normal with var = sigma2_c
                 }
    else:
        theta = theta_start

    # Store posterior distribution of Y
    Theta = pd.DataFrame(data={'beta_c':[float('nan')]*ndraws,'beta_t':[float('nan')]*ndraws,
                               'sigma2_c':[float('nan')]*ndraws,'sigma2_t':[float('nan')]*ndraws})
    # Store ate_fs and ate-sp
    Estimands = pd.DataFrame(data={'ate_fs':[float('nan')]*ndraws,'ate_sp':[float('nan')]*ndraws})

    for ell in range(1,niter+1):

        # Gibbs resampling applied

        ##Update beta_c # Full conditional
        Omega_c_obs = mat_inv(mat_inv(par_prior['Omega_c']) + XX_c.T@XX_c/theta['sigma2_c']) # Slide 37 - lesson short
        nu_c_obs = Omega_c_obs @ (mat_inv(par_prior['Omega_c'])@par_prior['nu_c'] + XX_c.T@Yobs[W==0]/theta['sigma2_c']) # Slide 37 - lesson short
        theta['beta_c'] = rmvnorm(mean = nu_c_obs, cov = Omega_c_obs)

        ##Update beta_t # Full conditional
        Omega_t_obs = mat_inv(mat_inv(par_prior['Omega_t']) + XX_t.T@XX_t/theta['sigma2_t']) # Slide 37 - lesson short
        nu_t_obs = Omega_t_obs @ (mat_inv(par_prior['Omega_t'])@par_prior['nu_t'] + XX_t.T@Yobs[W==1]/theta['sigma2_t']) # Slide 37 - lesson short
        theta['beta_t'] = rmvnorm(mean = nu_t_obs, cov = Omega_t_obs)

        ##Update sigma2_c # Full conditional
        a_c_obs = Nc + par_prior['a_c'] # Slide 37 - lesson short
        b2_c_obs =  (par_prior['a_c'] * par_prior['b2_c'] + np.sum((Yobs[W == 0] - XX_c @ theta['beta_c']) ** 2)) / a_c_obs # Slide 37 - lesson short
        theta['sigma2_c'] =  (a_c_obs * b2_c_obs) / rchisq(a_c_obs)

        ##Update sigma2_t
        a_t_obs = Nt + par_prior['a_t'] # Slide 37 - lesson short
        b2_t_obs =  (par_prior['a_t'] * par_prior['b2_t'] + np.sum((Yobs[W == 1] - XX_t @ theta['beta_t']) ** 2)) / a_t_obs # Slide 37 - lesson short
        theta['sigma2_t'] =  (a_t_obs * b2_t_obs) / rchisq(a_t_obs)

        # rm(omega2.c.obs, nu.c.obs, omega2.t.obs, nu.t.obs, a.c.obs, b2.c.obs, a.t.obs, b2.t.obs)

        # Save only specified iterations
        if (ell in draws):
            j = j+1

            for k,v in theta.items():
                Theta.loc[j, k] = js.dumps(v.tolist()) if isinstance(v,np.ndarray) else v

            # Impute the missing potential outcomes using Ymis | Yobs, W, X, theta
            Y0, Y1 = np.zeros(Yobs.shape[0])*float('nan'), np.zeros(Yobs.shape[0])*float('nan')

            Y0[W == 0] = Yobs[W == 0]
            Y0[W == 1] = rnorm(loc = XX_t@theta['beta_c'], scale = np.sqrt(theta['sigma2_c']), size=Nt)

            Y1[W == 0] = rnorm(loc = XX_c@theta['beta_t'], scale = np.sqrt(theta['sigma2_t']), size=Nc)
            Y1[W == 1] = Yobs[W == 1]

            if y_log==False:
                ##FINITE SAMPLE ATE
                Estimands.loc[j, "ate_fs"] = np.mean(Y1)-np.mean(Y0)

                ##SUPER-POPULATION ATE
                Estimands.loc[j, "ate_sp"] = np.mean(XX@theta['beta_t']-XX@theta['beta_c'])
            else:
                ##FINITE SAMPLE ATE
                Estimands.loc[j, "ate_fs"] = np.mean(np.exp(Y1))-np.mean(np.exp(Y0))

                ##SUPER-POPULATION ATE
                Estimands.loc[j, "ate_sp"] = np.mean(np.exp(XX@theta['beta_t']+0.5*theta['sigma2_t']**2)-np.exp(XX@theta['beta_c']+0.5*theta['sigma2_c']**2))

    return dict(Theta = Theta, Estimands = Estimands)
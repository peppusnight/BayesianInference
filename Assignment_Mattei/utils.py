import pandas as pd
import numpy as np
import scipy
import simplejson as js
from numpy.random import normal as rnorm
from numpy.random import chisquare as rchisq

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


            ##Imputate the missing potential outcomes using Ymis | Yobs, W, X, theta
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

def model_3_b(niter,  par_prior, Yobs, W, nburn=None, thin=1, seed=None, theta_start=None, y_log=False, save_results=False):
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
        theta = {'pi_c' : len(Yobs[W == 0])/len(Yobs) + rnorm(0, 0.01), # Y_c is bernoulli with p = pi_c
                 'pi_t' : len(Yobs[W == 1])/len(Yobs) + rnorm(0, 0.01), # Y_t is bernoulli with p = pi_t
                 }
    else:
        theta = theta_start

    # Store posterior distribution of Y
    Theta = pd.DataFrame(data={'pi_c':[float('nan')]*ndraws,'pi_t':[float('nan')]*ndraws})
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


            ##Imputate the missing potential outcomes using Ymis | Yobs, W, X, theta
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

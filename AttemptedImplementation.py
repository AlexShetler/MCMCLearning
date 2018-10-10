# -*- coding: utf-8 -*-

"""

Created on Tue Oct  2 23:26:39 2018



@author: AlexS

"""

import emcee

import numpy as np

import matplotlib.pyplot as plt

np.random.seed(123)


#Defines what we are looking for
def fn_predict(theta,x):

    a, b, c, d, q = theta #6 Dimmensions 
    
    #Returns model that data is predicting parameters for
    return  a+ b * x + c * x **2 +  d* np.sin(q *x) 



# Choose the "true" parameters.

m_true = 2

b_true = .1

amp_true = 0.05 

f_true = 0.534

pi_true = 3.14159

theta_true = [0, m_true, 0, amp_true,  pi_true]

N = 50

n = .05 * np.random.normal(loc=0.0,size=N)

# Generate some synthetic data from the model.

x = np.sort(10*np.random.rand(N))

yerr = 0.1*np.ones(N)

#True Model
y = m_true*x+b_true * np.sin(pi_true*x) + n




#Setup the matlab plot to see points in comparison to the truth of the function
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)

x0 = np.linspace(0, 5, num=1000)

plt.plot(x0, m_true*x0+b_true*np.sin(pi_true * x0), "k", alpha=0.3, lw=3)

plt.xlim(0, 10)

plt.xlabel("x")

plt.ylabel("y");

plt.savefig('input.png')



    


#Sets up the likelihood function
def log_likelihood(theta, x, y, yerr):

    model = fn_predict(theta,x)

    sigma2 = yerr**2 #+ model**2*np.exp(2*lnf)

    return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))


#Minimize function
from scipy.optimize import minimize

np.random.seed(42)

nll = lambda *args: -log_likelihood(*args)

initial = np.array([m_true, b_true,   pi_true,0,0]) + 0.1*np.random.randn(5)

soln = minimize(nll, initial, args=(x, y, yerr))

a_ml, b_ml,  q_ml, d_ml, c_ml = soln.x



print("Maximum likelihood estimates:")

print("a = {0:.3f}".format(a_ml))

print("b = {0:.3f}".format(b_ml))

print("c = {0:.3f}".format(c_ml))

print("d = {0:.3f}".format(d_ml))

print("q = {0:.3f}".format(q_ml))




plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)

plt.plot(x0, m_true*x0+b_true*np.sin(pi_true * x0), "k", alpha=0.3, lw=3, label="truth")

plt.plot(x0, fn_predict( [a_ml, b_ml, q_ml, d_ml, c_ml], x0) )

plt.legend(fontsize=14)

plt.xlim(0, 10)

plt.xlabel("x")

plt.ylabel("y");

plt.savefig('tmp.png')


#Define the priors per parameter. Priors set up where the rough values of the parameters are
def log_prior(theta):

    a, b, c, d, q = theta

    if -10.0 < a < 5.5 and -10.0 < b < 10.0  and -10.0<c<10.0 and -10.0<d<10.0 and -10.0<q<10.0:

        return 0.0

    return -np.inf


#Probability of finding the parameters with the priors 
def log_probability(theta, x, y, yerr):

    lp = log_prior(theta)

    if not np.isfinite(lp):

        return -np.inf

    return lp + log_likelihood(theta, x, y, yerr)


#Set up the number of walkers and dimmensions. Walkers = Dots that "feel out" the true values. Dimmensions means number of parameters
ndim, nwalkers = len(theta_true), 1000

pos = [soln.x + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


#MCMC Part of this implementation
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))

sampler.run_mcmc(pos, 1000);


#Set up the walker step count so you can see where each walker was when
fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)

samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

labels = ["a", "b",  "q", "c", "d"]

for i in range(ndim):

    ax = axes[i]

    ax.plot(samples[:, i], "k", alpha=0.3)

    ax.set_xlim(0, len(samples))

    ax.set_ylabel(labels[i])

    ax.yaxis.set_label_coords(-0.1, 0.5)



axes[-1].set_xlabel("step number");


#Set up the corner plot which will help you analyize the data from the MCMC
import corner

fig = corner.corner(samples, labels=["$a$", "$b$", "$c$" , "$d$", "q"],truths=theta_true)

fig.savefig("triangle.png")




#"Pretty print" for the results
from IPython.display import display, Math


for i in range(ndim):

    mcmc = np.percentile(samples[:, i], [16, 50, 84])

    q = np.diff(mcmc)

    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"

    txt = txt.format(mcmc[1], q[0], q[1], labels[i])

    display(Math(txt))


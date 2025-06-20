#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:38:46 2025

@author: thomas
"""

import numpy as np
from numpy.random import Generator, PCG64
import random
import matplotlib.pyplot as plt
import code

rg = Generator(PCG64())

class BernoulliBandit:
    
    def __init__(self, p, verbose=True):
        self.p = p
        if verbose:
            print("Creating BernoulliBandit with p = {:.2f}".format(p))
    
    def pull(self):
        return np.random.binomial(1, self.p)
    
class NormalBandit:
    
    def __init__(self, mu, sigma, verbose=True):
        self.mu = mu
        self.sigma = sigma
        if verbose:print("Creating NormalBandit with mu = {:.2f}".format(mu))
        # Setting actual reward values from normal distribution with K bandits
    
    def pull(self):
        return rg.normal(self.mu, self.sigma)
    
class BanditsGame:
    
    def __init__(self, K, T, epsilon, verbose=True):
        
        mu, sigma = 0, 1.0 # mean and standard deviation
        self.mu_values = rg.normal(mu,sigma,K)                              
        
        self.T = T
        self.K = K
        self.bandits = [NormalBandit(rg.normal(self.mu_values[i]), sigma, verbose) for i in range(K)]
        self.estimates = np.zeros(K)
        self.pulls = np.zeros(K) + 1
        self.verbose = verbose
        self.epsilon = epsilon

    def run_stochastic(self):
        
        results = np.zeros((self.T))
        
        
        for t in range(self.T):
            
            #k = random.randrange(self.K)
            if np.random.uniform() < 1.0-self.epsilon:
                k = self.estimates.max()
                k_index = self.estimates.argmax()
            else:
                k_index = random.randrange(self.K)
            results[t] = self.bandits[k_index].pull()
            if self.verbose:
                print("T={} \t Playing bandit {} \t Reward is {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f}".format(t, k_index, results[t], self.estimates[0], self.estimates[1] , self.estimates[2], self.estimates[3], self.estimates[4] , self.estimates[5], self.estimates[6], self.estimates[7] , self.estimates[8], self.estimates[9]))
            self.estimates[k_index] = self.estimates[k_index] + 1/self.pulls[k_index]*(results[t] - self.estimates[k_index])
            self.pulls[k_index] += 1
            
            
        return results
    
def run_simulation(n_runs, runs_per_game, K, T, e):
    
    results = np.zeros((K,T))
    
    for run in range(n_runs):

        #run_results = np.zeros((K,T))
        run_results = np.zeros(T)

        for run in range(runs_per_game):
            game = BanditsGame(K=K, T=T, epsilon=e, verbose=False)
            run_results += game.run_stochastic()
            #code.interact(banner="inner_loop")

        results += run_results / runs_per_game
        #code.interact(banner="outer_loop")
    results = results / n_runs
    
    return results
    
#game = BanditsGame(K=10, T=20)
#game.run_stochastic()
# plotting bandit distributions
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
plt.ylabel("Reward Distribution")
plt.xlabel("Action")
axs.set_title('10-armed Bandit')
for epsilon in [0,.05,.1]:
    stochastic_results = run_simulation(n_runs=1, runs_per_game=1000, K=10, T=2000, e=epsilon)
    stochastic_results = stochastic_results.mean(axis=0)
    axs.plot(stochastic_results)
    print("Mean reward: {:.2f}".format(stochastic_results.mean()))
    print("G: {:.2f}".format(stochastic_results.sum()))
plt.show


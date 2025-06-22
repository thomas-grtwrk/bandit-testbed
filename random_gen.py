#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:53:54 2025

@author: thomas
"""

import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import random
import math
import code

rg = Generator(PCG64())
all_runs = []
all_optimal_actions = []
all_optimal_bandit_options = []

epsilon = .1
set_optimistic_values = 0
gradient_bandit = 0
drift=1
alpha = .1

def choose_max(choices):
    max_indexs = np.argwhere(choices == np.amax(choices))
    max_indexs = max_indexs.flatten().tolist()
    number_of_max = len(max_indexs)
    #code.interact(banner="in choose max",local=locals())
    if number_of_max == 1:
        return max_indexs[0]
    else:
        return max_indexs[random.randrange(number_of_max)]
    
def create_drift(steps, reversion=True):
    
    mu, sigma = 0, 1.0 # mean and standard deviation
    initial_reward_values = rg.normal(mu,sigma,10)
    initial_reward_values = np.array(initial_reward_values)
    # TODO change optimal bandit according to drift
    optimal_bandit = initial_reward_values.argmax()
    
    bandit_drifts = [rg.normal(0,.01**2,steps) for i in np.arange(0,10,1)]
    bandit_drifts = np.array(bandit_drifts)
    
    bandit_drifts[:,0] = bandit_drifts[:,0] + initial_reward_values
    if reversion:
        #print("True")
        for step in np.arange(1,steps,1):
            bandit_drifts[:,step] = .5 * bandit_drifts[:,step-1] + bandit_drifts[:,step]
            #code.interact(local=locals())
    else:
        #print("False")
        for step in np.arange(1,steps,1):
            bandit_drifts[:,step] = bandit_drifts[:,step-1] + bandit_drifts[:,step]
    
    def normal(mu):
        return rg.normal(mu,1)
    
    mu_array = bandit_drifts
    bandit_drifts = np.vectorize(normal)(bandit_drifts)
    
    return bandit_drifts,optimal_bandit,mu_array

def standard_rewards(steps):
    
    mu, sigma = 0, 1.0 # mean and standard deviation
    # Setting actual reward values from normal distribution with 10 bandits
    actual_reward_values = rg.normal(mu,sigma,10)
    
    # Sampling distribution with given actual reward as mean
    rewards = [rg.normal(mu_i,sigma,steps) for mu_i in actual_reward_values]
    optimal_bandit = actual_reward_values.argmax()
    rewards_array = np.array(rewards)
    return rewards_array,optimal_bandit,actual_reward_values
    
for sim in np.arange(0,1000,1):
    
    
    if drift:
        rewards_array, optimal_bandit,mu_array = create_drift(2000)
    else:
        rewards_array, optimal_bandit,mu_array = standard_rewards(2000)
        
    """    
    mu, sigma = 0, 1.0 # mean and standard deviation
    # Setting actual reward values from normal distribution with 10 bandits
    actual_reward_values = rg.normal(mu,sigma,10)
    
    # Sampling distribution with given actual reward as mean
    rewards = [rg.normal(mu_i,sigma,2000) for mu_i in actual_reward_values]
    optimal_bandit = actual_reward_values.argmax()
    rewards_array = np.array(rewards)
    """
    
    greedy_estimates = np.zeros(10)
    choice_estimates = np.zeros(10)
    gradient_estimates = np.zeros(10)
    average_award = 0
    soft_max = np.zeros(10)
    
    if set_optimistic_values:
        choice_estimates = choice_estimates + np.percentile(rewards_array[optimal_bandit],99.5)
    
    steps = np.zeros(10)
    steps = steps 
    
    greedy_average_award = []
    choice_average_award = []
    #soft_max = []
    best_award = []
    overall_average = []
    optimal_action = []
    
    optimal_bandit_action = []


    for step in np.arange(0,2000,1):
        # step through 10 sampled distributions one sample at a time
        step_rewards = rewards_array[:,step]
        # all knowing oracle takes best solution every step
        greward = step_rewards.max() # value
        best_award.append(greward)
        greward_selected = step_rewards.argmax() # index
        
        if gradient_bandit:
            best_choice_selected = choose_max(gradient_estimates)
            best_choice = step_rewards[best_choice_selected]
        # best estimate choice at every step
        else:
            if np.random.uniform() < 1.0-epsilon:
                best_choice_selected = choose_max(choice_estimates)
                #code.interact(banner="choose",local=locals())
                best_choice = step_rewards[best_choice_selected]
            else:
                random_choice  = random.randrange(10)
                best_choice = step_rewards[random_choice]
                best_choice_selected = random_choice    
        
        # gather optimal action %
        if best_choice_selected == greward_selected:
            optimal_action.append(1)
        else:
            optimal_action.append(0)
        if best_choice_selected == optimal_bandit:
            optimal_bandit_action.append(1)
        else:
            optimal_bandit_action.append(0)
        optimal_per = sum(optimal_action)/len(optimal_action)
        
        # build running tally of greedy estimates
        #greedy_estimates[greward_selected] =  greedy_estimates[greward_selected] + 1/steps[greward_selected]*(greward-greedy_estimates[greward_selected])
        # build running tally of each arm estimates
        steps[best_choice_selected] = steps[best_choice_selected] + 1
        choice_estimates[best_choice_selected] = choice_estimates[best_choice_selected] + 1/steps[best_choice_selected]*(best_choice-choice_estimates[best_choice_selected])
        average_award = average_award + 1/(step+1)*(best_choice-average_award)
        
        # gradient bandit
        if gradient_bandit:
            e_sum = sum(math.e**np.array(gradient_estimates))
            #print("e_sum",e_sum)
            for index in np.arange(0,10,1):
                soft_max[index] = math.e**(gradient_estimates[index])/e_sum
            #print("soft max",soft_max)
            for index in np.arange(0,10,1):
                if index == best_choice_selected:
                    gradient_estimates[best_choice_selected] = gradient_estimates[best_choice_selected] + alpha*(best_choice-average_award)*(1-soft_max[best_choice_selected])
                else:
                    gradient_estimates[index] = gradient_estimates[index] - alpha*(best_choice-average_award)*soft_max[index]
        
        #greedy_average_award.append(greedy_estimates[greward_selected])
        choice_average_award.append(best_choice)
        
    all_runs.append(choice_average_award)
    all_optimal_actions.append(optimal_action)
    all_optimal_bandit_options.append(optimal_bandit_action)
    
#code.interact(local=locals())
results = np.array(all_runs)
results_average = results.mean(axis=0)
#plt.plot(results_average)

oa = np.array(all_optimal_actions)
oasum = oa.sum(axis=0)
per = oasum/1000

oab = np.array(all_optimal_bandit_options)
oabsum = oab.sum(axis=0)
oab_per = oabsum/1000



#plt.plot(per)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9))
axs[0].plot(results_average)
axs[1].plot(per)
axs[2].plot(oab_per)
fig.suptitle('10-armed Bandit Testbed')
axs[0].set_ylabel('Average Reward')
axs[1].set_ylabel('Optimal Value')
axs[2].set_ylabel('Optimal Arm')


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

class Config:
    
    def __init__(self, epsilon=0, optimistic_values=0, gradient_bandit_flag=0, alpha=.1):
        
        self.epsilon = epsilon
        self.optimistic_values = optimistic_values
        self.gradient_bandit_flag = gradient_bandit_flag
        self.alpha = alpha
        
config = Config(epsilon=0.01, optimistic_values=0, gradient_bandit_flag=0, alpha=.1)

#epsilon = .1
#set_optimistic_values = 0
#gradient_bandit = 0
#drift=0
#alpha = .1

def choose_max(choices):
    max_indexs = np.argwhere(choices == np.amax(choices))
    max_indexs = max_indexs.flatten().tolist()
    number_of_max = len(max_indexs)
    #code.interact(banner="in choose max",local=locals())
    if number_of_max == 1:
        return max_indexs[0]
    else:
        return max_indexs[random.randrange(number_of_max)]

class Rewards:
    
    def __init__(self,total_rewards=2000, drift=False, reversion=False, abrupt_change=False):
        
        self.total_rewards = total_rewards
        self.drift = drift
        self.reversion = reversion
        self.abrupt_change = abrupt_change
        
    def create_drift(self,steps, reversion=False, abrupt_change=True):
        
        mu, sigma = 0, 1.0 # mean and standard deviation
        initial_reward_values = rg.normal(mu,sigma,10)
        initial_reward_values = np.array(initial_reward_values)
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
            if abrupt_change:
                for step in np.arange(1,500,1):
                    bandit_drifts[:,step] = bandit_drifts[:,step-1] + bandit_drifts[:,step]
                change_reward_values = rg.normal(mu,sigma,10)
                change_reward_values = np.array(change_reward_values)
                bandit_drifts[:,501] = change_reward_values
                for step in np.arange(502,steps,1):
                    bandit_drifts[:,step] = bandit_drifts[:,step-1] + bandit_drifts[:,step]
            else:
                for step in np.arange(1,steps,1):
                    bandit_drifts[:,step] = bandit_drifts[:,step-1] + bandit_drifts[:,step]
    
        def normal(mu):
            return rg.normal(mu,1)
        
        optimal_bandits = []
        for step in np.arange(0,steps,1):
            optimal_bandits.append(bandit_drifts[:,step].argmax())
        optimal_bandits = np.array(optimal_bandits)
        
        mu_array = bandit_drifts
        bandit_drifts = np.vectorize(normal)(bandit_drifts)
        
        return bandit_drifts, optimal_bandit, mu_array, optimal_bandits 

    def standard_rewards(self, steps):
        
        mu, sigma = 0, 1.0 # mean and standard deviation
        # Setting actual reward values from normal distribution with 10 bandits
        actual_reward_values = rg.normal(mu,sigma,10)
        
        # Sampling distribution with given actual reward as mean
        rewards = [rg.normal(mu_i,sigma,steps) for mu_i in actual_reward_values]
        optimal_bandit = actual_reward_values.argmax()
        optimal_bandits = np.zeros(steps)
        optimal_bandits = optimal_bandit
        rewards_array = np.array(rewards)
        #code.interact(local=locals())
        return rewards_array,optimal_bandit,actual_reward_values,optimal_bandits
    
    def generate_rewards(self):
        
        if self.drift:
            rewards_array, optimal_bandits,mu_array,optimal_bandits = self.create_drift(self.total_rewards)
        else:
            rewards_array, optimal_bandit, mu_array, optimal_bandits = self.standard_rewards(self.total_rewards)

        return rewards_array, optimal_bandit, mu_array, optimal_bandits

class ActionValue:
    
    def __init__(self, config):
        
        self.epsilon = config.epsilon
        self.optimistic_values = config.optimistic_values
        self.gradient_bandit = config.gradient_bandit_flag
        self.alpha = config.alpha
        
        self.average_award = 0
        self.reward_recieved = []
        
        
    def select_bandit(self, rewards, estimates, step):
        """
        if self.gradient_bandit:
            bandit_index = choose_max(estimates)
            bandit_reward = rewards[bandit_index]
        else:
        """
        if np.random.uniform() < 1.0-self.epsilon:
            bandit_index = choose_max(estimates)
            bandit_reward = rewards[bandit_index]
        else:
            bandit_index = random.randrange(10)
            bandit_reward = rewards[bandit_index]
                
        self.average_award = self.average_award + 1/(step+1)*(bandit_reward-self.average_award)
        self.reward_recieved.append(bandit_reward)
                
        return bandit_index, bandit_reward
     
class Estimation:
    
    def __init__(self, actionvalue):
        
        self.actionvalue = actionvalue
        self.estimates = np.zeros(10)
        if actionvalue.optimistic_values:
            self.estimates = estimates + np.percentile(rewards_array[optimal_bandit],99.5)
        self.soft_max = np.zeros(10)
        self.steps = np.zeros(10) + 1
        
    def incremental_mean(self, estimates, reward, i):
        
        estimates[i] = estimates[i] + 1/self.steps[i]*(reward-estimates[i])
        #code.interact(banner="im",local=locals())
        return estimates
        
    def bandit_gradient(self, estimates, reward, average_award, i):
        e_sum = sum(math.e**np.array(estimates))
        for index in np.arange(0,10,1):
            self.soft_max[index] = math.e**(estimates[index])/e_sum
        for index in np.arange(0,10,1):
            if index == i:
                estimates[i] = estimates[i] + self.actionvalue.alpha*(reward-average_award)*(1-self.soft_max[i])
            else:
                estimates[index] = estimates[index] - self.actionvalue.alpha*(reward-average_award)*self.soft_max[index]
                
        return estimates
              
    def update_estimates(self, estimates, reward, average_award, i):
        
        if self.actionvalue.gradient_bandit:
            estimates = self.bandit_gradient(estimates, reward, average_award, i)
        else:
            estimates = self.incremental_mean(estimates, reward, i)
            
        self.steps[i] = self.steps[i] + 1
            
        #return estimates
            
class Optimality:

    def __init__(self):
        
        self.optimal_rewards = []
        self.optimal_action = []
        self.optimal_bandit = []
        
    def determine_optimality(self, rewards, best_choice_selected, best_choice, 
                             optimal_bandits,step,mu_array,drift):
        
        self.optimal_rewards.append(rewards.max())
        optimal_action = rewards.argmax()
        if drift:
            optimal_bandit = optimal_bandits[step]
        else: 
            optimal_bandit = optimal_bandits
         
        if best_choice_selected == optimal_action:
            self.optimal_action.append(1)
        else:
            self.optimal_action.append(0)

            
        if best_choice_selected == optimal_bandit:
            self.optimal_bandit.append(1)
        else:
            self.optimal_bandit.append(0)
            
class BanditsRun:
    
    def __init__(self, config):
        
        self.config = config
        self.rewards = Rewards()
        self.rewards_array, self.optimal_bandit, self.mu_array, self.optimal_bandits = self.rewards.generate_rewards()
        self.actions = ActionValue(self.config)
        self.estimate = Estimation(self.actions)
        self.optimal = Optimality()
        
        
    def start(self):
        
        for step in np.arange(0,2000,1):
            step_rewards = self.rewards_array[:,step]
            index, reward = self.actions.select_bandit(step_rewards,
                                                  self.estimate.estimates,
                                                  step)
            self.estimate.update_estimates(self.estimate.estimates,
                                                  reward,
                                                  self.actions.average_award,
                                                  index)
            self.optimal.determine_optimality(step_rewards,
                                         index,
                                         reward,
                                         self.optimal_bandits,
                                         step,
                                         self.mu_array,
                                         self.rewards.drift)

class Testbed:
    
    def __init__(self, number_of_runs, config):
        
        self.config = config
        self.number_of_runs = number_of_runs
        self.all_runs = []
        self.all_optimal_actions = []
        self.all_optimal_bandits = []
        
    def run_testbed(self):
        
        for each_run in np.arange(0,self.number_of_runs):
            
            run = BanditsRun(self.config)
            run.start()
            
            self.all_runs.append(run.actions.reward_recieved)
            self.all_optimal_actions.append(run.optimal.optimal_action)
            self.all_optimal_bandits.append(run.optimal.optimal_bandit)
    
    def generate_plots(self):
        
        results = np.array(self.all_runs)
        results_average = results.mean(axis=0)
        #plt.plot(results_average)

        oa = np.array(self.all_optimal_actions)
        oasum = oa.sum(axis=0)
        per = oasum/1000

        oab = np.array(self.all_optimal_bandits)
        oabsum = oab.sum(axis=0)
        oab_per = oabsum/1000

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 9))
        axs[0].plot(results_average)
        axs[1].plot(per)
        axs[2].plot(oab_per)
        fig.suptitle('10-armed Bandit Testbed')
        axs[0].set_ylabel('Average Reward')
        axs[1].set_ylabel('Optimal Value')
        axs[2].set_ylabel('Optimal Arm')

config = Config(epsilon=0.01, optimistic_values=0, gradient_bandit_flag=0, alpha=.1)
test = Testbed(1000, config)
test.run_testbed()
test.generate_plots()
        

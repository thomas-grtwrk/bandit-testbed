#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:53:54 2025

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

"""
config parameters for action-value methods and rewards creation
"""
class Config:
    
    ######################################################################
    # 
    # epsilon: epsilon-greedy value where epsilon gives probability of
    #          randomly selecting from bandtis
    # optimistic_values: initializes estimates with high value rewards
    # gradient_bandit_flag: use gradient bandit algorithm
    # alpha: parameter for gradient bandit algorithm
    # drift: creates drift in the creation of the reward array
    # total_rewards: sets the number of bandit pulls
    # reversion: creates revrsion of means in the rewards array
    # abrupt_change: changes means of rewards at a set point
    #
    ######################################################################
    
    def __init__(self, epsilon=0, optimistic_values=0,
                 gradient_bandit_flag=False, alpha=.1, drift=False,
                 total_rewards=2000, reversion=False, abrupt_change=False):
        
        self.epsilon = epsilon
        self.optimistic_values = optimistic_values
        self.gradient_bandit_flag = gradient_bandit_flag
        self.alpha = alpha
        self.drift = drift
        self.total_rewards = total_rewards
        self.reversion = reversion
        self.abrupt_change = abrupt_change
        
"""
builds distributional arrays for possible reward values to select
"""
class Rewards:
    
    def __init__(self,config):
        
        self.rng = np.random.default_rng()
        self.sd_rng = np.random.default_rng(seed=7)
        self.total_rewards = config.total_rewards
        self.drift = config.drift
        self.reversion = config.reversion
        self.abrupt_change = config.abrupt_change
        
    def create_drift(self,steps, reversion=False, abrupt_change=False):
        
        mu, sigma = 0, 1.0 # mean and standard deviation
        initial_reward_values = self.rng.normal(mu,sigma,10)
        initial_reward_values = np.array(initial_reward_values)
        optimal_bandit = initial_reward_values.argmax()
        self.optimal_bandit_value = initial_reward_values.max()
        bandit_drifts = [self.sd_rng.normal(0,.01**2,steps) for i in np.arange(0,10,1)]
        bandit_drifts = np.array(bandit_drifts)
        
        bandit_drifts[:,0] = bandit_drifts[:,0] + initial_reward_values
        if reversion:
            for step in np.arange(1,steps,1):
                bandit_drifts[:,step] = .5 * bandit_drifts[:,step-1] + bandit_drifts[:,step]
        else:
            if abrupt_change:
                for step in np.arange(1,500,1):
                    bandit_drifts[:,step] = bandit_drifts[:,step-1] + bandit_drifts[:,step]
                change_reward_values = self.sd_rng.normal(mu,sigma,10)
                change_reward_values = np.array(change_reward_values)
                bandit_drifts[:,501] = change_reward_values
                for step in np.arange(502,steps,1):
                    bandit_drifts[:,step] = bandit_drifts[:,step-1] + bandit_drifts[:,step]
            else:
                for step in np.arange(1,steps,1):
                    bandit_drifts[:,step] = bandit_drifts[:,step-1] + bandit_drifts[:,step]
    
        def normal(mu):
            return self.rng.normal(mu,1)
        
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
        actual_reward_values = self.rng.normal(mu,sigma,10)
        
        # Sampling distribution with given actual reward as mean
        rewards = [self.rng.normal(mu_i,sigma,steps) for mu_i in actual_reward_values]
        optimal_bandit = actual_reward_values.argmax()
        self.optimal_bandit_value = actual_reward_values.max()
        # foolish placeholder to keep return values consistent
        optimal_bandits = np.zeros(steps)
        optimal_bandits = optimal_bandit
        rewards_array = np.array(rewards)

        return rewards_array,optimal_bandit,actual_reward_values,optimal_bandits
    
    def generate_rewards(self):
        
        if self.drift:
            rewards_array, optimal_bandit,mu_array,optimal_bandits = self.create_drift(self.total_rewards, reversion=self.reversion, abrupt_change=self.abrupt_change)
        else:
            rewards_array, optimal_bandit, mu_array, optimal_bandits = self.standard_rewards(self.total_rewards)

        return rewards_array, optimal_bandit, mu_array, optimal_bandits

"""
impliments the acion-value methods to select bandit arms based on rewards
"""
class ActionValue:
    
    def __init__(self, config):
        
        self.epsilon = config.epsilon
        self.optimistic_values = config.optimistic_values
        self.gradient_bandit = config.gradient_bandit_flag
        self.alpha = config.alpha
        
        self.average_award = 0
        self.reward_recieved = []
        
    def choose_max(self,choices):
        max_indexs = np.argwhere(choices == np.amax(choices))
        max_indexs = max_indexs.flatten().tolist()
        number_of_max = len(max_indexs)
        if number_of_max == 1:
            return max_indexs[0]
        else:
            return max_indexs[random.randrange(number_of_max)]
        
    def select_bandit(self, rewards, estimates, step):
        """
        
        """
        if np.random.uniform() < 1.0-self.epsilon:
            bandit_index = self.choose_max(estimates)
            bandit_reward = rewards[bandit_index]
        else:
            bandit_index = random.randrange(10)
            bandit_reward = rewards[bandit_index]
                
        self.average_award = self.average_award + 1/(step+1)*(bandit_reward-self.average_award)
        self.reward_recieved.append(bandit_reward)
                
        return bandit_index, bandit_reward

"""
handles estimation accounting as rewards are recieved for bandit run
"""    
class Estimation:
    
    def __init__(self, actionvalue, rewards):
        
        self.actionvalue = actionvalue
        self.estimates = np.zeros(10)
        if actionvalue.optimistic_values:
            self.estimates = self.estimates + np.percentile(rewards.optimal_bandit_value,99.5)
        self.soft_max = np.zeros(10)
        self.steps = np.zeros(10) + 1
        
    def incremental_mean(self, estimates, reward, i):
        
        estimates[i] = estimates[i] + 1/self.steps[i]*(reward-estimates[i])
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

"""
tracks bandit arm selection to determine optimal selection
"""                    
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

"""
runs a single bandit simulation
"""
class BanditsRun:
    
    def __init__(self, config):
        
        self.config = config
        self.rewards = Rewards(config)
        self.rewards_array, self.optimal_bandit, self.mu_array, self.optimal_bandits = self.rewards.generate_rewards()
        self.actions = ActionValue(self.config)
        self.estimate = Estimation(self.actions, self.rewards)
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

"""
runs multiple bandit simulations to gauge methodologies utility
"""
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

"""
Running Testbed

config1 = Config(epsilon=0.0, optimistic_values=False,
                gradient_bandit_flag=False, alpha=.1, drift=False,
                total_rewards=2000, reversion=False, abrupt_change=False)
test1 = Testbed(1000, config1)
test1.run_testbed()
#test1.generate_plots()

config2 = Config(epsilon=0.01, optimistic_values=False,
                gradient_bandit_flag=False, alpha=.1, drift=False,
                total_rewards=2000, reversion=False, abrupt_change=False)

test2 = Testbed(1000, config2)
test2.run_testbed()
#test2.generate_plots()

config3 = Config(epsilon=0.05, optimistic_values=False,
                gradient_bandit_flag=False, alpha=.1, drift=False,
                total_rewards=2000, reversion=False, abrupt_change=False)

test3 = Testbed(1000, config3)
test3.run_testbed()
#test3.generate_plots()

config4 = Config(epsilon=0.1, optimistic_values=False,
                gradient_bandit_flag=False, alpha=.1, drift=False,
                total_rewards=2000, reversion=False, abrupt_change=False)

test4 = Testbed(1000, config4)
test4.run_testbed()
#test4.generate_plots()

config5 = Config(epsilon=0, optimistic_values=True,
                gradient_bandit_flag=False, alpha=.1, drift=True,
                total_rewards=2000, reversion=False, abrupt_change=False)

test5 = Testbed(1000, config5)
test5.run_testbed()

config6 = Config(epsilon=0, optimistic_values=True,
                gradient_bandit_flag=True, alpha=.1, drift=True,
                total_rewards=2000, reversion=False, abrupt_change=False)

test6 = Testbed(1000, config6)
test6.run_testbed()

config7 = Config(epsilon=0, optimistic_values=True,
                gradient_bandit_flag=True, alpha=.2, drift=True,
                total_rewards=2000, reversion=False, abrupt_change=False)

test7 = Testbed(1000, config7)
test7.run_testbed()

config8 = Config(epsilon=0, optimistic_values=True,
                gradient_bandit_flag=True, alpha=.4, drift=True,
                total_rewards=2000, reversion=False, abrupt_change=False)

test8 = Testbed(1000, config8)
test8.run_testbed()

config9 = Config(epsilon=0, optimistic_values=True,
                gradient_bandit_flag=True, alpha=.8, drift=True,
                total_rewards=2000, reversion=False, abrupt_change=False)

test9 = Testbed(1000, config9)
test9.run_testbed()

config10 = Config(epsilon=0, optimistic_values=True,
                gradient_bandit_flag=True, alpha=.05, drift=True,
                total_rewards=2000, reversion=False, abrupt_change=False)

test10 = Testbed(1000, config10)
test10.run_testbed()

config11 = Config(epsilon=0, optimistic_values=False,
                gradient_bandit_flag=True, alpha=.02, drift=True,
                total_rewards=2000, reversion=False, abrupt_change=False)

test11 = Testbed(1000, config11)
test11.run_testbed()


config12 = Config(epsilon=0, optimistic_values=False,
                gradient_bandit_flag=True, alpha=.02, drift=True,
                total_rewards=2000, reversion=False, abrupt_change=True)

test12 = Testbed(1000, config12)
test12.run_testbed()

config13 = Config(epsilon=.1, optimistic_values=False,
                gradient_bandit_flag=False, alpha=.02, drift=True,
                total_rewards=2000, reversion=True, abrupt_change=False)

test13 = Testbed(1000, config13)
test13.run_testbed()


config14 = Config(epsilon=0, optimistic_values=False,
                gradient_bandit_flag=False, alpha=.02, drift=False,
                total_rewards=2000, reversion=True, abrupt_change=False)

test14 = Testbed(1000, config14)
test14.run_testbed()

config15 = Config(epsilon=0, optimistic_values=False,
                gradient_bandit_flag=True, alpha=.05, drift=False,
                total_rewards=2000, reversion=True, abrupt_change=False)

test15 = Testbed(1000, config15)
test15.run_testbed()
"""


# bandit-testbed

A testbed for multi-arm bandit problems. Currently impliments various action-value methods
for staionary and non-stationary multi-arm bandits. Rewards are pre-determined from statistical distributions
to offer flexibility in the manipulation of stationary and non-stationary rewards.

## Examples of usage

This configuration will run a greedy with non-optimistic initial values bandit learning algorithm.
Each individual bandit run will have 2000 time steps indicated by the total_rewards parameter.
The testbed will run 1000 of the bandit simulations and plot the averaged results over all bandit runs.
The plots show the average reward for each time step over the 1000 simulations. The percentage of time 
the algorithm selected the optimal value (the max reward at each time step over all possible bandits) 
and the optimal arm (the optimal bandit based on the best actual mean of all bandit distributions.)


'''
config = Config(epsilon=0, optimistic_values=False,
                gradient_bandit_flag=False, alpha=.1, drift=False,
                total_rewards=2000, reversion=False, abrupt_change=False)
test = Testbed(1000, config)
test.run_testbed()
test.generate_plots()
'''
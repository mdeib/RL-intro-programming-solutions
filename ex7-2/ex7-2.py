#RL: An Introduction exercise 7.2
#Comparison of n-step bootstrapping error and sum of TD errors
#random walk env
#Solution by Matt Deible

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import random
import itertools

STATES = 19 #any odd number - 19 was used in the book
INITIAL = (STATES // 2)

optimal_values = np.zeros(STATES)
for i in range(INITIAL):
    optimal_values[INITIAL - 1 - i] = -(i + 1) / (INITIAL + 1)
    optimal_values[INITIAL + 1 + i] = (i + 1) / (INITIAL + 1)

policy = lambda S: 1 if random.random() < 0.5 else -1 #policy is random step

#Experiment parameters
#No discount in this environment
#using sampling of all possible alpha and all n shown in figure 7.2 pg. 145
learning_rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

EPISODES = 10 #same episode count used as in figure 7.2
REPITITIONS = 1000 #REPITITIONS was increased to provide a smoother result

for n in n_values:
    print("Running experiment with n =", n)
    error_difference = []

    for ALPHA in learning_rates:
        print("Testing alpha =", ALPHA)
        if ALPHA == 0:
            error_difference.append(0) #won't learn with alpha = 0
            continue

        nstep_avg_rms = 0
        TD_avg_rms = 0

        for repetition in range(REPITITIONS):
            value = np.zeros(STATES)
            TD_value = [np.zeros(STATES + 1)] #need to keep history of value function for TD summation, extra state is value of terminal state = 0

            for episode in range(EPISODES):
                TD_value = [TD_value[-1].copy()] #clear TD_value history from last episode
                S = [INITIAL]
                T = math.inf
                for t in itertools.count(): #iterate through timesteps t = 0, 1, 2,...
                    if t < T:
                        S.append(S[t] + policy(S[t]))
                        if S[t + 1] == -1 or S[t + 1] == STATES: T = t + 1 #check if state is terminal

                    TAU = t - n + 1
                    if TAU >= 0:
                        #first do n-step value updating
                        if T == math.inf: G = value[S[TAU + n]] #if not at terminal state all rewards are 0 so G is just estimate of state
                        elif S[-1] == -1: G = -1 #terminal state that causes -1 reward
                        else: G = 1 #terminal state that causes 1 reward
                        value[S[TAU]] += ALPHA * (G - value[S[TAU]])

                        #now do summation of TD(0) updates
                        #initialize G with reward if terminal state is reached
                        if S[-1] == -1: update = -1 #far left got reward of -1
                        elif S[-1] == STATES: update = 1 #far right got reward of 1
                        else: update = 0 #if haven't reached terminal state, no rewards recieved
                        for i in range(min(n, T - TAU)): #now add up all TD(0) updates that would have happened (all other rewards were 0, so they are omitted)
                            update += TD_value[TAU + i][S[TAU + i + 1]] - TD_value[TAU + i][S[TAU + i]]

                        TD_value[t][S[TAU]] += (ALPHA * update) #update current TD_value
                        TD_value[t][S[TAU]] = sorted([-1, TD_value[t][S[TAU]], 1])[1] #Bound new value estimation by max and min returns

                    if TAU == T - 1: break
                    TD_value.append(TD_value[t].copy()) #save history t - 1

            nstep_avg_rms += math.sqrt(mean_squared_error(optimal_values, value))
            TD_avg_rms += math.sqrt(mean_squared_error(optimal_values, TD_value[-1][:-1]))
        nstep_avg_rms /= REPITITIONS
        TD_avg_rms /= REPITITIONS

        error_difference.append(nstep_avg_rms - TD_avg_rms)

    plt.plot(learning_rates, error_difference)

plt.title("n-step vs TD(0) summation updating")
plt.legend(n_values, title = "n value")

plt.xlabel('Alpha')
plt.ylabel('RMSE difference (n-step - TD(0))')
plt.show()

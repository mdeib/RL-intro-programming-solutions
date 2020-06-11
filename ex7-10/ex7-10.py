#RL: An Introduction exercise 7.10
#Demonstrating data efficiency of control variate methiods
#Windy gridworld env from ex6.9
#Solution by Matt Deible

import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt

HEIGHT = 7
WIDTH = 10
STATES = WIDTH * HEIGHT

WIND = np.zeros(WIDTH)
WIND[3:9] = 1
WIND[6:8] = 2

GOAL = (3, 7)
START = (3, 0)

ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] #As originally posed
#ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)] #kings moves
#ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (0, 0)] #kings moves with option for no move

n = 16
EPSILON = 0.1
GREEDY_ISR = 1 / ((1 - EPSILON) + (EPSILON / len(ACTIONS))) #this will be the importance sampling ratio for all actions in greedy policy
ALPHA = 0.2
REPITITIONS = 500
EPISODES = 20

#transition to next state
next_state = lambda S, A: (int(sorted([0, S[0] + A[0] + WIND[S[1]], HEIGHT - 1])[1]), int(sorted([0, S[1] + A[1], WIDTH - 1])[1]))

#get greedy action from S with respect to V
def greedy(V, S):
    Q = []
    for A in ACTIONS:
        Q.append(V[next_state(S, A)])
    return np.argmax(np.array(Q))

#get action using epsilon greedy policy from S with respect to V
def epsilon_greedy(V, S):
    if (random.random() > EPSILON): return greedy(V, S)
    else: return(random.randint(0, len(ACTIONS) - 1))

def no_control_variate_rho(V, S):
    if len(S) == 1: return 1
    greedy_action = greedy(V, S[0])
    if S[1] != next_state(S[0], ACTIONS[greedy_action]): return 0 #if not greedy action, pi(A | S) = 0, so whole ratio = 0
    return GREEDY_ISR * no_control_variate_rho(V, S[1:])

def control_variate_return(V, S):
    if len(S) == 1: return V[S[0]] #if at end of horizon, return value of horizon
    greedy_action = greedy(V, S[0])
    if S[1] != next_state(S[0], ACTIONS[greedy_action]): return V[S[0]] #if not greedy action, rho = 0 so just return V(S_t)
    return GREEDY_ISR * (-1 + control_variate_return(V, S[1:])) + ((1 - GREEDY_ISR) * V[S[0]])

#BEGIN EXPERIMENT
if __name__ == '__main__':
    avg_cv_returns = np.zeros(EPISODES)
    avg_no_cv_returns = np.zeros(EPISODES)

    for repetition in range(REPITITIONS):
        print("Starting repition", repetition)
        cv_value = np.random.rand(HEIGHT, WIDTH) #initialize value function randomly
        cv_value[GOAL] = 0 #terminal value = 0
        no_cv_value = cv_value.copy() #make both value functions intialized to same random values for easy comparison

        #calculate using control variate using equations 7.13 and 7.2
        for episode in range(EPISODES):
            S = [START]
            T = math.inf
            for t in itertools.count(): #iterate through timesteps t = 0, 1, 2 ...
                if t < T:
                    S.append(next_state(S[-1], ACTIONS[epsilon_greedy(cv_value, S[t])]))
                    if S[-1] == GOAL: T = t + 1
                TAU = t - n + 1
                if TAU >= 0:
                    G = control_variate_return(cv_value, S[TAU:]) #Eq 7.13
                    cv_value[S[TAU]] += ALPHA * (G - cv_value[S[TAU]]) #Eq 7.2

                if TAU == T - 1:
                    avg_cv_returns[episode] -= T
                    break

        #calculate without using control variate using equations 7.1 and 7.9
        for episode in range(EPISODES):
            S = [START]
            T = math.inf
            for t in itertools.count(): #iterate through timesteps t = 0, 1, 2 ...
                if t < T:
                    S.append(next_state(S[-1], ACTIONS[epsilon_greedy(no_cv_value, S[t])]))
                    if S[-1] == GOAL: T = t + 1
                TAU = t - n + 1
                if TAU >= 0:
                    RHO = no_control_variate_rho(no_cv_value, S[TAU:])
                    G = -1 * min(n, T - TAU) + no_cv_value[S[-1]] #Eq 7.1
                    no_cv_value[S[TAU]] += min(ALPHA * RHO, 1) * (G - no_cv_value[S[TAU]]) #Eq 7.9

                if TAU == T - 1:
                    avg_no_cv_returns[episode] -= T
                    break

    avg_cv_returns /= REPITITIONS
    avg_no_cv_returns /= REPITITIONS

    plt.plot(avg_cv_returns)
    plt.plot(avg_no_cv_returns)
    plt.legend(["control variate", "no control variate"])

    plt.title("Control variate data efficiency at n = " + str(n))
    plt.xlabel('episode')
    plt.ylabel('return')
    plt.show()

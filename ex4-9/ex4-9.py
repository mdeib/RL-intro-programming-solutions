#RL: An Introduction exercise 4.9
#Implementation of value iteration
#Gambling game env
#Solution by Matt Deible

import numpy as np
import matplotlib.pyplot as plt

STATES = np.arange(101)
ACTIONS = np.arange(100)
DISCOUNT = 1
THETA = 0.000000001

fig, ax = plt.subplots(2, 2)
fig_col = 0

def get_updated_value(s, PROB_WIN, PROB_LOSE):
    if s == 0: return 0 #lose
    if s == 100: return 1 #win
    max_q = 0
    a = 1
    while a <= s:
        case_win = 1 if s + a > 100 else DISCOUNT * value[s + a]
        case_lose = DISCOUNT * value[s - a]

        new_q = (PROB_WIN * case_win) + (PROB_LOSE * case_lose)
        if new_q > max_q: max_q = new_q
        a += 1
    return max_q

for PROB_WIN in [0.25]:
    PROB_LOSE = 1 - PROB_WIN
    value = np.zeros(len(STATES))
    value[-1] = 1

    #Value iteration
    while True:
        delta = 0
        for s in STATES:
            v = value[s]
            value[s] = get_updated_value(s, PROB_WIN, PROB_LOSE)
            delta = max(delta, abs(v - value[s]))
        if (delta <= THETA):
            break

    policy = np.zeros(len(STATES)) #construct greedy policy
    for s in STATES:
        if s == 0 or s == 100: continue #terminal states

        max_q = 0
        a = 1
        while a <= s:
            case_win = 1 if s + a > 100 else DISCOUNT * value[s + a]
            case_lose = DISCOUNT * value[s - a]
            new_q = (PROB_WIN * case_win) + (PROB_LOSE * case_lose)
            if new_q > max_q:
                max_q = new_q
                policy[s] = a
            a += 1

    ax[0, fig_col].title.set_text("win rate = " + str(PROB_WIN) + " policy")
    ax[0, fig_col].plot(policy, 'r')
    ax[1, fig_col].title.set_text("win rate = " + str(PROB_WIN) + " value")
    ax[1, fig_col].plot(value, 'b')

    fig_col += 1

plt.show()

#RL: An Introduction exercise 6.10
#Implementation of TD control
#Stochastic windy gridworld env
#Solution by Matt Deible

import numpy as np
import random
import matplotlib.pyplot as plt

HEIGHT = 7
WIDTH = 10

WIND = np.zeros(WIDTH)
WIND[3:9] = 1
WIND[6:8] = 2

GOAL = (3, 7)
START = (3, 0)

#ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] #As originally posed
ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)] #kings moves
#ACTIONS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (0, 0)] #kings moves with option for no move

#transition to next state
def next_state(S, A):
    wind = WIND[S[1]]
    stochastic = random.random()
    if wind > 0: #wind is only stochastic if there is any
        if stochastic < (1.0 / 3.0): wind -= 1
        elif stochastic > (2.0 / 3.0): wind += 1
    return (int(sorted([0, S[0] + A[0] + wind, HEIGHT - 1])[1]), int(sorted([0, S[1] + A[1], WIDTH - 1])[1]))

#get action using epsilon greedy policy with respect to Q
def epsilon_greedy (Q, S, EPSILION):
    if (random.random() > EPSILON): return(np.argmax(Q[S[0], S[1]]))
    else: return(random.randint(0, len(ACTIONS) - 1))

if __name__ == '__main__':
    EPISODES = 10000000
    EPSILON = 0.1
    EPSILON_DECAY = (EPISODES - 1) / EPISODES
    ALPHA = 0.5

    Q = np.random.rand(HEIGHT, WIDTH, len(ACTIONS))
    Q[GOAL[0], GOAL[1], :] = 0

    for episode in range(EPISODES):
        EPSILON *= EPSILON_DECAY
        S = START
        A = epsilon_greedy(Q, S, EPSILON)

        while S != GOAL:
            Sp = next_state(S, ACTIONS[A])
            Ap = epsilon_greedy(Q, Sp, EPSILON)
            Q[S[0], S[1], A] += ALPHA * (-1 + Q[Sp[0], Sp[1], Ap] - Q[S[0], S[1], A])
            S = Sp
            A = Ap
        if episode % 1000 == 0: print("episode:", episode, "epsilon:", EPSILON)

    fig, ax = plt.subplots(1, 2)

    #Calculate optimal path
    path = np.zeros((HEIGHT, WIDTH))
    path[START[0], START[1]] = 1
    S = START
    while S != GOAL:
        S = next_state(S, ACTIONS[np.argmax(Q[S[0], S[1]])])
        path[S[0], S[1]] = 2
    path[GOAL[0], GOAL[1]] = 3

    ax[0].matshow(path)
    ax[0].title.set_text("trajectory")
    ax[0].invert_yaxis()

    #Calculate value function from Q function
    value = np.zeros((HEIGHT, WIDTH))
    for j in range(HEIGHT):
        for i in range(WIDTH):
            value[j, i] = np.max(Q[j, i])

    ax[1].matshow(value)
    ax[1].title.set_text("value function")
    ax[1].invert_yaxis()

    plt.show()

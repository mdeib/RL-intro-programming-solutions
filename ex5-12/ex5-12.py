#RL: An Introduction exercise 5.12
#Implementation of MC control
#Racetrack right turn env
#Solution by Matt Deible

import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
import math
import random

VEL_LIMIT = 5

OOB = -1
START = 1
FINISH = 2
WIDTH = 17
HEIGHT = 32
track = np.zeros((HEIGHT, WIDTH))

#out of bounds
track[:3, :3] = OOB
track[3:10, :2] = OOB
track[10:18, 0] = OOB
track[28, 0] = OOB
track[29:31, :2] = OOB
track[31, :3] = OOB
track[:25, 9:] = OOB
track[25, 10:] = OOB

#start states
LEFTMOST_START = 3
track[0, LEFTMOST_START:9] = START
INITIAL_STATES = [(0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 8, 0, 0)]

#finish states
track[26:, 16] = FINISH

distance_to_vel = [0, 1, 3, 6, 10]
STATES = INITIAL_STATES.copy()
for y in range(HEIGHT):
    for x in range(WIDTH):
        if track[y, x] == -1: continue #no states out of bounds
        for yv in range(VEL_LIMIT):
            if y < distance_to_vel[yv]: continue #no states before car can get to its y velocity
            for xv in range(VEL_LIMIT):
                if x < distance_to_vel[xv] + LEFTMOST_START: continue #no states before car can get to its x velocity
                if yv > 0 or xv > 0: STATES.append((y, x, yv, xv)) #only initial states can have both velocities 0

def get_valid_actions(s): #get valid actions in state s such that agent can't get negative velocity, exceed the limit or stop moving
    if s[2] == 1 and s[3] == 0: return [(0, 0), (0, 1), (1, 0), (1, 1), (-1, 1)]
    if s[2] == 1 and s[3] == 1: return [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1)]
    if s[2] == 0 and s[3] == 1: return [(0, 0), (0, 1), (1, 0), (1, 1), (1, -1)]

    if s[2] == 0:
        if s[3] == 0: return [(0, 1), (1, 0), (1, 1)]
        if s[3] == VEL_LIMIT - 1: return [(0, 0), (0, -1), (1, 0), (1, -1)]
        return [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1)]

    if s[2] ==  VEL_LIMIT - 1:
        if s[3] == 0: return [(0, 0), (0, 1), (-1, 0), (-1, 1)]
        if s[3] ==  VEL_LIMIT - 1: return [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        return [(0, 0), (0, 1), (0, -1), (-1, 0), (-1, 1), (-1, -1)]

    if s[3] == 0: return [(0, 0), (0, 1), (1, 0), (1, 1), (-1, 0), (-1, 1)]
    if s[3] ==  VEL_LIMIT - 1: return [(0, 0), (0, -1), (1, 0), (1, -1), (-1, 0), (-1, -1)]

    return [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)]

def gen_episode(target_policy, epsilon, initial_state_indice, no_action_chance): #use epsilon-greedy version of target_policy to generate episode
    MAX = 0
    states = []
    actions = []
    states.append(INITIAL_STATES[initial_state_indice]) #start with random state
    while True:
        valid_actions = get_valid_actions(states[-1])
        if random.random() < no_action_chance and valid_actions[0] == (0, 0): #with 10 percent chance, make change to both velocities 0 if not in start position
            actions.append(0)
        else:
            if random.random() > epsilon:
                actions.append(int(target_policy[states[-1][0], states[-1][1], states[-1][2], states[-1][3]]))
            else:
                actions.append(random.randint(0, len(valid_actions) - 1))

        a = valid_actions[actions[-1]]

        velocity = [states[-1][2] + a[0], states[-1][3] + a[1]] #new (y,x) velocity
        next_state = (states[-1][0] + velocity[0], states[-1][1] + velocity[1], velocity[0], velocity[1])

        crashed = False
        inty, intx = skimage.draw.line(states[-1][0], states[-1][1], next_state[0], next_state[1]) #intercepted tiles
        for tile in range(len(intx)):
            if inty[tile] >= HEIGHT or track[inty[tile], intx[tile]] == OOB:
                crashed = True
                break
            if track[inty[tile], intx[tile]] == FINISH: #crossed finish, return episode
                states.append((inty[tile], intx[tile])) #finish line cross for displaying trajectory
                return states, actions
        if crashed: states.append(INITIAL_STATES[initial_state_indice]) #if crashed, restart at beginning
        else: states.append(next_state) #else go to next state

if __name__ == '__main__':
    EPSILON = 0.2
    DISCOUNT = 1
    EPISODES = 5000000

    Q = {}
    C = {}
    target_policy = np.zeros((HEIGHT, WIDTH, VEL_LIMIT, VEL_LIMIT))
    for s in STATES:
        Q[s] = np.random.rand(len(get_valid_actions(s))) - 1000 # initializing Qs to low values so agent does not get stuck
        C[s] = np.zeros(len(get_valid_actions(s)))
        target_policy[s[0], s[1], s[2], s[3]] = np.argmax(Q[s])

    for episode in range(EPISODES):
        states, actions = gen_episode(target_policy, EPSILON, random.randint(0, len(INITIAL_STATES) - 1), 0.1)
        G = 0
        W = 1
        T = len(states) - 1 #don't include finish line cross state
        if episode % 1000 == 0:
            print("generated episode", episode, "of length:", T)
        for t in range(T):
            S = states[T - 1 - t]
            A = actions[T - 1 - t]
            G = DISCOUNT * G - 1 #reward is always -1
            C[S][A] += W
            Q[S][A] += W / C[S][A] * (G - Q[S][A])
            target_policy[S[0], S[1], S[2], S[3]] = np.argmax(Q[S])
            if A != target_policy[S[0], S[1], S[2], S[3]]: #if episode took different action from target_policy, no longer usable
                break
            W /= (1 - EPSILON) + (EPSILON / len(get_valid_actions(S))) #if made to this point, action must be currently greedy action

    plt.matshow(track)

    for i in range(len(INITIAL_STATES)):
        states, actions = gen_episode(target_policy, 0, i, 0) #noise turned off and greedy policy
        x = []
        y = []
        for s in states:
            y.append(s[0])
            x.append(s[1])
        plt.plot(x, y)

    plt.gca().invert_yaxis()
    plt.show()

#RL: An Introduction exercise 4.7
#Implementation of policy iteration
#Car rental service optimization env
#Solution by Matt Deible

import numpy as np
import math
import matplotlib.pyplot as plt


#Constants
MODIFIED_REWARD = True #set to false for problem as originally posed, true to add changes stated in exercise 4.7
MAX_CARS_IN_LOT = 20
DISCOUNT = 0.9
THETA = 0.1

STATES = []
for car_num1 in range(MAX_CARS_IN_LOT + 1):
    for car_num2 in range(MAX_CARS_IN_LOT + 1):
        STATES.append((car_num1, car_num2))

ACTIONS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def poisson_dist(l):
    dist = []
    for k in range(MAX_CARS_IN_LOT + 1):
        dist.append((l**k * math.exp(-l)) / math.factorial(k))
    return dist

REQUESTS_ONE = poisson_dist(3)
REQUESTS_TWO = poisson_dist(4)
RETURNS_ONE = poisson_dist(3)
RETURNS_TWO = poisson_dist(2)

#functions

bound = lambda cars: sorted([0, cars, MAX_CARS_IN_LOT])[1]

def get_reward(state, action, req1, req2):
    reward = (min(req1, bound(state[0] - action)) * 10) + (min(req2, bound(state[1] + action)) * 10) - (abs(action) * 2)
    if MODIFIED_REWARD:
        if action > 0: reward += 2 #reimburse for one free move to lot 1 if it happens
        if state[0] - action > 10: reward -= 4 #if >10 cars in lot 1 after movement, charge $4
        if state[1] + action > 10: reward -= 4 #if >10 cars in lot 2 after movement, charge $4
    return reward

def next_state(state, action, ret1, req1, ret2, req2):
    return [int(bound(bound(state[0] - action - req1) + ret1)), int(bound(bound(state[1] + action - req2) + ret2))]

def get_action_q(s, a, value):
    q = 0
    for req1 in range(11):
        for req2 in range(12):
            r = get_reward(s, a, req1, req2)
            for ret1 in range(11):
                for ret2 in range(10):
                    sp = next_state(s, a, ret1, req1, ret2, req2)
                    probability = REQUESTS_ONE[req1] * REQUESTS_TWO[req2] * RETURNS_ONE[ret1] * RETURNS_TWO[ret2]
                    case_value = r + (DISCOUNT * value[sp[0], sp[1]])
                    q += (probability * case_value)
    return q


#PART 1: INITIALIZATION
policy = np.zeros((21, 21))
value = np.zeros((21, 21))

iteration = 0
policy_unstable = True
while policy_unstable:
    #PART 2: POLICY EVALUATION
    print("POLICY EVALUATION")
    round = 0
    while True:
        delta = 0
        for s in STATES:
            v = value[s[0], s[1]]
            value[s[0], s[1]] = get_action_q(s, policy[s[0], s[1]], value)
            delta = max(delta, abs(v - value[s[0], s[1]]))
        round += 1
        print("done with sweep:", round)
        if (delta <= THETA):
            break

    #PART 3: POLICY IMPROVEMENT
    print("POLICY IMPROVEMENT")
    policy_unstable = False
    for s in STATES:
        old_q = get_action_q(s, policy[s[0], s[1]], value)

        valid_actions = []
        for a in ACTIONS:
            if s[0] >= a and s[1] >= -a: valid_actions.append(a)
        q = []
        for a in valid_actions:
            q.append(get_action_q(s, a, value))

        policy[s[0], s[1]] = valid_actions[q.index(max(q))]
        if max(q) != old_q:
            policy_unstable = True

    iteration += 1
    print("____________________ ITERATION:", iteration, "COMPLETED ____________________")

plt.matshow(policy)
plt.gca().invert_yaxis()
plt.ylabel("# of cars at first location")
plt.xlabel("# of cars at second location")
plt.show()

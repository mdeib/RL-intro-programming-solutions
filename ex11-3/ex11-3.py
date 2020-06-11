#RL: An Introduction exercise 11.3
#Semi-gradient Q-learning divergence
#Gaussian reward distribution env
#Solution by Matt Deible

import numpy as np
import matplotlib.pyplot as plt
import random

class bairds_counterexample:
    def __init__(self):
        #construct feature map for 7 states
        self.features = np.zeros((7, 8))
        self.features[:6, 7] = 1
        for i in range(6):
            self.features[i, i] = 2
        self.features[6, 6] = 1
        self.features[6, 7] = 2

        self.action_space = 2
        self.state = 6

    def step(self, action):
        old_state = self.state
        if action == 0: self.state = random.randrange(6)
        elif action == 1: self.state = 6

        #return features of current state and the new state
        return self.features[old_state], self.features[self.state]

class semi_gradient_Q_learner:
    def __init__(self, env):
        self.alpha = 0.01
        self.gamma = 0.99
        self.env = env
        self.W = np.array([1, 1, 1, 1, 1, 1, 10, 1, 1])
        self.W_history = []

    def behavior_policy(self):
        #pick action according to behavior policy --- is independent of current state
        if random.randrange(7) == 6: return 1
        return 0

    def Q(self, state, action):
        return np.dot(np.append(state, action), self.W)

    def act(self):
        action = self.behavior_policy()
        state, next_state = env.step(action) #act in environment and get features of current and next state

        #update history and then update weights
        self.W_history.append(self.W)
        gradient = np.append(state, action) #linear gradient is equal to the value of each feature
        next_Q_max = max(self.Q(next_state, 0), self.Q(next_state, 1))
        self.W = self.W + (self.alpha * ((self.gamma * next_Q_max) - self.Q(state, action)) * gradient)

if __name__ == '__main__':
    steps = 10000
    env = bairds_counterexample()
    agent = semi_gradient_Q_learner(env)

    for step in range(steps):
        agent.act()

    legend = []
    weight_history = np.array(agent.W_history)
    for W in range(9):
        legend.append("Weight " + str(W + 1))
        plt.plot(weight_history[:, W])

    plt.legend(legend)
    plt.xlabel("Steps")
    plt.show()

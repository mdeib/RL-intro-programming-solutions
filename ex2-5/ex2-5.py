#RL: An Introduction exercise 2.5
#Nonstationary Bandit Comparison
#Nonstationary Ten Armed Test Bed Env
#Solution by Matt Deible

import numpy as np
import matplotlib.pyplot as plt
import random

class nonstationary_bandit:
    def __init__(self, k):
        self.Q_star = np.zeros(k)
        self.k = k

    def step(self, action):
        self.Q_star = self.Q_star + np.random.normal(0, 0.01, self.k)
        reward = np.random.normal(self.Q_star[action], 1)
        return reward

class Q_agent:
    def __init__(self, env, alpha):
        self.env = env
        self.epsilon = 0.1
        self.Q = np.zeros(env.k)
        self.returns_history = [0]
        self.optimal_actions = [0]
        self.alpha = alpha
        if self.alpha == 0: #alpha of 0 uses running average, so need to keep trak of samples from each action
            self.samples = np.zeros(env.k)

    def learn(self, steps):
        for step in range(steps):
            self.act()

    def act(self):
        if random.random() < self.epsilon:
            action = random.randrange(10)
        else:
            action = np.argmax(self.Q)
        reward = self.env.step(action)

        self.returns_history.append(self.returns_history[-1] + reward)
        self.optimal_actions.append(self.optimal_actions[-1] + int(action == np.argmax(self.env.Q_star)))

        self.update(action, reward)

    def update(self, action, reward):
        if self.alpha == 0: #running average case
            self.samples[action] += 1
            self.Q[action] += (1 / self.samples[action]) * (reward - self.Q[action])
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action])

if __name__ == '__main__':
    steps = 10000
    runs = 2000
    simple_average_data = np.zeros(steps + 1)
    exponential_average_data = np.zeros(steps + 1)

    for run in range(runs):
        print("Run number:", run)
        simple_average_agent = Q_agent(nonstationary_bandit(10), 0)
        exponential_average_agent = Q_agent(nonstationary_bandit(10), 0.1)

        simple_average_agent.learn(steps)
        exponential_average_agent.learn(steps)

        simple_average_data = simple_average_data + np.array(simple_average_agent.optimal_actions)
        exponential_average_data = exponential_average_data + np.array(exponential_average_agent.optimal_actions)

    simple_average_data = simple_average_data / runs
    exponential_average_data = exponential_average_data / runs

    for i in range(steps):
        exponential_average_data[i + 1] = 100 * exponential_average_data[i + 1] / (i + 1)
        simple_average_data[i + 1] = 100 * simple_average_data[i + 1] / (i + 1)

    plt.plot(simple_average_data)
    plt.plot(exponential_average_data)

    plt.legend(["simple average", "alpha = 0.1"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.show()

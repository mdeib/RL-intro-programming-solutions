#RL: An Introduction exercise 2.11
#Nonstationary Full Algo Parameter Study
#Nonstationary Ten Armed Test Bed Env
#Solution by Matt Deible

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
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
    def __init__(self, type, env, alpha, special_param):
        self.env = env
        self.t = 0
        self.N = np.zeros(env.k)
        self.alpha = alpha
        self.second_half_return = 0
        self.type = type

        if type == "gradient":
            self.H = np.zeros(env.k)
            self.average_reward = 0
            self.alpha = special_param

        elif type == "optimistic":
            self.Q = np.ones(env.k) * special_param
        else:
            self.Q = np.zeros(env.k)

        if type == "UCB":
            self.c = special_param

        if type == "epsilon-greedy":
            self.epsilon = special_param

    def learn(self, steps):
        for step in range(steps):
            self.act()

    def policy(self):
        if self.type == "gradient":
            pi = softmax(self.H)
            return np.random.choice(a = np.arange(len(self.H)), size = 1, p = pi) #picks an action according to the distribution

        elif self.type == "UCB":
            if np.min(self.N) == 0: return np.argmin(self.N) #UCB maximizes untaken actions
            else: return np.argmax(self.Q + (self.c * ((np.log(self.t) / self.N) ** 0.5)))

        elif self.type == "optimistic": return np.argmax(self.Q)

        else:
            if random.random() < self.epsilon: return random.randrange(10)
            else: return np.argmax(self.Q)

    def act(self):
        self.t += 1
        action = self.policy()
        reward = self.env.step(action)
        if self.t > 100000: self.second_half_return += reward
        self.update(action, reward)

    def update(self, action, reward):
        self.N[action] += 1
        if self.type == "gradient":
            self.average_reward += (1 / self.t) * (reward - self.average_reward)
            pi = softmax(self.H)
            self.H = self.H - (self.alpha * (reward - self.average_reward) * pi)
            self.H[action] = self.H[action] + (self.alpha * (reward - self.average_reward))
            #multiplication by 1 - pi(A) not necessary since we added a factor of pi(A) in the previous line
        else:
            if self.alpha == 0:
                self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])
            else:
                self.Q[action] += self.alpha * (reward - self.Q[action])

def parameter_test(type, alpha, params):
    RUNS = 100
    avg_results = np.zeros(len(params))
    for run in range(RUNS):
        results = []
        for param in params:
            print("Testing", type, "with parameter =", param, "run", run + 1, "/", RUNS)
            agent = Q_agent(type, nonstationary_bandit(10), alpha, param)
            agent.learn(200000)
            metric = agent.second_half_return / 100000
            results.append(metric)
        avg_results = avg_results + np.array(results)
    return avg_results / RUNS

if __name__ == '__main__':
    params = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]

    plt.plot(parameter_test("epsilon-greedy", 0, params))
    plt.plot(parameter_test("epsilon-greedy", 0.1, params))
    plt.plot(parameter_test("UCB", 0, params))
    plt.plot(parameter_test("UCB", 0.1, params))
    plt.plot(parameter_test("optimistic", 0.1, params))
    plt.plot(parameter_test("gradient", 0, params))

    plt.legend(["epsilon-greedy", "epsilon-greedy constant step 0.1", "UCB", "UCB constant step 0.1", "optimistic", "gradient"])
    plt.xlabel("Special Parameter")
    plt.ylabel("Average Return Over Last 100000 Steps")
    plt.xticks(np.arange(len(params)), params)
    plt.show()

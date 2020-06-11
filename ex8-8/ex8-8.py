#RL: An Introduction exercise 8.8
#Comparison of uniform and on-policy sampling
#Gaussian reward distribution env
#Solution by Matt Deible

import numpy as np
import matplotlib.pyplot as plt
import math
import random

class gaussian_env:
    def __init__(self, states, action_space, b ):
        self.b = b
        self.states = states
        self.action_space = action_space
        self.rewards = np.random.normal(0, 1, (states, action_space, b))
        self.transition = np.random.randint(states, size = (states, action_space, b))

    def generate_trajectory(self, agent): #generates trajectory of state-action pairs for on-policy updating
        states = [0]
        actions = []
        while True:
            actions.append(agent.epsilon_greedy(states[-1]))
            if random.random() < 0.1:
                return states, actions
            else:
                states.append(self.transition[states[-1], actions[-1], random.randrange(b)])

    def sample_greedy_return(self, agent): #generates a sample return from the agent's greedy policy
        S = 0
        G = 0
        while True:
            A = agent.greedy(S)
            if random.random() < 0.1:
                return G
            else:
                branch = random.randrange(b)
                G += self.rewards[S, A, branch]
                S = self.transition[S, A, branch]

    def estimate_start_value(self, agent, sample_size):
        if self.b == 1:
            return self.get_expected_value(0, agent, 0) #with no branching, it is much faster and more accurate to calculate true average returns
        else:
            total_returns = 0
            for i in range(sample_size):
                total_returns += self.sample_greedy_return(agent)

            return (total_returns / sample_size)

    def get_expected_value(self, S, agent, depth):
        if depth == 100: #at 100 iterations deep, true value is calculated to a great precision (0.9^100 is uncalculated)
            return 0
        A = agent.greedy(S)
        value = 0
        for branch in range(self.b): #there is no discount, but only 90 percent chance of reaching the next state (other 10 percent is termination)
            value += (1 / b) * (self.rewards[S, A, branch] + (0.9 * self.get_expected_value(self.transition[S, A, branch], agent, depth + 1)))
        return value

class planning_agent:
    def __init__(self, env, type):
        self.type = type
        self.epsilon = 0.1
        self.env = env
        self.Q = np.zeros((env.states, env.action_space))
        self.computational_updates = 0

    def greedy(self, S):
        return np.random.choice(np.argwhere(self.Q[S] == np.amax(self.Q[S])).flatten())

    def epsilon_greedy(self, S):
        if random.random() > self.epsilon:
            return(self.greedy(S))
        else:
            return(random.randrange(0, self.env.action_space))

    def update(self, S, A):
        self.computational_updates += 1
        self.Q[S, A] = 0
        for branch in range(self.env.b):
            R = self.env.rewards[S, A, branch]
            Sp = self.env.transition[S, A, branch]
            self.Q[S, A] += (1 / self.env.b) * (R + (0.9 * np.max(self.Q[Sp])))

    def improve_Q(self, record_freq, max_updates):
        greedy_policy_values = []

        if self.type == "uniform":
            while True:
                for S in range(self.env.states):
                    for A in range(self.env.action_space):
                        if self.computational_updates % record_freq == 0:
                            greedy_policy_values.append(self.value_of_greedy())
                        if self.computational_updates >= max_updates:
                            return greedy_policy_values
                        self.update(S, A)

        elif self.type == "on-policy":
            while True:
                states, actions = env.generate_trajectory(self)
                for i in range(len(states)):
                    if self.computational_updates % record_freq == 0:
                        greedy_policy_values.append(self.value_of_greedy())
                    if self.computational_updates >= max_updates:
                        return greedy_policy_values
                    self.update(states[i], actions[i])

    def value_of_greedy(self):
        return env.estimate_start_value(self, 1000)

if __name__ == '__main__':
    branching_factors = [1, 3, 10]
    sweeping_types = ["uniform", "on-policy"]
    states = 1000

    repititions = 200
    data_points = 101 #number of points to record
    total_updates = states * 20
    record_freq = total_updates / (data_points - 1)

    x = np.arange(0, total_updates + 1, record_freq)
    legend_names = []

    for b in branching_factors:
        for type in sweeping_types:

            legend_names.append(type + " sweeping b = " + str(b))
            avg_comp_values = np.zeros(data_points)

            for rep in range(repititions):
                print(type, "sweeping b =", b, "repitition", rep)

                env = gaussian_env(states, 2, b)
                agent = planning_agent(env, type)
                comp_values = agent.improve_Q(record_freq, total_updates)
                avg_comp_values = avg_comp_values + np.array(comp_values)

            plt.plot(x, avg_comp_values / repititions)

    plt.legend(legend_names)
    plt.title("Uniform vs On-policy updating")
    plt.show()

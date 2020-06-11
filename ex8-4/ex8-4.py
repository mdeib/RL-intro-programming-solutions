#RL: An Introduction exercise 8.4
#Comparison of Dyna-Q+ exploration methods
#Gridworld Env
#Solution by Matt Deible

import numpy as np
import matplotlib.pyplot as plt
import math
import random

class maze_env:
    def __init__(self, type):
        self.type = type
        self.height = 6
        self.width = 9
        self.action_space = 4
        self.time_step = 0
        if type == "Blocking": self.blocked = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
        elif type == "Shortcut": self.blocked = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]
        self.reset()

    def reset(self):
        self.state = (0, 3)

    def step(self, action): #[up, down, right, left]
        self.time_step += 1

        if self.type == "Blocking" and self.time_step == 1000:
            self.blocked.remove((2, 0))
            self.blocked.append((2, 8))

        if self.type == "Shortcut" and self.time_step == 3000:
            self.blocked.remove((2, 8))

        if action == 0: next_state = (min(self.state[0] + 1, self.height - 1), self.state[1]) #up
        elif action == 1: next_state = (max(self.state[0] - 1, 0), self.state[1]) #down
        elif action == 2: next_state = (self.state[0], max(self.state[1] - 1, 0)) #left
        elif action == 3: next_state = (self.state[0], min(self.state[1] + 1, self.width - 1)) #right

        if not next_state in self.blocked: self.state = next_state

        if self.state == (5, 8):
            self.reset()
            return self.state, 1
        return self.state, 0

class dynaQ:
    def __init__(self, type, env):
        self.type = type
        self.env = env

        self.epsilon = 0.1
        self.alpha = 1
        self.kappa = 0.002
        self.discount = 0.95

        self.Q = np.zeros((env.height, env.width, env.action_space))
        self.tau = np.zeros((env.height, env.width, env.action_space))
        self.model = np.zeros((env.height, env.width, env.action_space, 3))

        if self.type == "DynaQ+": #pg 168 footnote specifies that DynaQ+ model should be initialized with reward zero and sent back to thier own state
            for j in range(env.height):
                for i in range(env.width):
                    for a in range(env.action_space):
                        self.model[j, i, a, 1] = j
                        self.model[j, i, a, 2] = i

    def learn(self, S, A, R, Sp):
        self.Q[S][A] += self.alpha * (R + (self.discount * np.max(self.Q[Sp])) - self.Q[S][A])

    def update_model(self, S, A, R, Sp):
        self.model[S][A, 0] = R
        self.model[S][A, 1] = Sp[0]
        self.model[S][A, 2] = Sp[1]

    def plan(self, n):
        for update in range(n):
            not_visited = True #picks state randomly until it has been visited
            while not_visited:
                S = (random.randrange(0, self.env.height), random.randrange(0, self.env.width))
                if np.sum(self.model[S]) > 0:
                    not_visited = False
            state_model = self.model[S]

            not_acted = True #picks action from that state randomly until it has been acted
            while not_acted:
                A = (random.randrange(0, self.env.action_space))
                if np.sum(state_model[A]) > 0:
                    not_acted = False

            R = state_model[A, 0]
            if self.type == "DynaQ+": R += (self.kappa * np.sqrt(self.tau[S][A])) #DynaQ plus takes into account time since action when planning
            Sp = (int(state_model[A, 1]), int(state_model[A, 2]))

            self.learn(S, A, R, Sp) #learn from model

    def epsilon_greedy(self, state):
        if random.random() > self.epsilon:
            if self.type == "DynaQ+ experimental": #experimental DyanQ plus takes into account time since action in greedy selection
                exploration_Q = self.Q[state] + (self.kappa * np.sqrt(self.tau[state]))
                best_actions = np.argwhere(exploration_Q == np.amax(exploration_Q)).flatten()
            else:
                best_actions = np.argwhere(self.Q[state] == np.amax(self.Q[state])).flatten()
            action = np.random.choice(best_actions)
        else:
            action = random.randrange(0, self.env.action_space)

        self.tau = self.tau + 1 #update time since state action pair
        self.tau[state][action] = 0
        return action

def test(agent_type, env_type, steps, repititions, n):
    total_rewards = np.zeros(steps + 1)
    for rep in range(repititions):
        print("Testing:", agent_type, "in", env_type, "Maze on repitition", rep + 1, "/", repititions)
        env = maze_env(env_type)
        agent = dynaQ(agent_type, env)
        rewards = np.zeros(steps + 1)

        for t in range(steps):
            S = env.state               #a
            A = agent.epsilon_greedy(S) #b
            Sp, R = env.step(A)         #c
            agent.learn(S, A, R, Sp)    #d
            agent.update_model(S, A, R, Sp)   #e
            agent.plan(n)              #f
            rewards[t + 1] = rewards[t] + R

        total_rewards = total_rewards + rewards
    return (total_rewards / repititions)

if __name__ == '__main__':
    environments = ["Blocking", "Shortcut"]
    steps = [3000, 6000]
    agents = ["DyanQ", "DynaQ+", "DynaQ+ experimental"]
    repititions = 100

    fig, ax = plt.subplots(len(environments))

    for i, env_type in enumerate(environments):
        for agent_type in agents:
            ax[i].plot(test(agent_type, env_type, steps[i], repititions, 10))
            ax[i].title.set_text(env_type + " maze")
            ax[i].legend(agents)

    plt.show()

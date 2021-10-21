import matplotlib.pyplot as plt
import numpy as np


def mplotLearning(scores, filename, wait=None, x=None, window=100):
    # N = len(scores[0])
    R = scores.shape[0]
    N = scores.shape[1]
    print(R, N)
    running_avg = np.empty((R, N))
    running_var = np.empty((R, N))
    for i in range(R):
        for t in range(N):
            running_avg[i, t] = np.mean(scores[i, max(0, t - window):(t + 1)])
            running_var[i, t] = np.var(scores[i, max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.title(f"{filename}Time{wait}")

    #     for i,c in zip(range(R),["green","orange","blue"]):
    #         plt.plot(x, running_avg[i,:],color=c)
    #         plt.fill_between(x, running_avg[i,:] + np.sqrt(running_var[i,:]),
    #                          running_avg[i,:] - np.sqrt(running_var[i,:]), facecolor=c, alpha=0.1)
    for i in range(R):
        plt.plot(x, running_avg[i, :])
        plt.fill_between(x, running_avg[i, :] + np.sqrt(running_var[i, :]),
                         running_avg[i, :] - np.sqrt(running_var[i, :]), alpha=0.1)
    plt.savefig(filename)

def plotLearning(scores, filename, wait=None, x=None, window=100):
    N = len(scores)
    running_avg = np.empty(N)
    running_var = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
        running_var[t] = np.var(scores[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(1,N+1)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.title(f"Time{wait}")
    plt.plot(x, running_avg)
    plt.fill_between(x, running_avg + np.sqrt(running_var), running_avg - np.sqrt(running_var), facecolor='gray', alpha=0.1)
    plt.savefig(filename)


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

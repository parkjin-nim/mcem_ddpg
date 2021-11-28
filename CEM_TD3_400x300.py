import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import pybullet_envs
import time
from Bipedal_fig_r2 import plotLearning


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr+=1

    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_size, self.mem_cntr)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions= self.action_memory[batch]
        rewards= self.reward_memory[batch]
        states_= self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir='tmp/cem-td3_400x300'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_cem-td3')

        # input is observation + n_actions
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        q1 = self.q1(q1_action_value)
        return q1

    def save_checkpoint(self):
        print("....saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(".....loading checkpoint....")
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha,input_dims,fc1_dims,fc2_dims,
                 n_actions,name,chkpt_dir='tmp/cem-td3_400x300'):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_cem-td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # 4 actions corresponds to the components of action vector
        # TD3 is based deterministic pg. no need of sigma
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob = T.tanh(self.mu(prob)) # if action is
        return prob

    def save_checkpoint(self):
        print("....saving checkpoint ....")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(".... loading checkpoint....")
        self.load_state_dict(T.load(self.checkpoint_file))

class GeneticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir='tmp/cem-td3_400x300'):
        super(GeneticNetwork,self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        #self.n_outputs = n_outputs
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'cem-td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # for continuous, we take n_actions=2 is mean & sigma
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.to(self.device)

        self.to(device)

    def forward(self, observation):
        # to put obs of env(numpy) to state of pytorch.nn(tensor)
        # . then cuda float tensor is different from just float tensor
        state = T.tensor(observation, dtype=T.float).to(device)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = T.tanh(self.fc3(x))
        return x # x is size 2 output, mu & sigma

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = torch.tanh(self.fc2(x))
    #     x = torch.tanh(self.fc3(x))
    #     return x.cpu().data

    def set_weights(self, weights):
        s_size = self.input_dims[0]
        fc1_size = self.fc1_dims
        fc2_size = self.fc2_dims
        a_size = self.n_actions

        # separate the weights for each layer
        fc1_end = (s_size * fc1_size) + fc1_size
        fc1_W = T.from_numpy(weights[:s_size * fc1_size].reshape(s_size, fc1_size))
        fc1_b = T.from_numpy(weights[s_size * fc1_size:fc1_end])
        fc2_end = fc1_end + (fc1_size * fc2_size) + fc2_size
        fc2_W = T.from_numpy(weights[fc1_end:fc1_end + (fc1_size * fc2_size)].reshape(fc1_size, fc2_size))
        fc2_b = T.from_numpy(weights[fc1_end + (fc1_size * fc2_size):fc2_end])
        fc3_W = T.from_numpy(weights[fc2_end:fc2_end + (fc2_size * a_size)].reshape(fc2_size, a_size))
        fc3_b = T.from_numpy(weights[fc2_end + (fc2_size * a_size):])

        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
        self.fc3.weight.data.copy_(fc3_W.view_as(self.fc3.weight.data))
        self.fc3.bias.data.copy_(fc3_b.view_as(self.fc3.bias.data))

    def get_weights_dim(self):
        return (self.input_dims[0] + 1) * self.fc1_dims + (self.fc1_dims + 1) * self.fc2_dims + (self.fc2_dims + 1) * self.n_actions

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = env.reset()
        for t in range(max_t):
            state = T.from_numpy(state).float().to(device)
            action = self.forward(state).to(device)
            action = action.cpu().detach().numpy()
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])

            state, reward, done, _ = env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                        layer2_size,n_actions=n_actions,name='target_actor')
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions,name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions,name='target_critic_2')

        self.noise = noise

        # copy exactly, instead of soft update rule
        self.update_network_parameters(tau=1)

    #####
        self.cem_actor = GeneticNetwork(alpha, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='cem_actor')

    def cem_act(self, state):
        state = T.from_numpy(state).float().to(device)
        with T.no_grad():
            action = self.cem_actor.forward(state)
        return action
    #####

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        # add some exploratory noise
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        # pass it from Torch tensor to the Cuda tensor(our device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
                         T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)

        # Without below, broken(not convergin) if elements of min and max are not all equal
        target_actions = T.clamp(target_actions, self.min_action[0],
                                 self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)
        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # Loss fuction for Q_theta(Critic) is L2 of Q_targ - Q
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        # Loss function for Pi_pi(actor) is the same as -Q
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + \
                                        (1-tau) * target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
                                        (1-tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_model(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        self.cem_actor.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        self.cem_actor.load_checkpoint()




def cem(agent, best_weight, max_t=5000, gamma=1.0, pop_size=50, elite_frac=0.2, sigma=0.5):
    n_elite = int(pop_size * elite_frac)
    # Define the candidates and get the reward of each candidate
    weights_pop = [best_weight + (sigma * np.random.standard_normal(agent.cem_actor.get_weights_dim())) for i in range(pop_size)]
    rewards = np.array([agent.cem_actor.evaluate(weights, gamma, max_t) for weights in weights_pop])

    # Select best candidates from collected rewards
    elite_idxs = rewards.argsort()[-n_elite:]
    elite_weights = [weights_pop[i] for i in elite_idxs]
    best_weight = np.array(elite_weights).mean(axis=0)

    # reward = agent.cem_actor.evaluate(best_weight, gamma=1.0)
    # def evaluate(self, weights, gamma=1.0, max_t=5000):
    data = []
    agent.cem_actor.set_weights(best_weight)
    score = 0.0
    state = env.reset()
    for t in range(max_t):
        state = T.from_numpy(state).float().to(device)
        action = agent.cem_actor.forward(state).to(device)
        action = action.cpu().detach().numpy()
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        state_, reward, done, _ = env.step(action)
        score += reward * math.pow(gamma, t)
        data.append([state, action, reward, state_, done])#
        state = state_#
        if done:
            break

    return score, best_weight, data

def td3(agent):
    observation = env.reset()
    done = False
    score = 0.0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        score += reward
        observation = observation_

    return score

def simulate(agent,eval=False):
    if eval:
        agent.load_model()
        agent.warmup = 0 # don't warmup
        for i in range(3):
            state = env.reset()
            max_t = 2000
            for j in range(max_t):
                action = agent.choose_action(state)
                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break
        env.close()
    return


def main():
    global env
    # env_id = "LunarLanderContinuous-v2"
    env_id = 'BipedalWalker-v3'
    # env_id = "HalfCheetahBulletEnv-v0"
    env = gym.make(env_id)
    env.seed(101); np.random.seed(101)
    best_score = env.reward_range[0]
    print("best_score, action_size, obs_size:", best_score, env.action_space.shape[0],env.observation_space.shape[0])

    agent = Agent(alpha=0.001,beta=0.001,input_dims=env.observation_space.shape, tau=0.005,
                  env=env,batch_size=100,layer1_size=400, layer2_size=300,
                  n_actions=env.action_space.shape[0])

    # True to simulate trained parameters
    simulate(agent, False)

    n_games = 1000
    score_history = []

    sigma = 0.5
    W = sigma * np.random.standard_normal(agent.cem_actor.get_weights_dim())

    for game in range(1, n_games+1):
        td3_score = td3(agent)
        cem_score, W_, data = cem(agent,W,gamma=0.999)
        print(f'td3_score:{td3_score}, cem_score:{cem_score}')
        W = W_

        if cem_score > td3_score:
            score = cem_score
            for s, a, r, s_, d in data:
                agent.remember(s,a,r,s_,d)
        else:
            score = td3_score

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if game % 10 == 0:
            print('episode ',game, 'score %.1f'%score, 'average score %.1f'%avg_score)
            if avg_score > best_score:
                best_score = avg_score
                agent.save_model()

    filename = 'CEM-TD_400x300_' + env_id + '_' + str(n_games) + '_games' + '4'
    figure = 'plots/' + filename + '.png'
    np.save("npy/" + filename + ".npy", np.array(score_history))
    plotLearning(score_history, figure)
    simulate(agent, False)
    return

if __name__ == "__main__":
    global device
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    main()

"""
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
If lander moves away from landing pad it loses reward back.
Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points.
Each leg ground contact is +10. Firing main engine is -0.3 points each frame.
Solved is 200 points. Landing outside landing pad is possible.
Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
Action is two real values vector from -1 to +1.
First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power.
Engine can't work with less than 50% power.
Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.
"""
# for MAC os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import gym
import math
import time
from Utils import plotLearning, ReplayBuffer

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class CriticNetwork(nn.Module):
    def __init__(self, env, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/cem_ddpg'):
        super(CriticNetwork, self).__init__()
        self.env = env
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'cem_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        # self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        # self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.initialize_net()
        self.optimizer = optim.Adam(self.parameters(), lr=beta,weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialize_net(self):
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        f4 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        # state_value = F.relu(state_value)
        # action_value = F.relu(self.action_value(action))
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        # state_action_value = T.add(state_value, action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class GeneticNetwork(nn.Module):
    def __init__(self, env, lr, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/cem_ddpg'):
        super(GeneticNetwork,self).__init__()
        self.env = env
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        #self.n_outputs = n_outputs
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'cem_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # for continuous, we take n_actions=2 is mean & sigma
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # send entire network to device
        self.to(self.device)

    def forward(self, observation):
        # to put obs of env(numpy) to state of pytorch.nn(tensor)
        # . then cuda float tensor is different from just float tensor
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = T.tanh(self.fc3(x))
        return x # x is size 2 output, mu & sigma

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
        return (self.input_dims[0] + 1) * self.fc1_dims + (self.fc1_dims + 1) * self.fc2_dims + (
                    self.fc2_dims + 1) * self.n_actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, env, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300,
                 batch_size=64):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # add actor for es
        self.es_actor = GeneticNetwork(env, alpha, input_dims, 64, 64, n_actions=n_actions, name='actor')

        self.actor = GeneticNetwork(env, alpha, input_dims, 64, 64, n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(env, beta, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name='critic')
        self.target_actor = GeneticNetwork(env, alpha, input_dims, 64, 64,n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(env, beta, input_dims, fc1_dims, fc2_dims,n_actions=n_actions, name='target_critic')

        self.update_network_parameters(tau=1)

    def es_evaluate(self, weights, gamma=1.0, max_t=5000):
        self.es_actor.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()

        for t in range(max_t):
            # es_choose_action
            # first, add batch dim. from numpy
            state = T.tensor([state], dtype=T.float).to(self.es_actor.device)
            #state = T.from_numpy(state).float().to(self.es_actor.device)
            mu = self.es_actor.forward(state).to(self.es_actor.device)
            mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.es_actor.device)
            action = mu_prime.cpu().detach().numpy()[0]
            #action = np.array(action).reshape((1,))

            state, reward, done, _ = self.env.step(action)
            #if reward > 0: print("e", end="")
            episode_return += reward * math.pow(gamma, t)
            if done:
                break

        return episode_return

    def es_learn(self):
        if self.memory.mem_cntr < self.batch_size: return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # no actor learning

        self.update_critic_network_parameters()

        # no actor update parameter

    def update_critic_network_parameters(self, tau=None):
        if tau is None: tau = self.tau
        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        critic_state_dict = dict(critic_params)
        target_critic_state_dict = dict(target_critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)


    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(),dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size: return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)


if __name__ == '__main__':
    #env_id = 'MountainCarContinuous-v2'
    env_id = 'LunarLanderContinuous-v2'
    #env_id = 'HalfCheetahBulletEnv-v0'
    env = gym.make(env_id)
    best_score = env.reward_range[0]
    action_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape
    print("best_score", best_score)
    print("action_size:", action_size)
    print("observation_size", obs_size)
    env.seed(200);
    np.random.seed(200)
    agent = Agent(env=env, alpha=0.0001, beta=0.001,
                input_dims=obs_size, tau=0.001, batch_size=64,
                fc1_dims=400, fc2_dims=300, n_actions=action_size)

    EVALUATE = True
    CONTINUE = False
    if EVALUATE:
        agent.load_models()
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
    if CONTINUE:
        agent.load_models()

    n_games = 1000
    scores = []
    best_score = -np.inf
    start = time.perf_counter()
    baton = 1  # int(n_games/2)

    # Cross Entropy Method starts
    pop_size = 100
    elite_frac = 0.2
    sigma = 0.5
    n_elite = int(pop_size * elite_frac)
    w_dim = agent.es_actor.get_weights_dim()
    best_weight = sigma * np.random.randn(w_dim)

    for game in range(1, n_games + 1):
        weights_pop = [best_weight + (sigma * np.random.randn(w_dim)) for i in range(pop_size)]
        rewards = np.array([agent.es_evaluate(weights, gamma=1.0, max_t=1000) for weights in weights_pop])
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.es_evaluate(best_weight, gamma=1.0)
        scores.append(reward)
        avg_score = np.mean(scores[-100:])

        if game % 10 == 0:
            print('episode ',game, 'score %.1f'%reward, 'average score %.1f'%avg_score)
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        # Teach DDPG value function with elitism
        baton += 1
        agent.actor.set_weights(best_weight)
        agent.target_actor.set_weights(best_weight)
        for j in range(20):
            observation = env.reset()
            done = False
            agent.noise.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                agent.remember(observation, action, reward, observation_, done)
                agent.es_learn()
                observation = observation_

        if avg_score > 0:
            #print('Episode {}\tAverage Score: {:.2f}'.format(i, avg_score))
            break

    # Relay baton to DDPG, if avg_score > 0
    for game in range(baton, n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if game % 10 == 0:
            print('episode ',game, 'score %.1f'%score, 'average score %.1f'%avg_score)
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

    finish = time.perf_counter()
    wait = round(finish - start, 2)
    print(f'Finished in {wait} seconds')

    filename = 'CEM_DDPG_' + env_id + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    np.save("npy/" + filename + ".npy", np.array(scores))
    plotLearning(scores, figure_file, wait=wait)

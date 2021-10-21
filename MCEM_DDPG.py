"""
CEM multiprocessing with seedsync.
1.Goal
- 96 population is divided into 8 multiprocessing tasks x (12 batches x  w_dim)
- weights = weights_pop.reshape(self.n_agents, -1, self.w_dim)
2.Effect
- 1132 seconds(18 min.) per 100 iter.
- Requires Seed Synchronization.
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
import pybullet_envs
import math
import time
import multiprocessing
from Utils import plotLearning, ReplayBuffer

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        super(OUActionNoise, self).__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def get_es_noise(self, t_noise):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * t_noise #np.random.normal(size=self.mu.shape) #
        self.x_prev = x
        return x
    def get_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape) #
        self.x_prev = x
        return x

class CriticNetwork(nn.Module):
    def __init__(self, env, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/mcemddpg'):
        super(CriticNetwork, self).__init__()
        self.env = env
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_mcemddpg')

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
                 chkpt_dir='tmp/mcemddpg'):
        super(GeneticNetwork,self).__init__()
        self.env = env
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_mcemddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # send entire network to device
        self.to(self.device)

    def forward(self, observation):
        # obs(numpy) to T.tensor to put nn.Module.
        # Note cuda float tensor is different from just float tensor
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
    def __init__(self, env_id, number, n_actions, n_agents, alpha, beta, gamma, tau,
                 input_dims, fc1_dims=400, fc2_dims=300, batch_size=64,
                 pop_size=None, best_weight=None, sigma=None):
        # Multi-agent Env.
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.number = number
        self.name = "W%02i" % number
        self.n_agents = n_agents
        self.n_actions = n_actions

        # CEM network. low precision network 64x64
        self.pop_size = pop_size
        self.best_weight = best_weight
        self.seed = None
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.sigma = sigma
        self.es_actor = GeneticNetwork(self.env, alpha, input_dims, fc1_dims=64, fc2_dims=64,
                                       n_actions=self.n_actions, name='es_actor')
        self.w_dim = self.es_actor.get_weights_dim()
        if self.best_weight is None:
            self.best_weight = self.sigma * np.random.randn(self.w_dim)
        self.es_actor.set_weights(self.best_weight)

        # DDPG actor-critic block
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # set default gamma 0.99
        self.tau = tau
        self.memory = ReplayBuffer(1000000, input_dims, n_actions)
        self.batch_size = batch_size
        # add actor for es
        self.actor = GeneticNetwork(self.env, self.alpha, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(self.env, self.beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name='critic')
        self.target_actor = GeneticNetwork(self.env, self.alpha, input_dims, fc1_dims, fc2_dims,
                                           n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(self.env, self.beta, input_dims, fc1_dims, fc2_dims,
                                           n_actions=n_actions, name='target_critic')

        self.update_network_parameters(tau=1)

    def set_weight_seed(self, weight, seed):
        self.best_weight = weight
        self.es_actor.set_weights(weight)
        self.seed = seed

    def es_evaluate(self, return_dict, gamma=1.0, max_t=5000):
        # synchronize pop. # distribute pop to ith multiprocessing task
        np.random.seed(self.seed)
        random_noise = np.random.randn(self.pop_size, self.w_dim)
        weights_pop = np.array([self.best_weight + (self.sigma * random_noise[i, :]) for i in range(self.pop_size)])
        weights = weights_pop.reshape(self.n_agents, -1, self.w_dim)
        weights = weights[self.number]

        for i,weight in enumerate(weights):
            # Seed Sync module
            # set identical seed before reset()
            self.env.seed(200)
            state = self.env.reset()
            # st identical numpy seed before QU noise generation
            np.random.seed(self.seed)
            t_noise = np.random.randn(max_t, self.n_actions)
            # reset QU noise every pop's episode
            self.noise.reset()

            self.es_actor.set_weights(weight)
            episode_return = 0.0
            self.t = []
            for t in range(max_t):
                state = T.tensor([state], dtype=T.float).to(self.es_actor.device)
                mu = self.es_actor.forward(state).to(self.es_actor.device)
                #mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.es_actor.device)
                mu_prime = mu + T.tensor(self.noise.get_es_noise(t_noise[t]), dtype=T.float).to(self.es_actor.device)
                action = mu_prime.cpu().detach().numpy()[0]
                # clip if action value is over the limit
                action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
                state, reward, done, _ = self.env.step(action)
                episode_return += reward * math.pow(gamma, t)
                if done:
                    self.t.append(t)
                    break
            subname = "%02i" % i
            return_dict[self.name+subname] = episode_return

    # test module to check if the seed of global agent is synchronized with m x agents
    def g_evaluate(self, weights, gamma=1.0, max_t=5000):
        # Seed Sync module. set identical seed before reset()
        self.env.seed(200)
        state = self.env.reset()
        # st identical numpy seed before QU noise generation
        np.random.seed(self.seed)

        self.es_actor.set_weights(weights)
        episode_return = 0.0
        self.t = []
        for t in range(max_t):
            state = T.tensor([state], dtype=T.float).to(self.es_actor.device)
            mu = self.es_actor.forward(state).to(self.es_actor.device)
            mu_prime = mu + T.tensor(self.noise.get_noise(), dtype=T.float).to(self.es_actor.device)
            action = mu_prime.cpu().detach().numpy()[0]

            # clip if over the range of env's action limit
            action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
            state_, reward, done, _ = self.env.step(action)

            # Bonus long-term effect behavior by rolling out the trial episode of elite parameter
            self.remember(state, action, reward, state_, done)
            self.learn(critic_only=True)

            state = state_
            episode_return += reward * math.pow(gamma, t)
            if done:
                self.t.append(t)
                break
        return episode_return

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()[0]

        # clip if over the range of env's action limit
        action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.es_actor.save_checkpoint()
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.es_actor.load_checkpoint()
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, critic_only=False):
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

        target = rewards + self.gamma*critic_value_  #self.gamma = 0.99 default
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        if not critic_only:
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic.forward(states, self.actor.forward(states))
            actor_loss = T.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

        self.update_network_parameters(critic_only=critic_only)

    def update_network_parameters(self, tau=None, critic_only=False):
        if tau is None: tau = self.tau

        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        critic_state_dict = dict(critic_params)
        target_critic_state_dict = dict(target_critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)
        # self.target_critic.load_state_dict(critic_state_dict, strict=False)

        if not critic_only:
            actor_params = self.actor.named_parameters()
            target_actor_params = self.target_actor.named_parameters()
            actor_state_dict = dict(actor_params)
            target_actor_state_dict = dict(target_actor_params)
            for name in actor_state_dict:
                 actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                     (1-tau)*target_actor_state_dict[name].clone()
            self.target_actor.load_state_dict(actor_state_dict)
            #self.target_actor.load_state_dict(actor_state_dict, strict=False)

class MAgent():
    def __init__(self,env_id, n_actions, n_agents, alpha, beta, gamma, tau,
                    input_dims, fc1_dims, fc2_dims,
                    pop_size, best_weight, sigma):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.env_id = env_id
        for i in range(self.n_agents):
            self.agents.append(Agent(env_id=env_id, number=i, n_actions=n_actions, n_agents=n_agents,
                        alpha=alpha, beta=beta, gamma=gamma, tau=tau,
                        input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                        pop_size=pop_size, best_weight=best_weight, sigma=sigma))

    def set_weight_seed(self, weight, seed):
        for agent in self.agents:
            agent.set_weight_seed(weight,seed)

    def evaluate(self, gamma):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []

        for i in range(self.n_agents):
            p = multiprocessing.Process(target=self.agents[i].es_evaluate, args=(return_dict, gamma))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()

        ord_tvalues = sorted(return_dict.items(), key=lambda x:x[0])
        ord_values = [j for _,j in ord_tvalues]
        return ord_values

def run(cnt):
    # Sim. Environment
    #env_id = 'MountainCarContinuous-v0'
    #env_id = "LunarLanderContinuous-v2"
    env_id = 'BipedalWalker-v3'
    #env_id = "HalfCheetahBulletEnv-v0"
    # env_id = "AntBulletEnv-v0"
    env = gym.make(env_id)
    best_score = env.reward_range[0]
    action_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape
    print(f"action_size:{action_size} observation_size: {obs_size} best_score: {best_score}")

    s = [200, 300, 400, 500, 600]   # 5 random seed
    print("=====Run Counter=====: ",cnt)
    env.seed(s[cnt]);
    np.random.seed(s[cnt])

    # === Global agent setting=== #
    g_agent_name = 100              # global agent id in multiprocessing
    n_agents = 8                       # my Mac has 8 cores
    pop_size = 96                   # 8core x 12tasks = 96 as population
    sigma = 0.5                     # 0.5 fixed sigma to prevent starvation
    elite_frac = 0.2                # 20% elitism
    n_elite = int(pop_size * elite_frac)
    n_games = 3000                  # running episodes
    scores = []                     # score charts
    fc1_dims, fc2_dims = 400, 300   # 400x300 Actor critic. 64x64 CEM long-term bonus net
    g_agent = Agent(env_id=env_id, number=g_agent_name, n_actions=action_size, n_agents=n_agents,
                    alpha=0.0001, beta=0.001, gamma=0.99, tau=0.001,
                  input_dims=obs_size, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                  pop_size=pop_size, best_weight=None, sigma=sigma)

    filename = 'MCEM_DDPG_'+env_id[:5]+'_alpha_'+str(g_agent.alpha)+'_beta_'+str(g_agent.beta) + '_fc1_' + str(fc1_dims) + \
               '_fc1_' + str(fc2_dims) + '_' + str(n_games) + 'games_' + str(cnt) + 'times'
    # test mode
    EVALUATE, CONTINUE = False, False
    if EVALUATE:
        g_agent.load_models()
        for i in range(3):
            state = env.reset()
            max_t = 2000
            for j in range(max_t):
                #action = g_agent.es_choose_action(state)
                action = g_agent.choose_action(state)
                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break
        env.close()
        return
    if CONTINUE:
        g_agent.load_models()
        scores = list(np.load("..."))
        "!! Change run number !!"

    # === Multiple agent CEM === #
    m_agent = MAgent(env_id=env_id, n_actions=action_size, n_agents=n_agents,
                     alpha=0.0001, beta=0.001, gamma=0.99, tau=0.001,
                    input_dims=obs_size, fc1_dims=400, fc2_dims=300,
                    pop_size=pop_size, best_weight=None, sigma=sigma)

    start = time.perf_counter()
    best_score = -np.inf
    w_dim = g_agent.w_dim
    best_weight = g_agent.best_weight
    for game in range(1, n_games + 1):
        # get a random value, and set it as random seed.
        seed_value = np.random.randint(2 ** 32 - 1)
        np.random.seed(seed_value)
        # set the seed for g_agent & m_agent
        g_agent.set_weight_seed(best_weight, seed_value)
        m_agent.set_weight_seed(best_weight, seed_value)

        # generate pop w/ predefined random noise sequence
        random_noise = np.random.randn(pop_size, w_dim)
        weights_pop = np.array([best_weight + (sigma * random_noise[i, :]) for i in range(pop_size)])
        # get rewards from m_agents
        rewards = np.array(m_agent.evaluate(gamma=1.0))
        rewards = rewards.flatten()

        # get elitism
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        # Bonus long-term effect behavior by rolling out the trial episode of elite parameter
        es_reward = g_agent.g_evaluate(best_weight, gamma=1.0)
        #baton += 1                                     # Sequential combination test
        # g_agent.actor.set_weights(best_weight)        # Direct search guidance test
        # g_agent.target_actor.set_weights(best_weight) # Setting target parameter test
        # for j in range(1): #range(int(np.sqrt(game))):# Decay # of bonus

        # DDPG
        observation = env.reset()
        done = False
        score = 0
        g_agent.noise.reset()
        while not done:
            action = g_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            g_agent.remember(observation, action, reward, observation_, done)
            g_agent.learn(critic_only=False)
            score += reward
            observation = observation_

        scores.append(score) # second
        avg_score = np.mean(scores[-100:])
        if game % 10 == 0:
            print('episode ',game, 'reward %.1f'%es_reward, 'score %.1f'%score, 'average score %.1f'%avg_score)
            if avg_score > best_score:
                best_score = avg_score
                g_agent.save_models()
                wait = round(time.perf_counter() - start, 2)
                print(f'Finished in {wait} seconds')

    finish = time.perf_counter()
    wait = round(finish - start, 2)
    print(f'Finished in {wait} seconds')
    #agent.save_models()

    figure_file = 'plots/' + filename + '.png'
    np.save("npy/" + filename + ".npy", np.array(scores))
    plotLearning(scores, figure_file, wait=wait)

if __name__ == '__main__':
    for i in range(1,5):
        run(i)


# actor target을 elite로.
# cem과 elite 중 좋은 걸 target으로

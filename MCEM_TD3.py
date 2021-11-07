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

# class OUActionNoise():
#     def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
#         super(OUActionNoise, self).__init__()
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.reset()
#
#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
#
#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
#                 self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x
#
#     def get_es_noise(self, t_noise):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
#                 self.sigma * np.sqrt(self.dt) * t_noise #np.random.normal(size=self.mu.shape) #
#         self.x_prev = x
#         return x
#     def get_noise(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
#                 self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape) #
#         self.x_prev = x
#         return x

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/mcem_td3'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_mcem_td3')
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
    def __init__(self,alpha,input_dims,fc1_dims=400,fc2_dims=300,n_actions=None,name=None,chkpt_dir='tmp/mcem_td3'):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'mcem_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
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
        prob = T.tanh(self.mu(prob))
        return prob

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
        self.mu.weight.data.copy_(fc3_W.view_as(self.mu.weight.data))
        self.mu.bias.data.copy_(fc3_b.view_as(self.mu.bias.data))

    def get_weights_dim(self):
        return (self.input_dims[0] + 1) * self.fc1_dims + (self.fc1_dims + 1) * self.fc2_dims + (
                    self.fc2_dims + 1) * self.n_actions

    def save_checkpoint(self):
        print("....saving checkpoint ....")
        T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        print(".... loading checkpoint....")
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, env_id, number, n_actions, n_agents, alpha, beta, gamma, tau, input_dims,
                 fc1_dims=400, fc2_dims=300, batch_size=100, update_actor_interval=2, warmup=1000, max_size=1000000,
                 pop_size=None, best_weight=None, sigma=0.5, noise=0.1):
        # Multi-agent Env.
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.number = number
        self.name = "W%02i" % number
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.max_action = self.env.action_space.high
        self.min_action = self.env.action_space.low

        # CEM network. low precision network
        self.pop_size = pop_size
        self.best_weight = best_weight
        self.seed = None
        #self.QUnoise = OUActionNoise(mu=np.zeros(n_actions))
        self.sigma = sigma
        self.es_actor = ActorNetwork(alpha, input_dims, fc1_dims=64, fc2_dims=64,n_actions=n_actions, name='es_actor')
        self.w_dim = self.es_actor.get_weights_dim()
        if self.best_weight is None:
            self.best_weight = self.sigma * np.random.randn(self.w_dim)
        self.es_actor.set_weights(self.best_weight)

        # TD3 networks
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.update_actor_iter = update_actor_interval
        self.actor = ActorNetwork(self.alpha, input_dims, fc1_dims, fc2_dims,n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(self.beta, input_dims, fc1_dims, fc2_dims,n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(self.beta, input_dims, fc1_dims, fc2_dims,n_actions=n_actions,name='critic_2')
        self.target_actor = ActorNetwork(self.alpha, input_dims, fc1_dims, fc2_dims,n_actions=n_actions,name='target_actor')
        self.target_critic_1 = CriticNetwork(self.beta, input_dims, fc1_dims,fc2_dims,n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(self.beta, input_dims, fc1_dims,fc2_dims,n_actions=n_actions, name='target_critic_2')
        self.noise = noise
        self.update_network_parameters(tau=1)

    def set_weight_seed(self, weight, seed):
        self.best_weight = weight
        self.es_actor.set_weights(weight)
        self.seed = seed

    def es_evaluate(self, return_dict, gamma=1.0, max_t=5000):
        # synchronize pop.
        # distribute pop to ith multiprocessing task
        np.random.seed(self.seed)
        random_noise = np.random.randn(self.pop_size, self.w_dim)
        weights_pop = np.array([self.best_weight + (self.sigma * random_noise[i, :]) for i in range(self.pop_size)])
        weights = weights_pop.reshape(self.n_agents, -1, self.w_dim)
        weights = weights[self.number]

        for i,weight in enumerate(weights):
            self.env.seed(200)
            state = self.env.reset()
            np.random.seed(self.seed)
            #t_noise = np.random.randn(max_t, self.n_actions)

            self.es_actor.set_weights(weight)
            episode_return = 0.0

            for t in range(max_t):
                state = T.tensor([state], dtype=T.float).to(self.es_actor.device)
                mu = self.es_actor.forward(state).to(self.es_actor.device)
                # add some exploratory noise
                #mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.es_actor.device)
                #mu_prime = mu + T.tensor(self.noise.get_es_noise(t_noise[t]), dtype=T.float).to(self.es_actor.device)
                # mu_prime = mu + T.tensor(t_noise[t], dtype=T.float).to(self.es_actor.device)
                mu_prime = mu

                action = mu_prime.cpu().detach().numpy()[0]
                # clip if action value is over the limit
                action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
                state, reward, done, _ = self.env.step(action)
                episode_return += reward * math.pow(gamma, t)
                if done:
                    print("t, return: ", t, episode_return)
                    # give long-live bonus to long-term reward
                    # episode_return += t
                    break
            # Verify
            subname = "%02i" % i
            return_dict[self.name+subname] = episode_return

    # test module to check if the seed of global agent is synchronized with m x agents
    def g_evaluate(self, weights, gamma=1.0, max_t=5000):
        # Seed Sync module. set identical seed before reset()
        self.env.seed(200)
        state = self.env.reset()
        np.random.seed(self.seed)

        self.es_actor.set_weights(weights)
        episode_return = 0.0

        for t in range(max_t):
            state = T.tensor([state], dtype=T.float).to(self.es_actor.device)
            mu = self.es_actor.forward(state).to(self.es_actor.device)
            # mu_prime = mu + T.tensor(self.noise.get_noise(), dtype=T.float).to(self.es_actor.device)
            # mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.es_actor.device)
            mu_prime = mu

            action = mu_prime.cpu().detach().numpy()[0]
            action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
            state_, reward, done, _ = self.env.step(action)

            self.remember(state, action, reward, state_, done)
            self.learn(critic_only=True)

            state = state_
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return


    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        # add some exploratory noise   self.noise()
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def learn(self, critic_only=False):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # Without below, broken(not converging) if elements of min and max are not all equal
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

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
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.learn_step_cntr += 1
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        #
        if not critic_only:
            self.actor.optimizer.zero_grad()
            actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
            actor_loss = -T.mean(actor_q1_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

        self.update_network_parameters(critic_only=critic_only)

    def update_network_parameters(self, tau=None, critic_only=False):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)

        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_1_state_dict[name].clone()
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_2_state_dict[name].clone()
        #
        if not critic_only:
            for name in actor_state_dict:
                actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                         (1 - tau) * target_actor_state_dict[name].clone()
            self.target_actor.load_state_dict(actor_state_dict)

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)

    def save_model(self):
        self.es_actor.save_checkpoint()
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_model(self):
        self.es_actor.load_checkpoint()
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

class MAgent():
    def __init__(self,env_id, n_actions, n_agents, alpha, beta, gamma, tau, input_dims, fc1_dims, fc2_dims, 
            pop_size, best_weight, sigma):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.env_id = env_id
        for i in range(self.n_agents):
            self.agents.append(Agent(env_id=env_id,number=i,n_actions=n_actions, n_agents=n_agents, alpha=alpha, beta=beta,
                gamma=gamma, tau=tau, input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, update_actor_interval=2,
                warmup=1000, max_size=1000000,pop_size=pop_size, best_weight=best_weight, sigma=sigma))

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
    #env_id = "LunarLanderContinuous-v2"
    env_id = 'BipedalWalker-v3'
    #env_id = "HalfCheetahBulletEnv-v0"
    env = gym.make(env_id)
    best_score = env.reward_range[0]
    action_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape
    print(f"action_size:{action_size} observation_size: {obs_size} best_score: {best_score}")

    s = [200, 300, 400, 500, 600]   # 5 random seed
    print("=====Run Counter=====: ",cnt)
    env.seed(s[cnt]);
    np.random.seed(s[cnt])

    # 400x300 for Critic. 64x64 for Actor
    # === Global agent setting=== #
    g_agent_name = 100              # global agent id in multiprocessing
    n_agents = 8                    # my Mac has 8 cores
    pop_size = 96                   # 8core x 12tasks = 96 as population
    sigma = .5                       # 0.5 fixed sigma to prevent starvation
    elite_frac = 0.2                # 20% elitism
    n_elite = int(pop_size * elite_frac)
    n_games = 1000                  # running episodes
    scores = []                     # score charts
    fc1_dims, fc2_dims = 400, 300   # 400x300 Actor critic. 64x64 CEM long-term bonus net
    # g_agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape,
    #               tau=0.005, env=env, batch_size=100, layer1_size=fc1_dims, layer2_size=fc2_dims,
    #               n_actions=env.action_space.shape[0])
    g_agent = Agent(env_id=env_id, number=g_agent_name, n_actions=action_size, n_agents=n_agents, alpha=0.0001, beta=0.001,
                gamma=0.99, tau=0.001,input_dims=obs_size, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                  pop_size=pop_size, best_weight=None, sigma=sigma)

    filename = 'MCEM_TD3_400x300_'+env_id[:5]+'_alpha_'+str(g_agent.alpha)+'_beta_'+str(g_agent.beta) + '_fc1_' + str(fc1_dims) + \
               '_fc1_' + str(fc2_dims) + '_' + str(n_games) + 'games_' + str(cnt) + 'times'

    EVALUATE = False
    CONTINUE = False
    if EVALUATE:
        g_agent.load_model()
        g_agent.warmup = 0 # don't warmup
        for i in range(3):
            state = env.reset()
            max_t = 5000
            for j in range(max_t):
                action = g_agent.choose_action(state)
                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break
        env.close()
        return
    if CONTINUE:
        g_agent.load_model()

    # === Multiple agent CEM === #
    m_agent = MAgent(env_id=env_id, n_actions=action_size, n_agents=n_agents, alpha=0.0001, beta=0.001, gamma=0.99,tau=0.001,
                input_dims=obs_size, fc1_dims=400, fc2_dims=300,pop_size=pop_size, best_weight=None, sigma=sigma)

    start = time.perf_counter()
    best_score = -np.inf
    w_dim = g_agent.w_dim
    best_weight = g_agent.best_weight

    for game in range(1, n_games + 1):
        seed_value = np.random.randint(2 ** 32 - 1)
        np.random.seed(seed_value)

        g_agent.set_weight_seed(best_weight, seed_value)
        m_agent.set_weight_seed(best_weight, seed_value)

        random_noise = np.random.randn(pop_size, w_dim)
        weights_pop = np.array([best_weight + (sigma * random_noise[i, :]) for i in range(pop_size)])
        rewards = np.array(m_agent.evaluate(gamma=0.99999))
        rewards = rewards.flatten()

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)
        es_reward = g_agent.g_evaluate(best_weight, gamma=0.99999)
        print("es_reward: ", es_reward)

        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = g_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            g_agent.remember(observation, action, reward, observation_, done)
            g_agent.learn()
            score += reward
            observation = observation_
        print("pg_score: ", score)
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if game % 10 == 0:
            print('episode ',game, 'score %.1f'%score, 'average score %.1f'%avg_score)
            if avg_score > best_score:
                best_score = avg_score
                g_agent.save_model()

    finish = time.perf_counter()
    wait = round(finish - start, 2)
    print(f'Finished in {wait} seconds')

    #filename = 'CEM_TD3_64x64_' + env_id + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    np.save("npy/" + filename + ".npy", np.array(scores))
    plotLearning(scores, figure_file, wait=wait)

if __name__ == "__main__":
    for i in range(1):
        run(i)
#    run(0)

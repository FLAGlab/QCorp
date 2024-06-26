# Library Imports
import numpy as np
import os

from gym import Env

import torch
from torch.nn import functional as F

from replay_buffers.PER import PrioritizedReplayBuffer
from replay_buffers.utils import LinearSchedule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(torch.nn.Module):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, input_dim: int, beta: float, density: int = 512, name: str = 'critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.H4 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.Q = torch.nn.Linear(density, 1, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.hstack((state, action))
        value = F.relu(self.H1(value))
        value = F.relu(self.H2(value))
        value = self.drop(value)
        value = F.relu(self.H3(value))
        value = F.relu(self.H4(value))
        value = self.Q(value)
        return value

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + '.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + '.pth'))


class Actor(torch.nn.Module):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, input_dim: int, n_actions: int, alpha: float, density: int = 512, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.H4 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.mu = torch.nn.Linear(density, n_actions, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = F.relu(self.H1(state))
        action = F.relu(self.H2(action))
        action = self.drop(action)
        action = F.relu(self.H3(action))
        action = F.relu(self.H4(action))
        action = torch.tanh(self.mu(action))
        return action

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + '.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + '.pth'))


class Agent:
    def __init__(self, env: Env, datapath: str = 'tmp/', n_games: int = 250, training: bool = True,
                 alpha=0.0001, beta=0.002, gamma=0.99, tau=0.004,
                 batch_size: int = 64, noise: str = 'normal',
                 per_alpha: float = 0.6, per_beta: float = 0.4):

        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.tau = tau
        self.n_actions: int = env.action_space.shape[0]
        self.obs_shape: int = env.observation_space.shape[0]
        self.datapath = datapath
        self.n_games = n_games
        self.optim_steps: int = 0
        self.max_size: int = 2500000
        self.is_training = training
        self.memory = PrioritizedReplayBuffer(self.max_size, per_alpha)
        self.beta_scheduler = LinearSchedule(n_games, per_beta, 1.0)

        self.batch_size = batch_size
        self.noise = noise
        self.max_action = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
        self.min_action = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)

        self.actor = Actor(self.obs_shape, self.n_actions, alpha, name='actor')
        self.critic = Critic(self.obs_shape + self.n_actions, beta, name='critic')
        self.target_actor = Actor(self.obs_shape, self.n_actions, alpha, name='target_actor')
        self.target_critic = Critic(self.obs_shape + self.n_actions, beta, name='target_critic')

        if self.noise == 'normal':
            self.noise_param: float = 0.1
            self.noise_scheduler = LinearSchedule(n_games, self.noise_param, self.noise_param / 20.0)

        else:
            raise NotImplementedError("This noise is not implmented!")

        self._update_networks()

    def _update_networks(self, tau: float = 1.0):

        for critic_weights, target_critic_weights in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_weights.data.copy_(tau * critic_weights.data + (1.0 - tau) * target_critic_weights.data)

        for actor_weights, target_actor_weights in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_weights.data.copy_(tau * actor_weights.data + (1.0 - tau) * target_actor_weights.data)

    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        if self.noise == 'normal':
            noise_param = self.noise_scheduler.value(self.optim_steps)
            noise = np.random.uniform(0.0, noise_param, action.shape)
            action += torch.as_tensor(noise, dtype=torch.float32, device=device)

        return action

    def _action_scaling(self, action: torch.Tensor) -> torch.Tensor:
        neural_min = -1.0 * torch.ones_like(action)
        neural_max = 1.0 * torch.ones_like(action)

        env_min = self.min_action * torch.ones_like(action)
        env_max = self.max_action * torch.ones_like(action)

        return ((action - neural_min) / (neural_max - neural_min)) * (env_max - env_min) + env_min

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
        self.actor.eval()
        state = torch.as_tensor(observation, dtype=torch.float32, device=device)
        action = self.actor.forward(state)

        if self.is_training:
            action = self._add_exploration_noise(action)

        action = self._action_scaling(action)

        return action.detach().cpu().numpy()

    def save_models(self):
        self.actor.save_model(self.datapath)
        self.target_actor.save_model(self.datapath)
        self.critic.save_model(self.datapath)
        self.target_critic.save_model(self.datapath)

    def load_models(self):
        self.actor.load_model(self.datapath)
        self.target_actor.load_model(self.datapath)
        self.critic.load_model(self.datapath)
        self.target_critic.load_model(self.datapath)

    def optimize(self):
        if len(self.memory._storage) < self.batch_size:
            return

        beta = self.beta_scheduler.value(self.optim_steps)
        state, action, reward, new_state, done, weights, indices = self.memory.sample(self.batch_size, beta)

        state = torch.as_tensor(np.vstack(state), dtype=torch.float32, device=device)
        action = torch.as_tensor(np.vstack(action), dtype=torch.float32, device=device)
        done = torch.as_tensor(np.vstack(1 - done), dtype=torch.float32, device=device)
        reward = torch.as_tensor(np.vstack(reward), dtype=torch.float32, device=device)
        new_state = torch.as_tensor(np.vstack(new_state), dtype=torch.float32, device=device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()

        Q_target = self.target_critic.forward(new_state, self.target_actor.forward(new_state))
        Y = reward + (done * self.gamma * Q_target)
        Q = self.critic.forward(state, action)
        TD_errors = torch.sub(Y, Q)

        # Weight TD errors
        weighted_TD_errors = torch.mul(TD_errors, weights)
        zero_tensor = torch.zeros_like(weighted_TD_errors)

        # Compute & Update Critic losses
        critic_loss = F.mse_loss(weighted_TD_errors, zero_tensor)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.train()
        self.critic.eval()

        # Compute & Update Actor losses
        actor_loss = torch.mean(-1.0 * self.critic.forward(state, self.actor(state)))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        td_errors: np.ndarray = TD_errors.detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        self._update_networks(self.tau)
        self.optim_steps += 1
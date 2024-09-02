# %% argparse
import argparse

parser = argparse.ArgumentParser(description="Testing Reinforce Algorithm")
parser.add_argument(
    "--env_name",
    type=str,
    default="CartPole-v1",
    help="Official name of the gymnasium environment to test the algorithm",
)
parser.add_argument("--gamma", type=float, help="forgetting factor", default=0.999)
parser.add_argument("--episode_batch_number", type=int, help="Episode batch number", default=1)
parser.add_argument("--Adam_lr", type=float, help="Initial adam optimizer learning rate", default=5e-12)

# %% importing all torch and gym related modules
import numpy as np

# torch related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch util imports
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# gym related imports
import gymnasium as gym

# These wrappers help add more custom capabilities and features to the environment
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, TransformReward, TransformObservation


# %% creating agent
class parameterized_policy(nn.Module):
    def __init__(self, input_dim, layers_sizes: tuple, output_dim, device="cuda:0"):
        super(parameterized_policy, self).__init__()
        self.device = device
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, layers_sizes[0]))
        # Middle ones
        for it in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[it], layers_sizes[it + 1]))
        # Last layer
        layers.append(nn.Linear(layers_sizes[-1], output_dim))
        # Register layers as a ModuleList # ((This is important to have a successful summary call))
        self.Layers = nn.ModuleList(layers)
        self.to(self.device)

    def forward(self, net_input):
        x = net_input
        for i in range(len(self.Layers) - 1):
            x = self.Layers[i](x)
            x = torch.tanh(x)
        x = self.Layers[-1](x)
        out = torch.softmax(x, dim=1)
        return out


# %% checking the model architecture
agent_model_toy_example = parameterized_policy(10, [20, 10, 5], 2)
summary(agent_model_toy_example, (10,), device="cuda")
# %% parsing arguments
args = parser.parse_args()


# %% Reinforce Algorithm definition

"""
This implementation is only useful for episodic tasks.
"""


class REINFORCE:
    def __init__(self, episode_batch_number: int, policy_net: nn.Module, gamma: float, learning_rate: float):
        self.K = episode_batch_number
        self.policy_net = policy_net
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._Slist = []
        self._Alist = []
        self._Rlist = []
        self.G_list = []
        self.A_list = []
        self.S_list = []
        self.episode_counter = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def policy(self, obs):
        self.policy_net.eval()
        obs = torch.tensor(obs, dtype=torch.float32).to(policy_net.device)
        obs = obs.view(1, -1)
        probs = self.policy_net(obs)
        action = torch.multinomial(probs, 1).item()  # output would be in {0,1,...,n_action}
        return action

    def _delete_temp_lists(self):
        self._Slist = []
        self._Alist = []
        self._Rlist = []

    def _delete_main_lists(self):
        self.G_list = []
        self.A_list = []
        self.S_list = []

    def episode_started(self, first_obs):
        self._Slist.append(first_obs)

    def step_update(self, nex_obs, reward, terminated, truncated, action):
        self._Alist.append(action)
        self._Rlist.append(reward)
        if not (terminated or truncated):  # This implementation is not useful for continual tasks which are truncated!
            self._Slist.append(nex_obs)
        else:
            R_array = np.array(self._Rlist, dtype=np.float32)
            episode_length = len(self._Rlist)
            for t in range(episode_length):
                forget_array = np.array([self.gamma ** (k - t - 1) for k in range(t + 1, episode_length + 1)])
                modified_G = self.gamma**t * np.sum((R_array[t:] * forget_array))

                self.A_list.append(self._Alist[t])
                self.S_list.append(self._Slist[t])
                self.G_list.append(modified_G)

            self._delete_temp_lists()
            self.episode_counter += 1
            if self.episode_counter == self.K:
                self._policy_update()
                self._delete_main_lists()
                self.episode_counter = 0

    def _policy_update(self):
        self.policy_net.train()
        self.A_list = torch.tensor(self.A_list, dtype=torch.int64).to(self.policy_net.device)
        self.S_list = torch.tensor(self.S_list, dtype=torch.float32).to(self.policy_net.device)
        self.G_list = torch.tensor(self.G_list, dtype=torch.float32).to(self.policy_net.device)

        self.optimizer.zero_grad()
        probs = self.policy_net(self.S_list)
        P_A_S = torch.gather(probs, 1, self.A_list.view(-1, 1))
        loss = -self.G_list.view(1, -1) @ torch.log(P_A_S).view(-1, 1)
        loss.backward()
        self.optimizer.step()


# %% creating a simple environment
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
total_episode = 1000
video_path = "./runs/reinforce_algorithm/videos/"
tensor_path = "./runs/reinforce_algorithm/"
writer = SummaryWriter()
# creating the environment
env = gym.make(args.env_name, render_mode="rgb_array")
# env = RecordVideo(env, video_path, episode_trigger=lambda eps: eps % 50 == 0)
env = RecordEpisodeStatistics(env)
env = TransformReward(env, lambda r: r / 50)  # It is important to have normalized reward
env = TransformObservation(env, lambda obs: obs / np.array([4.8, 1, 0.418, 1]))  # I had no idea how to set coefs for velocities

obs_dim = env.unwrapped.observation_space.shape  # output is [4,]
action_dim = env.unwrapped.action_space.n  # output is [1,]
policy_net = parameterized_policy(obs_dim[0], [100, 4], action_dim)
summary(policy_net, obs_dim)
reinforce_agent = REINFORCE(args.episode_batch_number, policy_net, gamma=args.gamma, learning_rate=args.Adam_lr)

for episode_iter in range(total_episode):
    observation, info = env.reset()
    reinforce_agent.episode_started(observation)
    terminated = False
    truncated = False
    sumreward = 0
    while not (terminated or truncated):
        action = reinforce_agent.policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        reinforce_agent.step_update(next_observation, reward, terminated, truncated, action)
        # env.render()
        sumreward += reward
    episodic_retrun = info["episode"]["r"]
    episodic_length = info["episode"]["l"]
    writer.add_scalar("training/return", episodic_retrun, global_step=episode_iter)
    writer.add_scalar("training/length", episodic_length, global_step=episode_iter)
    writer.add_scalar("training/sumreward", sumreward, global_step=episode_iter)


env.close()
writer.flush()
writer.close()

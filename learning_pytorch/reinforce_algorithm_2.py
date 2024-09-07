# %% argparse
import argparse

parser = argparse.ArgumentParser(description="Testing Reinforce Algorithm")
parser.add_argument(
    "--env_name",
    type=str,
    default="CartPole-v1",
    help="Official name of the gymnasium environment to test the algorithm",
)
parser.add_argument("--gamma", type=float, help="forgetting factor", default=0.99)
parser.add_argument("--episode_batch_number", type=int, help="Episode batch number", default=1)
parser.add_argument("--Adam_lr", type=float, help="Initial adam optimizer learning rate", default=2**-10)
args = parser.parse_args()
# %% importing all torch and gym related modules
import numpy as np
from time import time

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


# %% creating agent with predetermined number of layers
class parameterized_policy(nn.Module):
    def __init__(self, input_dim, output_dim, device="cuda:0"):
        super(parameterized_policy, self).__init__()
        self.device = device
        # First layer
        self.fc1 = nn.Linear(input_dim, 10)
        # self.bn = nn.BatchNorm1d(10, track_running_stats=True)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, output_dim)
        self.to(self.device)

    def forward(self, net_input):
        x = net_input / torch.tensor([4.8, 5, 0.418, 5]).to(self.device)
        x = F.tanh(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = F.softmax(self.fc4(x), dim=1)
        return out


# agent_model_toy_example = parameterized_policy(10, 2, device="cuda:0")
# summary(agent_model_toy_example, (10,), device="cuda")
# %% Reinforce Algorithm definition

"""
This implementation is only useful for episodic tasks.
"""


class REINFORCE:
    def __init__(self, policy_net: nn.Module, gamma: float, learning_rate: float):
        self.policy_net = policy_net
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.trajectory = []
        self.probs = []

    def policy(self, obs):
        action = None
        self.policy_net.eval()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.policy_net.device)
        obs = obs.view(1, -1)
        prob = self.policy_net(obs)
        action = torch.multinomial(prob, 1).item()  # output would be in {0,1,...,n_action}
        self.probs.append(prob[0, action])
        return action

    def update_trajectory(self, reward, terminated_truncated):
        self.trajectory.append(reward)
        if terminated_truncated:
            self.update_policy()
            self.trajectory = []
            self.probs = []

    def update_policy(self):
        self.policy_net.train()
        policy_loss = torch.tensor(0.0).to(self.policy_net.device)
        for t, reward in enumerate(self.trajectory):
            G = 0
            for l, _reward_ in enumerate(self.trajectory[t:]):
                G += self.gamma ** (l - t) * _reward_
            coeff = G * self.gamma**t
            policy_loss += -torch.log(self.probs[t]) * coeff
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


# %% creating a simple environment

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

total_episode = 10000

tensor_path = "./runs/reinforce_algorithm/"
writer = SummaryWriter()
# creating the environment
env = gym.make(args.env_name, render_mode="human")
env = RecordEpisodeStatistics(env)

obs_dim = env.unwrapped.observation_space.shape
action_dim = env.unwrapped.action_space.n  # output is [1,]
policy_net = parameterized_policy(obs_dim[0], action_dim)
summary(policy_net, (obs_dim[0],))
reinforce_agent = REINFORCE(policy_net, gamma=args.gamma, learning_rate=args.Adam_lr)

for episode_iter in range(total_episode):
    print(f"iteration: {episode_iter}")
    observation, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = reinforce_agent.policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        reinforce_agent.update_trajectory(reward, terminated or truncated)
        # env.render()
    episodic_retrun = info["episode"]["r"]
    episodic_length = info["episode"]["l"]
    writer.add_scalar("training/return", episodic_retrun, global_step=episode_iter)
    writer.add_scalar("training/length", episodic_length, global_step=episode_iter)
    writer.flush()

env.close()
writer.close()

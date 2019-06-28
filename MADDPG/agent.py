import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
from model import *
import torch.optim as optim
import numpy as np

# This is a DDPG agent with a centralized critic
class SingleAgent():
    def __init__(self, env, agent_id, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2):
        # Environment Info
        self.observation_dim = env.observation_space[agent_id].shape[0]
        self.action_dim = env.action_space[agent_id].n
        self.num_agents = env.n
        self.agent_id = agent_id
        # hyperparams
        self.tau = tau
        self.gamma = gamma

        #critic_input_dim = int(np.sum([env.observation_space[agent].shape[0] for agent in range(env.n)])) + int(np.sum([env.action_space[agent].n for agent in range(env.n)]))
        critic_input_dim = int(np.sum([env.observation_space[agent].shape[0] for agent in range(env.n)]))
        actor_input_dim = self.observation_dim

        self.critic = CentralizedCritic(critic_input_dim, self.action_dim * env.n,  hidden_dims=[1024,512,300], output_dim=1)
        self.actor = Actor(actor_input_dim, hidden_dims=[512,128], output_dim=self.action_dim)
        self.critic_target = CentralizedCritic(critic_input_dim, self.action_dim * env.n,  hidden_dims=[1024,512,300], output_dim=1)
        self.actor_target = Actor(actor_input_dim, hidden_dims=[512,128], output_dim=self.action_dim)

        # assume an agent has information about other agents' policies
        ## self.other_agent_policy_inference = Actor(actor_input_dim, hidden_dims=[512,128], output_dim=1)

        # Copy params to target network params
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_actor_output(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0]
        return action

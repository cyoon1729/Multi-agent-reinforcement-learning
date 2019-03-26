import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
from model import *

# This is a DDPG agent with a centralized critic
class SingleAgent():
    def __init__(self, env, num_agents, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2):
        # Environment Info
        self.observation_dim = env.observation_space[0].shape[0]
        self.action_dim = env.action_space[0].n
        self.num_agents = num_agents
        
        # hyperparams
        self.tau = tau
        self.gamma = gamma

        critic_input_dim = self.observation_dim * num_agents + num_agents # observation_dim * num_agent -> the 'x' input | num_agents -> number of actions
        actor_input_dim = self.observation_dim

        self.critic = CentralizedCritic(critic_input_dim, hidden_dims=[1024,512,300], output_dim=1)
        self.actor = Actor(actor_input_dim, hidden_dims=[512,128], output_dim=1)
        self.critic_target = CentralizedCritic(critic_input_dim, hidden_dims=[1024,512,300], output_dim=1)
        self.actor_target = Actor(actor_input_dim, hidden_dims=[512,128], output_dim=1)
        
        # I will assume an agent has information about other agents' policies
        # self.other_agent_policy_inference = Actor(actor_input_dim, hidden_dims=[512,128], output_dim=1)
        
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
        action = action.detach().numpy()[0,0]

        return action

    def update(self, experience_batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = experience_batch
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        # Calculate actor (policy) loss
        policy_loss = self.critic.forward(state_batch, self.actor.forward(state_batch))
        policy_loss = -policy_loss.mean() 
        
        # calculate critic loss
        curr_Qvals = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(next_state_batch)
        new_Qvals = self.critic_target.forward(next_state_batch, next_actions)
        update_Qvals = reward_batch + self.gamma * new_Qvals
        critic_loss = self.critic_criterion(curr_Qvals - update_Qvals)
        
        # Update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # Update target_networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def cuda_update(self, experience_batch):
        # Will implement when I have access to cuda 
        pass
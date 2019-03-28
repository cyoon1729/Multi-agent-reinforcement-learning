import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

#[1024, 512,300]
class CentralizedCritic(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(CentralizedCritic, self).__init__()
        """
        Param:
          - input_dum: this is num_agent * observation_dimensions + num_agents * action_dimensions ???
          - hidden_dims: array of len 3 of hidden layer dimensions
          - output_dim is by default set to 1 (outputs single Q value)
        """
        self.linear1 = nn.Linear(input_dim, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.linear4 = nn.Linear(hidden_dims[2], output_dim)
    
    
    def forward(self, states, actions):
        """
        Param:
          - states: observations of all agents, type: Tensor
          - actions: actions of all agents, type: Tensor
        Return:
          - Q value for agent i
        """
        x = torch.cat([states, actions], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x

        
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

# [512,128]
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Actor, self).__init__()
        """
        Param:
         - input_dim: observation dim of environment
         - hidden_dims: array of len 2 for hidden layer dimensions
         - output_dim: action dim of environment
        """
        self.linear1 = nn.Linear(input_dim, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], output_dim)
    
    def forward(self, state):
        """
        Param:
         - state: observation of agent i, type: Tensor
        Return:
         - deterministic actoin for agent i
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

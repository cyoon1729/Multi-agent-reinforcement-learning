import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

class CentralizedCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims, output_dim=1):
        super(CentralizedCritic, self).__init__()
        """
        Param:
          - input_dum: this is num_agent * observation_dimensions + num_agents * action_dimensions ???
          - hidden_dims: array of len 3 of hidden layer dimensions
          - output_dim is by default set to 1 (outputs single Q value)
        """
       # self.linear1 = nn.Linear(input_dim, hidden_dims[0])
       # self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
       # self.linear3 = nn.Linear(hidden_dims[1], hidden_dims[2])
       # self.linear4 = nn.Linear(hidden_dims[2], output_dim)
        """
        Param:
          - input_dim: num_agents * observation_dimensions
          - output_dim: 1 (single Q value)
        """
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.linear2 = nn.Linear(self.hidden_dims[0] + self.action_dim, self.hidden_dims[1])
        self.linear3 = nn.Linear(self.hidden_dims[1], self.hidden_dims[2])
        self.linear4 = nn.Linear(self.hidden_dims[2], self.output_dim)

    def forward(self, states, actions):
        """
        Param:
          - states: global observation (Float Tensor)
          - actions: actions of all agents (Float Tensor)
        Return:
          - array of Q value for agent i
        """
        #x = torch.cat([states, actions], 1)
        #x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        #x = F.relu(self.linear3(x))
        #x = self.linear4(x)

        x = F.relu(self.linear1(states))
        #print("state:" , str(x))
        #print("action:", str(actions))
        xs_concat = torch.cat([x, actions], 1)
        #print("xs_concat: ", str(xs_concat))
        #print("xs_concat_shape: ", str(xs_concat.shape()))
        xs_out = F.relu(self.linear2(xs_concat))
        xs_out = F.relu(self.linear3(xs_out))
        qvals = self.linear4(xs_out)

        return x

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Actor, self).__init__()
        """
        Param:
         - input_dim: observation dim of environment
         - output_dim: action dim of environment
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, state):
        """
        Param:
         - state: local observation, type: Tensor
        Return:
         - action
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

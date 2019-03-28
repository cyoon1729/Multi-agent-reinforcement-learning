from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from maddpg import MADDPG
import torch 
from torch.autograd import Variable
import numpy as np

scenario = scenarios.load("simple_tag.py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

agents = MADDPG(env, env.n)
state = env.reset()
actions  = agents.get_actions(state)
#print(actions)
state = np.array(state)
states_in = np.concatenate(state)
actions_in = np.concatenate(actions)

# states_in = torch.FloatTensor(states_in)
#actions_in = torch.FloatTensor(actions_in)

states_in = Variable(torch.from_numpy(states_in).float().unsqueeze(0))
actions_in = Variable(torch.from_numpy(actions_in).float().unsqueeze(0))

print("states in:", states_in)
print("actions in: ", actions_in)

value = agents.agents[0].critic.forward(states_in, actions_in)
print(value)
#state = torch.from_numpy(state).float().to(self.device)
#actions = torch.FloatTensor(actions)
#value = agents.agents[0].critic.forward(state, actions)
next_states, reward, done, _ = env.step(actions)


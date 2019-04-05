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


state = np.array(state)
states_in = np.concatenate(state)
actions_in = np.concatenate(actions)
states_in = Variable(torch.from_numpy(states_in).float().unsqueeze(0))
actions_in = Variable(torch.from_numpy(actions_in).float().unsqueeze(0))

# print("states in:", states_in)
# print("actions in: ", actions_in)

value = agents.agents[0].critic.forward(states_in, actions_in)

# print(value)
next_states, reward, done, _ = env.step(actions)


states = env.reset()
for _ in range(100):
    actions = agents.get_actions(state)
    new_states, reward, done, _ = env.step(actions)
    agents.replay_buffer.push(states, actions, reward, new_states, done)
    states = new_states

experience = agents.replay_buffer.sample(1)
s, a, r, ns, d = experience
s = np.concatenate(s)
a = np.concatenate(a)


experiences = agents.replay_buffer.sample(2)
ss, aa, rr, nnss, dd = experiences
# for s in ss:
#     print("-----")
#     print(s[0])
indvs = [s[agents.agents[3].agent_id] for s in ss]
ss = [np.concatenate(s_) for s_ in ss]
aa = [np.concatenate(a_) for a_ in aa]
rr = [r_.flatten()[agents.agents[3].agent_id] for r_ in rr]

# print(indvs)



ss = torch.FloatTensor(ss)
aa = torch.FloatTensor(aa)
rr = torch.FloatTensor(rr)
print("working input")
print(ss)
print(aa)
print(".............")
# print(rr)
value = agents.agents[0].critic.forward(ss, aa)
# print(rr + value)

#experiences = agents.replay_buffer.sample(1)
#agents.agents[0].update(experiences)
agents.update(10)
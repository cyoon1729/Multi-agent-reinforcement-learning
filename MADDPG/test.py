from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import torch
import numpy as np

from agent import DDPGAgent
from maddpg import MADDPG
from utils import MultiAgentReplayBuffer

def make_env(scenario_name, benchmark=False):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

env = make_env(scenario_name="simple_spread")
# for agents in range(env.n):
#     print(env.action_space[agents].n)

agents = [DDPGAgent(env, i) for i in range(env.n)]
state = env.reset()
# print(len(state))
# print(agents[0].get_action(state[0]))
states = torch.FloatTensor(np.concatenate(state))

actions = []
for agent in agents:
    actions.append(agent.get_action(state[agent.agent_id]))


next_state, _, _, _ = env.step(actions)
# print("next state: " + str(next_state))

actions = torch.cat(actions)
# print(states.unsqueeze(0))
# print(actions.unsqueeze(0))
# print(torch.cat([states, actions],0))
# print(agents[0].critic.forward(states, actions))



controller = MADDPG(env, 1000)
buffer = MultiAgentReplayBuffer(env.n, 1000)

state = env.reset()
for episodes in range(2):
    state = env.reset()
    for steps in range(5):
        
        actions = []
        for agent in agents:
            actions.append(agent.get_action(state[agent.agent_id]))
        
        next_state, reward, done, _ = env.step(actions)
        buffer.push(state, actions, reward, next_state, done)

obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
    state_batch, actions_batch, next_state_batch, done_batch = buffer.sample(5)

print("obs_batch" + str(obs_batch))
print("indiv_action_batch" + str(indiv_action_batch))
print("indiv_reward_batch" + str(indiv_reward_batch))
print("next_obs_batch" + str(next_obs_batch))

print("+++++++++++++++++++++++++++++++")

print("state_batch" + str(state_batch))
print("actions_batch" + str(actions_batch))
print("next_state_batch" + str(next_state_batch))
print("done_batch" + str(done_batch))

print("+++++++++++++++++++++++++++++++")

obs_batch_i = torch.FloatTensor(obs_batch[0])
print(obs_batch_i)
print(agent.actor.forward(obs_batch_i))

state_batch = torch.FloatTensor(state_batch)
actions_batch = torch.stack(actions_batch)
print(state_batch)
print(actions_batch)

print(agent.critic.forward(state_batch, actions_batch))


next_global_action = []
for agent in agents:
    obs_batch_i = torch.FloatTensor(obs_batch[agent.agent_id])
    indiv_action = agent.actor.forward(obs_batch_i)
    indiv_action = [agent.onehot_from_logits(indiv_action_j) for indiv_action_j in indiv_action]
    print(indiv_action)
    indiv_action = torch.stack(indiv_action)
    next_global_action.append(indiv_action)

print("+++++++++++")
print(next_global_action)
print(indiv_reward_batch[0])
ird_i = torch.FloatTensor(indiv_reward_batch[0])
ird_i = ird_i.view(ird_i.size(0), 1)
print(ird_i)
next_state_batch = torch.FloatTensor(next_state_batch)
next_global_action = torch.cat([actions for actions in next_global_action], 1)
print("next_global_action\n" + str(next_global_action))
print("global_next_state_batch\n" + str(next_state_batch))
print("Critic\n" + str(ird_i) + "\n" + str(agents[0].critic.forward(next_state_batch, next_global_action)) + "\n" + str(ird_i + agents[0].critic.forward(next_state_batch, next_global_action)))
#print(next_global_action)
#print(torch.cat([actions for actions in next_global_action], 1))
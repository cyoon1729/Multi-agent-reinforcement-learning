from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from maddpg import MADDPG

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

env = make_env(scenario_name="simple_tag")
#for agents in range(env.n):
#    print(env.action_space[agents].n)
#print(env.action_space.sample())
agents = MADDPG(env, env.n)
agents.train(max_episodes=100, max_steps=100, batch_size=50)

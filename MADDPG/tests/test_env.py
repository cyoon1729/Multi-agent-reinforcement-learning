from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from ..model import *
import maddpg 

scenario = scenarios.load("simple_tag.py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

agents = maddpg.MADDPG(env, env.n)
print(env.reset())

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

scenario = scenarios.load("simple_tag.py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

print(env.reset())

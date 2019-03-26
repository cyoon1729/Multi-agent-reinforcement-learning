from agent import *
from utils import *
import numpy as np 

class MADDPG():
    def __init__(self, env, num_agents, memory_maxlen=50000):
        self.env = env
        self.num_agents = num_agents

        # Initialize n agents with default params in agent.py -> class SingleAgent
        self.agents = [SingleAgent(self.env, i) for i in range(self.num_agents)]
        # Initialize replay_buffer
        self.replay_buffer = Memory(max_size = memory_maxlen)
        
    def get_actions(self, state):
        actions = []
        for agent in agents:
            action = agent.get_actor_output(state)
            action = action.detach().numpy()[0,0]
            actions.append(action)
        actions = np.stack(action)

        return actions

    def update(self, batch_size):
        experience_batch = replay_buffer.sample(batch_size)
        for agent in agents:
            agent.update(experience_batch)
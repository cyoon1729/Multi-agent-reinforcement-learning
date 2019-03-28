import sys
import numpy as np 
from agent import *
from utils import *
import torch 

class MADDPG():
    def __init__(self, env, num_agents, memory_maxlen=50000):
        self.env = env
        self.num_agents = num_agents

        # Initialize n agents with default params in agent.py -> class SingleAgent
        self.agents = [SingleAgent(self.env, i) for i in range(self.num_agents)]
        # Initialize replay_buffer
        self.replay_buffer = Memory(max_size = memory_maxlen)
        # Exploration noise
        #self.noise = OUNoise(self.env.action_space)
        
    def get_actions(self, states):
        actions = []
        for agent, state in zip(self.agents, states):
            #print(agent.agent_id, state)
            action = agent.get_actor_output(state)
            print(agent.agent_id, state, action)
            actions.append(action)
        actions = np.array(actions)
        return actions

    def update(self, batch_size):
        experience_batch = replay_buffer.sample(batch_size)
        for agent in agents:
            agent.update(experience_batch)
    
    def train(self, config):
        max_episodes = config.max_episodes
        max_steps = config.max_episodes
        batch_size = config.replay_batch_size
        rewards = []

        for episodes in range(max_episodes):
            states = self.env.reset()
            #self.noise.reset()
            epiode_reward = 0
            
            for steps in range(max_steps):
                actions = self.get_actions(states)
                #actions = [self.noise.get_action(action) for action in actions]
                new_states, rewards, dones, _ = self.env.step(actions)
                # we are only done when every agent is done
                if all(dones) or steps == max_steps - 1:
                    dones = [1 for _ in range(env.n)]    
                else:
                    dones = [0 for _ in range(env.n)]

                self.replay_buffer.push(states, actions, rewards, new_states, done)

                if len(self.replay_buffer) > batch_size:
                    self.update(batch_size)
                
                states = new_states
                episode_reward += np.sum(rewards)


                if all(dones):
                    rewards.append(episode_reward)
                    sys.stdout.write("episode: {}, reward: {}, rolling_10_average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                    break



                


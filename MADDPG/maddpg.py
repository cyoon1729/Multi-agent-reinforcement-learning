import sys
import numpy as np 
from agent import *
from utils import *
import torch 

"""
TODO: Fix the policy update mess. nested arrays
"""

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
            # print(agent.agent_id, state, action)
            actions.append(action)
        actions = np.array(actions)
        return actions
    
    def get_one_hot_actions(self):
        pass

    def update(self, batch_size):
        experiences = self.replay_buffer.sample(batch_size)
        states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = experiences

        # indiv_state_batch = [states[self.agent_id] for states in state_batch] # Observations of only agent i
        total_state_batch = [np.concatenate(states) for states in states_batch] # Concatenate observations of all agents
        action_batch = [np.concatenate(actions) for actions in actions_batch] # Concatenate action vectors of all agents
        # reward_batch = [rewards.flatten()[self.agent_id] for rewards in rewards_batch]  # Isolate rewards of agent i
        next_state_batch = [np.concatenate(next_states) for next_states in next_states_batch]

        # To tensors
        # indiv_state_batch = torch.FloatTensor(indiv_state_batch)
        total_state_batch = torch.FloatTensor(total_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        # reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        
        for agent in self.agents:
            
            """
            Obtain actions of all agents using current policies. 
            Psuedocode:

            1. for each state in state_batch:
                  Obtain action output (tensor) of each agent for all agents and store in tuple -> ( Tensor([a1]), Tensor([a2]), ..., Tensor([an]) ), where ai is action of agent i
                  concatenate tuple so that we have all actions in a single tensor -> Tensor([a1, a2, ... , an])
                  Store tensor to new_action_batch
           
            Output[1]: new_action_batch = [ Tensor([a1, a2, a3, a4, ..., an]_{state_1}, Tensor([a1, a2, a3, a4, ..., an]_{state_2}, ..., Tensor([a1, a2, a3, a4, ..., an]_{state_n}) ]

            2. Stack all tensors in new_action_batch

            Output[2]: new_action_batch = Tensor([[a1, a2, a3, a4, ..., an]_{state_1}, [a1, a2, a3, a4, ..., an]_{state_2}, ..., [a1, a2, a3, a4, ..., an]_{state_n}])      
            """
            new_action_batches = []
            for states in states_batch: # state_batch = [ [[n1] [n2] [n3] [n4] ... [nn]], [[n1] [n2] [n3] [n4] .. [nn]], .... ], states in state_batch = [[n1] [n2] [n3] [n4] ... [nn]]
                actions_tuple = ()
                for _agent in self.agents:
                    indiv_state = states[_agent.agent_id]
                    indiv_state = torch.FloatTensor(indiv_state)
                    action = _agent.actor.forward(indiv_state)
                    actions_tuple += (action,)

                new_action_batches.append(torch.cat(actions_tuple))
        
            new_action_batches = torch.stack(new_action_batches)
            
            """
            Compute policy loss
            """
            policy_loss = -(agent.critic.forward(total_state_batch, new_action_batches).mean())
            

            # Compute value loss

            #indiv_state_batch = [states[agent.agent_id] for states in states_batch]
            #indiv_state_batch = torch.FloatTensor(indiv_state_batch)
            #total_action_batch = torch.cat(actions)
         
            # indiv_reward_batch = [rewards.flatten()[self.agent_id] for rewards in rewards_batch] 
            
            
            
            # Update parameters
            # agent.actor_optimizer.zero_grad()
            # policy_loss.backward()
            # agent.actor_optimizer.step()

    
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

import sys
import numpy as np 
from agent import *
from utils import *
import torch 

"""
TODO: try on environment
"""

class MADDPG():
    def __init__(self, env, num_agents, memory_maxlen=50000):
        self.env = env
        self.num_agents = num_agents

        self.agents = [SingleAgent(self.env, i) for i in range(self.num_agents)]
        self.replay_buffer = Memory(max_size = memory_maxlen)
        # Exploration noise
        #self.noise = OUNoise(self.env.action_space)
        
    def get_actions(self, states):
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.get_actor_output(state)
            action = np.clip(action, -1, 1)
            action_one_hot = np.zeros(action.shape)
            action_one_hot[action.argmax()] = 1
            actions.append(action_one_hot)
            actions.append(action)
        actions = np.array(actions)

        return actions

    def update(self, batch_size, timestep):
        experiences = self.replay_buffer.sample(batch_size)
        states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = experiences

        total_state_batch = [np.concatenate(states) for states in states_batch] # Concatenate observations of all agents
        action_batch = [np.concatenate(actions) for actions in actions_batch] # Concatenate action vectors of all agents
        next_state_batch = [np.concatenate(next_states) for next_states in next_states_batch]

        total_state_batch = torch.FloatTensor(total_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        
        for agent in self.agents:
            
            """
            Obtain actions of all agents using current policies.
            
            state_batch = [ [[n1] [n2] [n3] [n4] ... [nn]], [[n1] [n2] [n3] [n4] .. [nn]], .... ], states in state_batch = [[n1] [n2] [n3] [n4] ... [nn]] 
            
            1. for each state in state_batch:
                  Obtain action output (tensor) of each agent for all agents and store in tuple -> ( Tensor([a1]), Tensor([a2]), ..., Tensor([an]) ), where ai is action of agent i
                  concatenate tuple so that we have all actions in a single tensor -> Tensor([a1, a2, ... , an])
                  Store tensor to new_action_batch
           
            Output[1]: new_action_batch = [ Tensor([a1, a2, a3, a4, ..., an]_{state_1}, Tensor([a1, a2, a3, a4, ..., an]_{state_2}, ..., Tensor([a1, a2, a3, a4, ..., an]_{state_n}) ]

            2. Stack all tensors in new_action_batch

            Output[2]: new_action_batch = Tensor([[a1, a2, a3, a4, ..., an]_{state_1}, [a1, a2, a3, a4, ..., an]_{state_2}, ..., [a1, a2, a3, a4, ..., an]_{state_n}])      
            """
            current_action_batch = []
            for states in states_batch: 
                actions_tuple = ()
                for _agent in self.agents:
                    indiv_state = states[_agent.agent_id]
                    indiv_state = torch.FloatTensor(indiv_state)
                    action = _agent.actor.forward(indiv_state)
                    actions_tuple += (action,)

                current_action_batch.append(torch.cat(actions_tuple))
        
            current_action_batch = torch.stack(current_action_batch)
            
            """
            Compute policy loss for agent i
            """
            policy_loss = -(agent.critic.forward(total_state_batch, current_action_batch).mean())
            

            """
            Obtain current Q value and new_Q value using target networks according to bellman equation
            """
            next_action_batch = []
            for next_states in next_states_batch: 
                next_actions_tuple = ()
                for _agent in self.agents:
                    next_indiv_state = states[_agent.agent_id]
                    next_indiv_state = torch.FloatTensor(next_indiv_state)
                    next_action = _agent.actor_target.forward(next_indiv_state)
                    next_actions_tuple += (next_action,)

                next_action_batch.append(torch.cat(next_actions_tuple))

            current_Q = agent.critic.forward(total_state_batch, current_action_batch.detach())
            next_action_batch = torch.stack(next_action_batch)
            next_Q = agent.critic_target.forward(next_state_batch, next_action_batch.detach())
            indiv_reward_batch = torch.FloatTensor([rewards.flatten()[agent.agent_id] for rewards in rewards_batch])
            indiv_reward_batch = torch.unsqueeze(indiv_reward_batch, dim=1)
            new_Q = indiv_reward_batch + agent.gamma * next_Q
            
            """
            Compute value loss with MSE of current_Q and new_Q
            """
            value_loss = agent.critic_criterion(current_Q, new_Q)
            
            """
            Update parameters
            """
            agent.actor_optimizer.zero_grad()
            policy_loss.backward() 
            agent.actor_optimizer.step()

            agent.critic_optimizer.zero_grad()
            value_loss.backward()
            agent.critic_optimizer.step()

            """
            Soft parameter update
            """ 
            # Delayed soft parameter update
            if timestep % 50 == 0:     
                for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(param.data)

                for target_param, param in zip(agent.critic_target.parameters(),agent.critic.parameters()):
                    target_param.data.copy_(param.data * agent.tau + target_param.data * (1.0 - agent.tau))
            
            # Soft parameter update 
            # for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
            #     target_param.data.copy_(param.data * agent.tau + target_param.data * (1.0 - agent.tau))
    
            # for target_param, param in zip(agent.critic_target.parameters(),agent.critic.parameters()):
            #     target_param.data.copy_(param.data * agent.tau + target_param.data * (1.0 - agent.tau))

    def train(self, max_episodes, max_steps, batch_size):   
        rewards = []
        timestep = 0

        for episode in range(max_episodes):
            states = self.env.reset()
            episode_reward = 0
            
            for steps in range(max_steps):
                actions = self.get_actions(states)
                new_states, rewards, dones, _ = self.env.step(actions)
                
                if all(dones) or steps == max_steps - 1:
                    dones = [1 for _ in range(self.env.n)]    
                else:
                    dones = [0 for _ in range(self.env.n)]

                self.replay_buffer.push(states, actions, rewards, new_states, dones)

                if len(self.replay_buffer) > batch_size:
                    self.update(batch_size, timestep)
                
                states = new_states
                episode_reward += sum(rewards)
                timestep += 1

                if all(dones) or steps == max_steps - 1:
                    rewards.append(episode_reward)
                    sys.stdout.write("episode: {}  |  reward: {}  \n".format(episode, np.round(episode_reward, decimals=4)))
                    break
                
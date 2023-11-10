import gymnasium
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import os

def choose_action(eps, agent, env, state):
    if random.random() < eps:
        # choose random action with eps%
        action = env.action_space.sample()
    else:
        # choose best action with (1-eps)%
        q_values = agent(torch.tensor(np.array(state), dtype=torch.float))
        action = torch.argmax(q_values).item()
        
    return action

def train(
        agent: nn.Module, 
        target: nn.Module, 
        env: gymnasium.Env, 
        number_episodes: int, 
        eps: float, 
        eps_decay: float,
        eps_min: float,
        epsisodes_before_target_update: int,
        ):
    """Trains a double DQN agent in a speicified environment using an epsilon greedy exploration method

    Args:
        agent (nn.Module): The agent to train
        target (nn.Module): A copy of the agent to train that acts as the target based on double DQN theory
        env (gymnasium.Env): The environment to train the agent in 
        number_episodes (int): The number of episodes to train the agent for (1 episode is 1 run through of the env)
        eps (float): The epsilon value for the epsilon greedy exploration method (higher means more exploration)
        eps_decay (float): The amount that eps decreases each step
        eps_min (float): The smallest value eps can reach via decay
        epsisodes_before_target_update (int): How often the double DQN target model is set back to the learning model 
    """
    
    writer = SummaryWriter()
    
    writer.add_text("config/model", agent.model_name)
    writer.add_text("config/model_details", agent.model_details)
    writer.add_text("config/optimizer", agent.optimizer.__class__.__name__)
    writer.add_text("config/environment", env.spec.id)
    
    global_step = 0
    
    print("Beginning Training")
    
    for episode in range(number_episodes):
        
        state, _ = env.reset()
        terminal = False
        truncated = False
        
        episode_reward = 0
        
        while terminal is not True and truncated is not True:
            
            action = choose_action(eps, agent, env, state)
                
            new_state, reward, terminal, truncated, _ = env.step(action)
            
            episode_reward += reward

            # if this is the final step in the episode - agent needs to know no future rewards coming for bellman eq
            done = terminal or truncated
            
            agent.experiance_buffer.append([state, action, reward, done, new_state])
            
            state = new_state
            
            # if experianced enough for a full batch of training
            if agent.check_buffer_satisfied():
                loss = agent.optimize_model(target)
                
                avg_loss = torch.mean(loss).item()
            
                writer.add_scalar("Loss/global_step", avg_loss, global_step)
                
                
            global_step += 1
            
            eps = max(eps*eps_decay, eps_min)
        
        writer.add_scalar("Score/episode", episode_reward, episode)
        writer.add_scalar("Eps/episode", eps, episode)
        
        if episode % epsisodes_before_target_update == 0:
                target.hard_update_model(agent.state_dict())
        
    writer.flush()

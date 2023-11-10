from runner import train
import gymnasium

from agents.DQN import DQN
from agents.DuelingDQN import DuelingDQN

env = gymnasium.make("CartPole-v1")

agent = DQN(env.observation_space.shape[0], env.action_space.n)
target = DQN(env.observation_space.shape[0], env.action_space.n)

train(agent, target, env, number_episodes=20000, eps=1, eps_decay=0.99999, eps_min=0.05, epsisodes_before_target_update=1000) 

from runner import choose_action
from agents.DQN import DQN

import gymnasium



def test_random_action_selection():
    env = gymnasium.make("CartPole-v1")
    
    agent = DQN(env.observation_space.shape[0], env.action_space.n)
    
    state, _ = env.reset()
    
    for i in range(1000):
    
        action = choose_action(1, agent, env, state)
        
        env_action_max = env.action_space.n
        
        assert(0 <= action < env_action_max)

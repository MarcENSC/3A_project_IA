import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import datetime
from env_wrappers.wrapper import env 
from pathlib import Path
from agent.dqn_agent  import DQN_agent

from logs.logger import MetricLogger



use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)




mario = DQN_agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)
import numpy as np
episodes = 10000
for e in range(episodes):

    state,info = env.reset()
    #state = np.array(state)  # Convert LazyFrames to NumPy array
    
    #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
    #print(state.shape)
    ## Play the game!
    while True:

        # Run agent on the state
        
        action = mario.act(state)
        

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            env.reset()
            break 
    
    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
env.close()
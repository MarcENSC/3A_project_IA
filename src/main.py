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
warm_start = True
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)




mario = DQN_agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

if warm_start:
    checkpoint_path="./checkpoints/2025-01-28T10-22-55/mario_net_34.chkpt"
    checkpoint = torch.load(checkpoint_path)
    mario.net.online.load_state_dict(checkpoint['model']['online'])
    mario.net.target.load_state_dict(checkpoint['model']['target'])
    mario.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mario.exploration_rate = 0.25
    mario.curr_step = checkpoint['curr_step']
   
   



logger = MetricLogger(save_dir)

episodes = 20000
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

        if info["flag_get"]:
            reward+=1000
        if done:
            reward+=-500
        
       
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
    mario.save()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
    
env.close()
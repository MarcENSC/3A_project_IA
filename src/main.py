################################################################################################################################################################
# Script to train the agent on the environment
# you can choose "SARSA" or "DQN" as the model
# you can choose to warm start the training by setting warm_start=True and providing a checkpoint path
# you can choose the number of episodes to train the model, default to 20 000 



################################################################################################################################################################



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
from agent.sarsa_agent import SARSA_agent

from logs.logger import Logger



def launch_training(model="DQN",warm_start=False,episodes=20000,checkpoint=None): 
    use_cuda = torch.cuda.is_available()
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    print(f"Using CUDA: {use_cuda}")
    if model=="DQN":
        mario = DQN_agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    elif model=="SARSA":
        mario = SARSA_agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    if warm_start and checkpoint:
        
        mario.net.online.load_state_dict(checkpoint['model']['online'])
        mario.net.target.load_state_dict(checkpoint['model']['target'])
        mario.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        mario.exploration_rate = 0.25
        mario.curr_step = checkpoint['curr_step']

    logger = Logger(save_dir)
    for e in range(episodes):
        state,info = env.reset()
        while True:

            action = mario.act(state)
            next_state, reward, done, trunc, info = env.step(action)

            if done and info["flag_get"]==False:
                reward += -100

            if info["flag_get"]:
                reward+=1000

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
        mario.save()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    mario.save()
    env.close()



launch_training("SARSA",warm_start=False,episodes=20000,checkpoint=None)
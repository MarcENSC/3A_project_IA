################################################################################################################################################################
# Script to test the agent on the environment based on the trained model with .chkpt file




################################################################################################################################################################


import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from env_wrappers.wrapper import env 
import torch
import time
from agent.dqn_agent import DQN_agent
from pathlib import Path
import datetime
import numpy as np


save_dir = Path("models_tests") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


model = DQN_agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
model.load("./dqn_model_weights.chkpt")  
model.eval() 
done=False

state, info = env.reset()

print(env.action_space)
print(env.action_space.n) 


total_reward=0

while not done:
    print(f"state: {state}")

    print("Info dictionary:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    env.render()
    time.sleep(0.02)

   
    state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(model.device)

    
    action_probs = model.net(state_tensor, model="online") 
    action = torch.argmax(action_probs).item()  

    
    next_state, reward, done,truncated, info = env.step(action)
    total_reward+=reward
    


    
    state = next_state
print(f"reward: {total_reward}")

env.close()

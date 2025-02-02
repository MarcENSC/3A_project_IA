################################################################################################################################################################
# Create the environment    
# Apply different wrappers to the environment
# 1. SkipFrame: Skip 4 frames to speed up the training
# 2. GrayScaleObservation: Convert the observation to grayscale 
# 3. ResizeObservation: Resize the observation to 84x84
# 4. FrameStack: Stack 4 frames together

# The final observation shape will be (4, 84, 84)

# if you wish to see the environment, set render_mode="human" instead of "rgb_array"
# for training set to "rgb_array"
################################################################################################################################################################






import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import warnings
import gym 
import gym_super_mario_bros
#warnings.filterwarnings("ignore", category=DeprecationWarning, message="distutils Version classes are deprecated. Use packaging.version instead")
#warnings.filterwarnings("ignore", category=DeprecationWarning)


from nes_py.wrappers import JoypadSpace
from gym.wrappers import  FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.env_wrappers.skip_frame_wrapper import SkipFrame, GrayScaleObservation, ResizeObservation


# if you wish to see the environment, set render_mode="human" instead of "rgb_array"
# for training set to "rgb_array"
env = gym.make("SuperMarioBros-1-1-v3", apply_api_compatibility=True,render_mode="rgb_array")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
print(env.observation_space.shape)

env = GrayScaleObservation(env)
print(env.observation_space.shape)

env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)


print(f"Frame Stack{env.observation_space.shape}")




print(f"Final wrapper output: {env.observation_space.shape}")
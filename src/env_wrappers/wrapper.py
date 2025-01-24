
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import warnings
import gym 
import gym_super_mario_bros
warnings.filterwarnings("ignore", category=DeprecationWarning, message="distutils Version classes are deprecated. Use packaging.version instead")
warnings.filterwarnings("ignore", category=DeprecationWarning)


from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation, record_video,capped_cubic_video_schedule
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.env_wrappers.skip_frame_wrapper import SkipFrame, StackedObservation
env = gym.make("SuperMarioBros-1-1-v3", apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
print(env.observation_space.shape)

env = GrayScaleObservation(env)
print(env.observation_space.shape)

env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
env = StackedObservation(env)
env= record_video(env, video_folder="videos", name_prefix="test-video", episode_trigger=capped_cubic_video_schedule)

print(f"Frame Stack{env.observation_space.shape}")



#env = RecordVideo(env=env, video_folder="videos", name_prefix="test-video", episode_trigger=capped_cubic_video_schedule)

print(f"okok{env.observation_space.shape}")
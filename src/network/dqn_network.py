import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import numpy as np
import torch


class DQN_network(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        input = input.float()
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
    
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
       
        self.net.online.load_state_dict(checkpoint['model']['online'])
        self.net.target.load_state_dict(checkpoint['model']['target'])

        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

       
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['curr_step']

       
        if 'hyperparameters' in checkpoint:
            self.gamma = checkpoint['hyperparameters'].get('gamma', self.gamma)
            self.batch_size = checkpoint['hyperparameters'].get('batch_size', self.batch_size)
            self.burnin = checkpoint['hyperparameters'].get('burnin', self.burnin)
            self.learn_every = checkpoint['hyperparameters'].get('learn_every', self.learn_every)
            self.sync_every = checkpoint['hyperparameters'].get('sync_every', self.sync_every)

        print(f"Loaded checkpoint from {checkpoint_path}")
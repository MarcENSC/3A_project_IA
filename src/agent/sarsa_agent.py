import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from network.dqn_network import DQN_network
import numpy as np
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict

class SARSA_agent:
    def __init__(self, state_dim, action_dim, save_dir,exploration_rate=1,exploration_rate_decay=0.99999975,exploration_rate_min=0.1,gamma=0.99,batch_size = 64,memory_size = 50000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_reward = -float('inf')
        self.episode_reward = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(memory_size, device=torch.device("cpu")))
        self.batch_size = batch_size
        self.gamma = gamma
        self.net = DQN_network(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        


        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving dqn network

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # Calcule la valeur Q pour le next_state avec le modèle "online"
        next_state_Q = self.net(next_state, model="online")

        # Choisir l'action avec la meilleure valeur Q pour next_state
        next_action = torch.argmax(next_state_Q, axis=1)  # Sélectionner l'action avec la plus grande Q-value

        # Obtenir la valeur Q associée à cette action à partir du modèle "target"
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), next_action]

        # Calculer la cible TD
        td_target_value = reward + (1 - done.float()) * self.gamma * next_Q

        return td_target_value.float()

    
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def act(self, state):

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)
        
        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)



        current_reward = reward.sum().item()  # Somme des rewards dans le batch
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.save_best_model()
        torch.cuda.empty_cache()
        return (td_est.mean().item(), loss)
    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
    
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
    
        with torch.no_grad():
            state = torch.tensor(state, device=self.device)
            next_state = torch.tensor(next_state, device=self.device)
            action = torch.tensor([action], device=self.device)
            reward = torch.tensor([reward], device=self.device)
            done = torch.tensor([done], device=self.device)
    
        # Add to memory and explicitly delete temporary tensors
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))
        
        # Accumulate the total reward for the episode
        self.episode_reward += reward.item()  # Use .item() to convert to a scalar
    
        if done:
            # Check if the current episode reward is the best so far
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
                self.save_best_model()
            
            # Reset the episode reward for the next episode
            self.episode_reward = 0
    
        # Explicitly delete tensors to help garbage collection
        del state, next_state, action, reward, done
    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = self.save_dir / f"sarsa_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            {
                'model': {
                    'online': self.net.online.state_dict(),
                    'target': self.net.target.state_dict()
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
                'exploration_rate': self.exploration_rate,
                'curr_step': self.curr_step,
                'hyperparameters': {
                    'gamma': self.gamma,
                    'batch_size': self.batch_size,
                    'burnin': self.burnin,
                    'learn_every': self.learn_every,
                    'sync_every': self.sync_every
                }
            },
               save_path
    )
    
    def save_best_model(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = self.save_dir / "sarsa_net_best_reward.chkpt"
        torch.save(
            {
                'model': {
                    'online': self.net.online.state_dict(),
                    'target': self.net.target.state_dict()
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
                'exploration_rate': self.exploration_rate,
                'curr_step': self.curr_step,
                'best_reward': self.episode_reward,  # Save the actual episode reward
                'hyperparameters': {
                    'gamma': self.gamma,
                    'batch_size': self.batch_size,
                    'burnin': self.burnin,
                    'learn_every': self.learn_every,
                    'sync_every': self.sync_every
                }
            },
            save_path
        )
        print(f"New best reward achieved: {self.episode_reward}. Model saved to {save_path}")
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # Load the model weights
        self.net.online.load_state_dict(checkpoint['model']['online'])
        self.net.target.load_state_dict(checkpoint['model']['target'])

        # Load the optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore exploration rate and current step
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['curr_step']

        # Restore other hyperparameters if needed
        if 'hyperparameters' in checkpoint:
            self.gamma = checkpoint['hyperparameters'].get('gamma', self.gamma)
            self.batch_size = checkpoint['hyperparameters'].get('batch_size', self.batch_size)
            self.burnin = checkpoint['hyperparameters'].get('burnin', self.burnin)
            self.learn_every = checkpoint['hyperparameters'].get('learn_every', self.learn_every)
            self.sync_every = checkpoint['hyperparameters'].get('sync_every', self.sync_every)

        print(f"Loaded checkpoint from {checkpoint_path}")
    def eval(self):
        self.net.online.eval()
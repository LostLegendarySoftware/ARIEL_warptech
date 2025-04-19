import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from core.ariel import ArielAgent, ArielSignals
from core.exceptions import IncentiveSystemError
from core.logging import logger

class PPONetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(PPONetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(state), self.critic(state)

class IncentiveSystem:
    def __init__(self, agent: ArielAgent, state_dim: int, action_dim: int):
        self.agent = agent
        self.ppo_network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ppo_network.parameters(), lr=3e-4)
        self.clip_param = 0.2
        self.ppo_epochs = 10
        self.batch_size = 64

    async def update(self, states: List[np.ndarray], actions: List[int], rewards: List[float], 
                     next_states: List[np.ndarray], dones: List[bool]):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute advantages
        with torch.no_grad():
            values = self.ppo_network.critic(states).squeeze()
            next_values = self.ppo_network.critic(next_states).squeeze()
            advantages = rewards + (1 - dones) * 0.99 * next_values - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_advantages = advantages[start:end]
                batch_values = values[start:end]

                probs, new_values = self.ppo_network(batch_states)
                new_probs = probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
                old_probs = probs.detach().gather(1, batch_actions.unsqueeze(1)).squeeze()

                ratio = new_probs / old_probs
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(new_values.squeeze(), batch_values + batch_advantages)

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        await ArielSignals.incentive_system_updated.emit()

    async def get_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.ppo_network(state)
        action = torch.multinomial(probs, 1).item()
        return action

    async def save_model(self, path: str):
        try:
            torch.save(self.ppo_network.state_dict(), path)
            logger.info(f"Incentive system model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save incentive system model: {str(e)}")
            raise IncentiveSystemError(f"Failed to save incentive system model: {str(e)}")

    async def load_model(self, path: str):
        try:
            self.ppo_network.load_state_dict(torch.load(path))
            logger.info(f"Incentive system model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load incentive system model: {str(e)}")
            raise IncentiveSystemError(f"Failed to load incentive system model: {str(e)}")
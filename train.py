import torch.optim as optim
from model import NeuroMap
from maze import Maze
import numpy as np
import torch.nn as nn
import torch

HIDDEN_SIZE = 128
LR = 0.0005
EPISODES = 3000
MAX_STEPS = 200

model = NeuroMap(hidden_size=HIDDEN_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for episode in range(EPISODES):
    maze = Maze()
    state = (torch.zeros(1, 1, HIDDEN_SIZE),
             torch.zeros(1, 1, HIDDEN_SIZE))
    episode_loss = 0
    total_reward = 0

    for step in range(MAX_STEPS):
        # Get current state
        sensors = maze.get_sensor_readings()
        input_tensor = torch.tensor(sensors).float().view(1, 1, -1)
        
        # Model prediction
        action_probs, new_state, _ = model(input_tensor, state)
        
        # Action selection with exploration
        if np.random.rand() < max(0.01, 0.3*(1 - episode/EPISODES)):
            action = np.random.randint(3)
        else:
            action = torch.argmax(action_probs).item()
        
        # Execute action
        old_pos = maze.robot_pos
        maze.move_robot(action)
        new_pos = maze.robot_pos
        
        # Reward calculation
        moved_forward = (action == 0) and (old_pos != new_pos)
        hit_wall = (action == 0) and (old_pos == new_pos)
        
        if moved_forward:
            target_action = 0
            reward = 3.0
        elif hit_wall:
            target_action = 1 if np.random.rand() < 0.5 else 2
            reward = -2.0
        elif action == 0:  # Tried to move forward but didn't
            target_action = 1 if np.random.rand() < 0.5 else 2
            reward = -1.0
        else:  # Turning
            if sensors[0] < 2:  # Wall is close
                target_action = action
                reward = 0.5
            else:
                target_action = 0  # Encourage forward
                reward = -0.3
        
        total_reward += reward
        
        # Calculate loss
        loss = criterion(action_probs, torch.tensor([target_action]))
        episode_loss += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update state
        state = (new_state[0].detach(), new_state[1].detach())

    if (episode+1) % 100 == 0:
        print(f"Episode {episode+1}/{EPISODES}, Loss: {episode_loss/MAX_STEPS:.4f}, Reward: {total_reward:.1f}")

torch.save(model.state_dict(), "neuromap_model.pth")
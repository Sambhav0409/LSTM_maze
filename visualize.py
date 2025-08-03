import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from maze import Maze
from model import NeuroMap
import torch
import numpy as np

def visualize_exploration(model_path="neuromap_model.pth"):
    # Initialize components
    maze = Maze()
    model = NeuroMap()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Initialize LSTM state
    state = (torch.zeros(1, 1, model.hidden_size),
             torch.zeros(1, 1, model.hidden_size))
    
    def update(frame):
        nonlocal state
        
        # Get sensor data
        sensors = maze.get_sensor_readings()
        input_tensor = torch.tensor(sensors).float().view(1, 1, -1)
        
        # Model prediction with forced exploration
        with torch.no_grad():
            action_probs, new_state, _ = model(input_tensor, state)
            
            if np.random.rand() < 0.3:  # 30% exploration
                if sensors[0] > 2:  # Path clear
                    action = 0 if np.random.rand() < 0.7 else np.random.randint(1, 3)
                else:
                    action = np.random.randint(3)
            else:
                action = torch.argmax(action_probs).item()
        
        # Execute action
        old_pos = maze.robot_pos
        maze.move_robot(action)
        new_pos = maze.robot_pos
        
        print(f"Step {frame}: Action {['Forward','Left','Right'][action]}, "
              f"Pos {old_pos}→{new_pos}, Orientation {maze.robot_orientation}, "
              f"Sensors {sensors}, Probs {action_probs.numpy().squeeze().round(2)}")
        
        state = new_state
        
        # Update visualization
        ax1.clear()
        ax1.imshow(maze.grid, cmap='binary')
        x, y = maze.robot_pos
        ax1.plot(x, y, 'ro', markersize=12)
        
        # Draw orientation
        if maze.robot_orientation == 0:  # Right
            ax1.arrow(x, y, 0.4, 0, head_width=0.3, color='red')
        elif maze.robot_orientation == 1:  # Down
            ax1.arrow(x, y, 0, 0.4, head_width=0.3, color='red')
        elif maze.robot_orientation == 2:  # Left
            ax1.arrow(x, y, -0.4, 0, head_width=0.3, color='red')
        else:  # Up
            ax1.arrow(x, y, 0, -0.4, head_width=0.3, color='red')
        
        ax1.set_title(f'Step {frame}: Position {maze.robot_pos}')
        
        # Action probabilities
        ax2.clear()
        actions = ['Forward', 'Left', 'Right']
        ax2.bar(actions, action_probs.numpy().squeeze())
        ax2.set_ylim(0, 1)
        ax2.set_title('Action Probabilities')
        
        return ax1, ax2
    
    ani = FuncAnimation(fig, update, frames=100, interval=400, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_exploration()
import numpy as np
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self._generate_maze()
        self.robot_pos = (1, 1)
        self.robot_orientation = 0  # 0:right, 1:down, 2:left, 3:up

    def _generate_maze(self):
        """Generate maze walls"""
        self.grid[0, :] = 1  # Top wall
        self.grid[-1, :] = 1  # Bottom wall
        self.grid[:, 0] = 1  # Left wall
        self.grid[:, -1] = 1  # Right wall
        
        # Internal walls
        self.grid[3, 1:8] = 1
        self.grid[6, 2:9] = 1
        self.grid[8, 3:7] = 1

    def get_sensor_readings(self):
        """Returns [front, left, right] distances"""
        x, y = self.robot_pos
        front = left = right = 0
        
        # Front distance
        if self.robot_orientation == 0:  # Right
            front = self._get_distance(x, y, 1, 0)
            right = self._get_distance(x, y, 0, 1)
            left = self._get_distance(x, y, 0, -1)
        elif self.robot_orientation == 1:  # Down
            front = self._get_distance(x, y, 0, 1)
            right = self._get_distance(x, y, -1, 0)
            left = self._get_distance(x, y, 1, 0)
        elif self.robot_orientation == 2:  # Left
            front = self._get_distance(x, y, -1, 0)
            right = self._get_distance(x, y, 0, -1)
            left = self._get_distance(x, y, 0, 1)
        else:  # Up
            front = self._get_distance(x, y, 0, -1)
            right = self._get_distance(x, y, 1, 0)
            left = self._get_distance(x, y, -1, 0)
            
        return [front, left, right]

    def _get_distance(self, x, y, dx, dy):
        """Calculate distance to wall in direction (dx, dy)"""
        distance = 0
        while True:
            x += dx
            y += dy
            if not (0 <= x < self.width and 0 <= y < self.height) or self.grid[y, x] == 1:
                break
            distance += 1
        return distance

    def move_robot(self, action):
        """Actions: 0=forward, 1=left, 2=right"""
        x, y = self.robot_pos
        
        if action == 0:  # Move forward
            if self.robot_orientation == 0:  # Right
                new_x, new_y = x + 1, y
            elif self.robot_orientation == 1:  # Down
                new_x, new_y = x, y + 1
            elif self.robot_orientation == 2:  # Left
                new_x, new_y = x - 1, y
            else:  # Up
                new_x, new_y = x, y - 1
                
            # Only move if not hitting wall
            if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_y, new_x] == 0:
                self.robot_pos = (new_x, new_y)
                
        elif action == 1:  # Turn left
            self.robot_orientation = (self.robot_orientation - 1) % 4
            
        elif action == 2:  # Turn right
            self.robot_orientation = (self.robot_orientation + 1) % 4

    def visualize(self):
        """Show maze with robot"""
        plt.imshow(self.grid, cmap='binary')
        plt.plot(self.robot_pos[0], self.robot_pos[1], 'ro', markersize=10)
        
        # Show orientation
        x, y = self.robot_pos
        if self.robot_orientation == 0:  # Right
            plt.arrow(x, y, 0.4, 0, head_width=0.2, color='red')
        elif self.robot_orientation == 1:  # Down
            plt.arrow(x, y, 0, 0.4, head_width=0.2, color='red')
        elif self.robot_orientation == 2:  # Left
            plt.arrow(x, y, -0.4, 0, head_width=0.2, color='red')
        else:  # Up
            plt.arrow(x, y, 0, -0.4, head_width=0.2, color='red')
            
        plt.title('Maze Environment')
        plt.show()
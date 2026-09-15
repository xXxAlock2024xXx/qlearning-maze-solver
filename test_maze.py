import numpy as np
from maze import Maze

# 0 = open, 1 = wall
grid = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 0]
])

start = (0, 0)
goal = (2, 0)

maze = Maze(grid, start, goal)

# Test 1: valid move (right, from start)
print(maze.step(3, start))        # expect (0, 1)

# Test 2: move into a wall (down, from (0,0) into (1,0) which is a wall)
print(maze.step(1, start))        # expect (0, 0) -- stays put, wall blocks it

# Test 3: move out of bounds (up, from (0,0))
print(maze.step(0, start))        # expect (0, 0) -- stays put, out of bounds

# Test 4: is_goal check
print(maze.is_goal((2, 0)))       # expect True
print(maze.is_goal((0, 0)))       # expect False
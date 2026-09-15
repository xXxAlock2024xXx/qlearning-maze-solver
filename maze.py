import numpy as np


class Maze:
    def __init__(self, grid , start , goal):
        """
        Args:
            grid: 2D , 0 = open, 1 = wall
            start: (row, column) tuple
            goal: (row, column) tuple
        """
        self.grid = grid
        self.start = start
        self.goal = goal

    def is_valid(self, position):
        row, col = position
        if row < 0 or row >= self.grid.shape[0]:
            return False
        if col < 0 or col >= self.grid.shape[1]:
            return False
        if self.grid[row,col] == 1:
            return False
        return True


    def step(self, action, position):
         row , col = position
         if action == 0:
             new_pos = (row - 1, col)
         elif action == 1:
             new_pos = (row + 1 , col)
         elif action == 2:
             new_pos = (row , col - 1)
         elif action == 3:
             new_pos = (row, col + 1)
         else:
            return position

         if self.is_valid(new_pos):
            return new_pos
         else:
             return position




    def is_goal(self, position):
        return position == self.goal
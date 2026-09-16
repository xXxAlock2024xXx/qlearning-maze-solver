import time
import numpy as np
import matplotlib.pyplot as plt
from maze import Maze
from agent import QLearningAgent
from train import train

def plot_learning_curve(steps_per_episode):
    plt.figure(figsize=(8, 5))
    plt.plot(steps_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Steps to reach goal")
    plt.title("Q-Learning Convergence on Maze")
    plt.savefig("learning_curve.png")
    plt.show()

def get_greedy_path(maze, agent, max_steps=50):
    """Run the agent with epsilon=0 (pure exploitation) to get its final learned path."""
    state = maze.start
    path = [state]
    for _ in range(max_steps):
        if maze.is_goal(state):
            break
        row, col = state
        action = np.argmax(agent.q_table[row, col])  # no exploration, best action only
        state = maze.step(action, state)
        path.append(state)
    return path

def plot_episode_times(times_per_episode):
    plt.figure(figsize=(8, 5))
    plt.plot(times_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Time (seconds)")
    plt.title("Per-Episode Training Time")
    plt.savefig("episode_times.png")
    plt.show()

def plot_solved_maze(maze, path):
    grid = maze.grid.copy().astype(float)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="binary")

    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax.plot(path_cols, path_rows, color="red", linewidth=2, marker="o", markersize=6)

    ax.plot(maze.start[1], maze.start[0], "g*", markersize=20, label="Start")
    ax.plot(maze.goal[1], maze.goal[0], "b*", markersize=20, label="Goal")

    ax.legend()
    ax.set_title("Agent's Learned Path")
    plt.savefig("solved_maze.png")
    plt.show()

if __name__ == "__main__":
    grid = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0]
    ])
    start = (0, 0)
    goal = (2, 0)

    maze = Maze(grid, start, goal)
    agent = QLearningAgent(maze_shape=grid.shape)

    start_time = time.time()
    steps_history, times_history = train(maze, agent, n_episodes=500)
    print(f"Training took {time.time() - start_time:.4f} seconds")

    plot_learning_curve(steps_history)
    plot_episode_times(times_history)

    path = get_greedy_path(maze, agent)
    print(f"Learned path: {path}")
    plot_solved_maze(maze, path)
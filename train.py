import numpy as np
from maze import Maze
from agent import QLearningAgent

def get_reward(maze, position):
    """
    Return the reward for landing on `position`.
    - Goal: +100
    - Otherwise: -1 (small cost per step, encourages shorter paths)
    """
    if maze.is_goal(position):
        return 100
    else:
        return -1

def train(maze, agent, n_episodes=500, max_steps=100):
    """
    Run n_episodes of training.
    Returns a list of step-counts per episode (for plotting later).
    """
    steps_per_episode = []

    for episode in range(n_episodes):
        state = maze.start
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            new_state = maze.step(action, state)
            reward = get_reward(maze, new_state)
            done = maze.is_goal(new_state)

            agent.update(state, action, reward, new_state, done)

            state = new_state
            steps += 1

        agent.decay_epsilon()
        steps_per_episode.append(steps)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{n_episodes}, steps: {steps}, epsilon: {agent.epsilon:.3f}")

    return steps_per_episode


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

    steps_history = train(maze, agent, n_episodes=500)

    print("Training complete.")
    print(f"First episode steps: {steps_history[0]}")
    print(f"Last episode steps: {steps_history[-1]}")
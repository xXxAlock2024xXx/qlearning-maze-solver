import numpy as np

class QLearningAgent:
    def __init__(self, maze_shape, n_actions=4, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        maze_shape: (n_rows, n_cols) — used to size the Q-table
        n_actions: number of possible actions (4: up/down/left/right)
        alpha: learning rate
        gamma: discount factor
        epsilon: initial exploration rate
        epsilon_min: floor for epsilon (never stop exploring completely)
        epsilon_decay: multiply epsilon by this after each episode
        """
        self.n_actions = n_actions
        self.q_table = np.zeros(maze_shape + (self.n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        """
        state: (row, col) tuple
        Return an action (0-3), using epsilon-greedy:
        - with probability self.epsilon, return a random action
        - otherwise, return the action with the highest Q-value for this state
        """
        row, col = state
        if np.random.random() < self.epsilon:
            return np.random.randint(0,self.n_actions)
        else:
            return np.argmax(self.q_table[row,col])

    def update(self, state, action, reward, new_state, done):
        """
        state: (row, col) before the action
        action: action taken (0-3)
        reward: reward received
        new_state: (row, col) after the action
        done: True if new_state is the goal (terminal state)
        """
        row, col = state
        new_row, new_col = new_state

        current_q = self.q_table[row,col,action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[new_row,new_col])

        self.q_table[row,col,action] = current_q + self.alpha * (target - current_q)


    def decay_epsilon(self):
        """Reduce epsilon after each episode, but never below epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
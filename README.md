# CDS524-Assignment-1-WANG-Jiawei
This Python program demonstrates the implementation of a Q-learning algorithm to navigate a maze using a graphical user interface (GUI) built with the Tkinter library. Q-learning is a model-free reinforcement learning algorithm that seeks to find the best action to take given the current state. The primary components of the program include the Maze environment and the Agent class.
![5b836d02-a257-45eb-8b3b-54858541126b](https://github.com/user-attachments/assets/d8f9f6b7-cc81-4b5d-abcd-e5aa2078531e)


The Maze class inherits from tk.Tk and is responsible for creating and displaying the maze. The maze is a 6x6 grid where each cell is 40 pixels by 40 pixels. The __init__ function sets up the window, draws the grid, and places various elements like traps, rewards, and the agent's starting position. The traps are represented by black rectangles, rewards by yellow rectangles, and the exit by a green rectangle. The agent is represented by a red oval.


The Agent class models the learning agent that navigates the maze. It initializes with parameters such as the learning rate (alpha) and the discount factor (gamma). The agent's state space consists of all possible positions in the maze, and its action space includes moving up, down, left, or right. The agent maintains a Q-table, which is a DataFrame where rows correspond to states and columns to actions, initialized with zeros.


The agent uses an epsilon-greedy policy to balance exploration and exploitation when choosing actions. With probability epsilon, the agent explores by selecting a random valid action, and with probability 1-epsilon, it exploits by choosing the action with the highest Q-value. The Q-values are updated using the Bellman equation.


The learn method trains the agent over a specified number of episodes. In each episode, the agent starts from the initial state and moves through the maze, updating its Q-values based on the rewards received until it reaches the exit. The test_agent method evaluates whether the agent can exit the maze within 36 steps, and the play method demonstrates the agent navigating the maze using the learned policy.


The if __name__ == '__main__': block initializes the maze environment and the agent, then trains the agent and allows it to navigate the maze. The agent's performance is improved through learning, and it demonstrates its ability to reach the exit efficiently.


This program showcases how reinforcement learning can be applied to solve navigation problems and highlights the interaction between the learning algorithm and the environment in a visual and interactive manner.

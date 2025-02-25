import random
import time
import tkinter as tk
import pandas as pd

class Maze(tk.Tk):
    '''Environment class (GUI), mainly used to draw the maze and the ball'''
    UNIT = 40  # pixels
    MAZE_R = 6  # grid row
    MAZE_C = 6  # grid column

    def __init__(self):
        super().__init__()
        self.title('Labyrinth')
        h = self.MAZE_R * self.UNIT
        w = self.MAZE_C * self.UNIT
        self.geometry('{0}x{1}'.format(h, w))  # window size
        self.canvas = tk.Canvas(self, bg='white', height=h, width=w)
        # Draw grid
        for c in range(1, self.MAZE_C):
            self.canvas.create_line(c * self.UNIT, 0, c * self.UNIT, h)
        for r in range(1, self.MAZE_R):
            self.canvas.create_line(0, r * self.UNIT, w, r * self.UNIT)
        # Draw entrance
        self._draw_rect(0, 0, 'blue')
        # Draw traps
        self._draw_rect(1, 0, 'black')
        self._draw_rect(1, 1, 'black')
        self._draw_rect(1, 2, 'black')
        self._draw_rect(1, 3, 'black')
        self._draw_rect(1, 4, 'black')
        self._draw_rect(3, 2, 'black')
        self._draw_rect(3, 3, 'black')
        self._draw_rect(3, 4, 'black')
        self._draw_rect(3, 5, 'black')
        self._draw_rect(4, 1, 'black')
        # Draw reward
        self._draw_rect(4, 4, 'yellow')
        # Draw exit
        self._draw_rect(5, 5, 'green')
        # Draw player
        self.rect = self._draw_oval(0, 0, 'red')
        self.canvas.pack()

    def _draw_rect(self, x, y, color):
        '''Draw rectangle, x,y represent column and row indices'''
        padding = 5
        coor = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,
                self.UNIT * (y + 1) - padding]
        return self.canvas.create_rectangle(*coor, fill=color)

    def _draw_oval(self, x, y, color):
        '''Draw oval, x,y represent column and row indices'''
        padding = 6
        coor = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,
                self.UNIT * (y + 1) - padding]
        return self.canvas.create_oval(*coor, fill=color)

    def move_agent_to(self, state):
        '''Move the player to a new position based on the incoming state'''
        coor_old = self.canvas.coords(self.rect)
        x, y = state % 6, state // 6  # column and row indices
        padding = 5
        coor_new = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,
                    self.UNIT * (y + 1) - padding]
        dx_pixels, dy_pixels = coor_new[0] - coor_old[0], coor_new[1] - coor_old[1]
        self.canvas.move(self.rect, dx_pixels, dy_pixels)
        self.update()

class Agent(object):
    '''Agent class'''
    MAZE_R = 6  # maze rows
    MAZE_C = 6  # maze columns

    def __init__(self, alpha=0.1, gamma=0.9):
        '''Initialization'''
        self.states = range(self.MAZE_R * self.MAZE_C)  # states
        self.actions = list('udlr')  # actions: up, down, left, right
        self.rewards = [0, -10, 0, 0, 0, 0,
                        0, -10, 0, 0, -10, 0,
                        0, -10, 0, -10, 0, 0,
                        0, -10, 0, -10, 0, 0,
                        0, -10, 0, -10, 1, 0,
                        0, 0, 0, -10, 0, 50]  # rewards
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.q_table = pd.DataFrame(data=[[0 for _ in self.actions] for _ in self.states],
                                    index=self.states,
                                    columns=self.actions)

    def choose_action(self, state, epsilon=0.8):
        '''Choose action based on current state using epsilon-greedy policy'''
        if random.uniform(0, 1) > epsilon:
            action = random.choice(self.get_valid_actions(state))
        else:
            s = self.q_table.loc[state].filter(items=self.get_valid_actions(state))
            action = random.choice(s[s == s.max()].index)
        return action

    def get_q_values(self, state):
        '''Get Q-values for given state'''
        q_values = self.q_table.loc[state, self.get_valid_actions(state)]
        return q_values

    def update_q_value(self, state, action, next_state_reward, next_state_q_values):
        '''Update Q-value using Bellman equation'''
        self.q_table.loc[state, action] += self.alpha * (
                next_state_reward + self.gamma * next_state_q_values.max() - self.q_table.loc[state, action])

    def get_valid_actions(self, state):
        '''Get all valid actions in current state'''
        valid_actions = set(self.actions)
        if state // self.MAZE_C == 0:
            valid_actions -= {'u'}
        elif state // self.MAZE_C == self.MAZE_R - 1:
            valid_actions -= {'d'}

        if state % self.MAZE_C == 0:
            valid_actions -= {'l'}
        elif state % self.MAZE_C == self.MAZE_C - 1:
            valid_actions -= {'r'}

        return list(valid_actions)

    def get_next_state(self, state, action):
        '''Get next state after performing action'''
        if action == 'u' and state // self.MAZE_C != 0:
            next_state = state - self.MAZE_C
        elif action == 'd' and state // self.MAZE_C != self.MAZE_R - 1:
            next_state = state + self.MAZE_C
        elif action == 'l' and state % self.MAZE_C != 0:
            next_state = state - 1
        elif action == 'r' and state % self.MAZE_C != self.MAZE_C - 1:
            next_state = state + 1
        else:
            next_state = state
        return next_state

    def learn(self, env=None, episode=222, epsilon=0.7):
        '''Q-learning algorithm'''
        print('Agent is learning...')
        for i in range(episode):
            current_state = self.states[0]
            env.move_agent_to(current_state)
            while current_state != self.states[-1]:
                current_action = self.choose_action(current_state, epsilon)
                next_state = self.get_next_state(current_state, current_action)
                next_state_reward = self.rewards[next_state]
                next_state_q_values = self.get_q_values(next_state)
                self.update_q_value(current_state, current_action, next_state_reward, next_state_q_values)
                current_state = next_state
                env.move_agent_to(current_state)
            print(i)
        print('\nLearning completed!')

    def test_agent(self):
        '''Test if the agent can exit the maze within 36 steps'''
        count = 0
        current_state = self.states[0]
        while current_state != self.states[-1]:
            current_action = self.choose_action(current_state, 1.0)
            next_state = self.get_next_state(current_state, current_action)
            current_state = next_state
            count += 1

            if count > self.MAZE_R * self.MAZE_C:
                print('No intelligence')
                return False
        print('Intelligent')
        return True

    def play(self, env=None):
        '''Play the game using learned policy'''
        print('Testing if the agent can exit the maze within 36 steps')
        if not self.test_agent():
            print("I need to learn before playing this game.")
            self.learn(env, episode=222, epsilon=0.7)
        print('Agent is playing...')
        current_state = self.states[0]
        env.move_agent_to(current_state)
        while current_state != self.states[-1]:
            current_action = self.choose_action(current_state, 1)
            next_state = self.get_next_state(current_state, current_action)
            current_state = next_state
            env.move_agent_to(current_state)
            time.sleep(0.4)
        print('\nCongratulations, Agent got it!')

if __name__ == '__main__':
    env = Maze()
    agent = Agent()
    agent.learn(env, episode=333, epsilon=0.7)
    agent.play(env)


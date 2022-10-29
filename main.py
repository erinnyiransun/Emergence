import random
from statistics import mean
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class Space:
    def __init__(self, rows, cols, agent_count, color_count, seed=None):
        """
        Create a Space Object
        :param rows: Number of Rows
        :param cols: Number of Columns
        :param agent_count: Total number of agents
        :param color_count: Number of agent types
        :param seed: Pattern generation seed
        """
        random.seed(seed)
        nums = random.choices(range(color_count + 1), weights=((rows * cols) - agent_count, agent_count), k=rows * cols)
        ar = np.reshape(nums, newshape=(rows, cols))
        random.seed(None)
        self.start = ar
        self.rows, self.cols = rows, cols
        self.board = []
        self.open = []
        self.agents = []
        self.unsatisfied = []
        self.radius = 0
        self.cutoff = 1
        self.avg_utility = 0
        self.reset()

    def reset(self):
        """ Return state object to original layout """
        self.board = self.start.copy()
        self.open.clear()
        self.agents.clear()
        self.unsatisfied.clear()
        self.avg_utility = 0
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.start[r][c]
                if val == 0:
                    self.open.append((r, c))
                else:
                    self.agents.append((r, c))

    def move(self, origin):
        """ Moves agent at origin to new coordinate with higher utility """
        to_check = list(self.open)
        color = self.board[origin]
        random.shuffle(to_check)
        for cord in to_check:
            if self.cord_utility(cord, color) > self.cord_utility(origin, color):
                self.move_utility((origin, cord), color, -1)
                self.board[cord] = color
                self.board[origin] = 0
                self.move_utility((origin, cord), color, 1)
                self.open.remove(cord)
                self.open.append(origin)
                self.agents.append(cord)
                self.agents.remove(origin)
                self.unsatisfied.remove(origin)
                return cord
        return -1

    def display(self):
        """ Show board as 2D grid """
        colormap = colors.ListedColormap(["white", "indigo", "pink"])
        plt.imshow(self.board, cmap=colormap, interpolation='nearest')
        plt.show()

    def simulate(self, radius, cutoff, moves, log):
        """ Simulate """
        time = []
        data = []
        stuck_check = 0
        self.radius = radius
        self.cutoff = cutoff
        self.update_unsatisfied("all")
        for a in self.agents:
            self.avg_utility += self.cord_utility(a, self.board[a])
        for t in range(moves):
            if len(self.unsatisfied) == 0:
                print("All agents satisfied")
                break
            cord = self.unsatisfied[random.randint(0, len(self.unsatisfied) - 1)]
            new_cord = self.move(cord)
            if new_cord != -1:
                stuck_check = 0
                self.update_unsatisfied(cord)
                self.update_unsatisfied(new_cord)
            else:
                stuck_check += 1
                if stuck_check == len(self.unsatisfied):
                    print("Stuck")
                    break
            if t % log == 0:
                print(t)
                time.append(t)
                data.append(self.avg_utility/10000)
        plt.plot(time, data)
        plt.show()
        print("Moves:", t+1)
        return self.avg_utility

    def update_unsatisfied(self, center):
        """ Update unsatisfied agent list """
        if center == "all":
            neighbors = self.agents
        else:
            neighbors = self.get_neighbors(center)

        for cord in neighbors:
            if cord in self.agents:
                util = self.cord_utility(cord, self.board[cord])
                if cord in self.unsatisfied:
                    if util >= self.cutoff:
                        self.unsatisfied.remove(cord)
                else:
                    if util < self.cutoff:
                        self.unsatisfied.append(cord)

    def get_neighbors(self, cord):
        """ Return list of neighbors based off radius """
        neighbors = []
        for r in range(cord[0] - self.radius, cord[0] + self.radius + 1):
            for c in range(cord[1] - self.radius, cord[1] + self.radius + 1):
                if r in range(0, self.rows) and c in range(0, self.cols):
                    neighbors.append((r, c))
        return neighbors

    def move_utility(self, cords, color, plus_minus):
        """ Change average utility after a move """
        delta_u = 0
        for cord in cords:
            neighbors = self.get_neighbors(cord)
            for neighbor in neighbors:
                if self.board[neighbor] == color:
                    delta_u += self.cord_utility(neighbor, color) * plus_minus
        self.avg_utility += delta_u

    def cord_utility(self, cord, color):
        """ Calculate the utility of a given coordinate """
        count = 0
        neighbors = self.get_neighbors(cord)
        for neighbor in neighbors:
            if self.board[neighbor] == color:
                count += 1
        if self.board[cord] == color:
            count -= 1
        return count / (len(neighbors) - 1)


if __name__ == "__main__":
    # Create Board

    s = Space(rows=100, cols=100, agent_count=7000, color_count=2, seed=10)
    # s = Space(rows=5, cols=5, agent_count=10, color_count=1, seed=15)

    s.display()
    s.simulate(radius=2, cutoff=0.5, moves=50000, log=100)
    s.display()

    # Simulate
    # utilities = []
    # s.display()
    # for radius in range(1, 4):
    #     print("R:", radius)
    #     u_r = []
    #     for i in range(1000):
    #         print(i)
    #         u = s.simulate(radius=radius, cutoff=0.5, moves=10000, log=10)
    #         u_r.append(u/10000)
    #         s.reset()
    #     n, bins, patches = plt.hist(x=u_r, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85)
    #     plt.show()
    #     utilities.append(mean(u_r))
    #
    # print(utilities)
    # plt.show()

    # n, bins, patches = plt.hist(x=utilities, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85)
    # plt.show()

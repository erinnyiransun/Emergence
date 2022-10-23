import math
from random import random
from random import randint
import numpy as np
import matplotlib.pyplot as plt


class Block:
    def __init__(self, H, N):
        '''
        :param H: Number of Sites in Block
        :param N: Number of Agents in Block
        '''
        self.H = H
        self.N = N


class Agent:
    def __init__(self, alpha, block):
        self.alpha = alpha
        self.block = block

    def utility(self, block1, block2, eta, xi):
        eps = np.finfo(float).eps
        u1 = eta * block1.N + np.log(
                block1.H - block1.N + eps) - xi * block1.N * block1.N - np.log(block1.N + eps)
        u2 = eta * (block2.N + 1) + max(0, np.log(block2.H - block2.N - 1)) - xi * (block2.N + 1) * (
                block2.N + 1) - np.log(block2.N + 1)


        return u1 - u2

    def move(self, g):
        """
        1. write self.utility
        2. eta * H = 4, xi = 0
        3. save the utility of every agent at every point in time
        4. as we change alpha, what happens
        5. density histogram
        """

        # randomly pick a block with vacancy
        while True:
            block = g.blocks[randint(0, g.dimension - 1)][randint(0, g.dimension - 1)]
            if block.N != block.H:
                break

        eps = np.finfo(float).eps

        # evaluate the gain of the agent (call self.utility)
        u_cur = g.eta * self.block.N + np.log(
            self.block.H - self.block.N + eps) - g.xi * self.block.N * self.block.N - np.log(self.block.N + eps)
        u_new = g.eta * (block.N + 1) + max(0, np.log(block.H - block.N - 1)) - g.xi * (block.N + 1) * (
                block.N + 1) - np.log(block.N + 1)
        delta_u = u_new - u_cur

        # delta_u = self.utility(self.block, block, g.eta, g.xi)

        # evaluate the change of others
        u1 = g.eta * block.N + np.log(block.H - block.N + eps) - g.xi * block.N * block.N - np.log(
            block.N + eps)
        U_cur = self.block.N * u_cur + block.N * u1
        u2 = g.eta * (self.block.N - 1) + np.log(self.block.H - self.block.N + 1) - g.xi * (self.block.N - 1) * (
                self.block.N - 1) - np.log(self.block.N - 1)

        U_new = (self.block.N - 1) * u2 + (block.N + 1) * u_new
        delta_U = U_new - U_cur

        # calculate the gain
        gain = delta_u + self.alpha * (delta_U - delta_u)

        # calculate the probability to move
        #P = 1 / (1 + np.exp(-1 * gain / g.T))

        move = gain > 0  # random() < P
        #move = random() < P

        if move:
            self.block.N -= 1
            self.block = block
            block.N += 1

            # print('An agent moves from block ' + str(cur_block.idx) + ' to block ' + str(block.idx))


class SchellingGame:
    def __init__(self, N, dimension, H, alpha, T=1, eta=0.08, xi=0):
        '''
        n: number of blocks, n > 1
        N: number of agents
        H: number of sites in each block
        alpha: altruistic factor of an agent
        T: temperature
        '''

        self.dimension = dimension
        self.n = dimension * dimension
        self.N = N
        self.H = H
        self.alpha = alpha
        self.T = T
        self.eta = eta
        self.xi = xi

        # initialize the blocks
        b = []
        for i in range(0, self.n):
            b.append(Block(H, 0))
        self.blocks = np.array(b).reshape(dimension, dimension)

        agents = []
        for i in range(0, N):
            while True:
                block = self.blocks[randint(0, dimension - 1)][randint(0, dimension - 1)]
                if block.N != block.H:
                    break
            agents.append(Agent(alpha, block))
            block.N += 1
        self.agents = agents


    def simulate(self, time_steps):
        for t in range(time_steps):
            agent = self.agents[randint(0, len(self.agents)-1)]
            agent.move(self)

    def heat_map(self):
        d = []
        for block in self.blocks.flatten():
            d.append(block.N / block.H)
        densities = np.array(d).reshape(self.dimension, self.dimension)

        plt.imshow(densities, cmap='Blues', interpolation='nearest')
        plt.show()


if __name__ == "__main__":
    '''
    N: number of agents
    n: number of blocks, n > 1
    H: number of sites in each block
    '''
    g = SchellingGame(10000, 15, 100, 1)
    g.heat_map()
    g.simulate(1000000)
    g.heat_map()

    # for a in np.arange(0, 1, 0.1):
    #     g = SchellingGame(10000, 15, 100, a)
    #     g.simulate(1000000)
    #     g.heat_map()

    # g = SchellingGame(10, 10, 5)
    # g.display_blocks()
    # g.heat_map()
    # for _ in range(1000000):
    #     g.move()
    # g.display_blocks()
    # g.heat_map()

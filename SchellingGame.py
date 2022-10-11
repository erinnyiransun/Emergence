#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:24:00 2022

@author: erinnsun
"""

from random import randint, random
import numpy as np




class Agent():
    
    def __init__(self, idx, alpha, block):
        
        self.idx = idx 
        self.alpha = alpha 
        self.block = block
        

class Block():
    
    def __init__(self, idx, H, N):
        
        self.idx = idx 
        self.H = H 
        self.N = N 


class SchellingGame():
    
    def __init__(self, N, n, H, alpha = 0, T = 1, eta = 0.08, xi = 0):
        '''
        n: number of blocks, n > 1
        N: number of agents
        H: number of sites in each block
        alpha: altruistic factor of an agent
        T: temperature
        '''
        
        self.n = n
        self.N = N
        self.H = H
        self.alpha = alpha
        self.T = T
        self.eta = eta
        self.xi = xi
        
        
        self.d_block = {}
        self.d_agent = {}
        
        available_blocks = []
        # initialize the blocks
        for j in range(1, n + 1):
            self.d_block[j] = Block(j, H, 0)
            available_blocks.append(j)
            
        # distribute the agents
        for i in range(1, N + 1):
            
            block = available_blocks[randint(0, len(available_blocks) - 1)]
            self.d_agent[i] = Agent(i, alpha, block)
            self.d_block[block].N += 1
            if self.d_block[block].N == self.d_block[block].H:
                available_blocks.remove(block)
        
        self.available_blocks = available_blocks
           
    
    def utility(self, agent_idx):
        pass
        
    def move(self):
        '''
        1. write self.utility
        2. eta * H = 4, xi = 0
        3. save the utility of every agent at every point in time
        4. as we change alpha, what happens
        5. density histogram
        '''
        
        # randomly pick an agent
        agent_idx = randint(1, len(self.d_agent))
        agent = self.d_agent[agent_idx]
        
        # randomly pick a block with vacancy
        block_idx = self.available_blocks[randint(0, len(self.available_blocks)) - 1]
        block = self.d_block[block_idx]
        
        # current block
        cur_block = self.d_block[agent.block]
        
        """Put this eps in utility function"""
        eps = np.finfo(float).eps
        
        # evaluate the gain of the agent (call self.utility)
        u_cur = self.eta * cur_block.N + np.log(cur_block.H - cur_block.N + eps) - self.xi * cur_block.N * cur_block.N - np.log(cur_block.N + eps)
        u_new = self.eta * (block.N + 1) + max(0, np.log(block.H - block.N - 1)) - self.xi * (block.N + 1) * (block.N + 1) - np.log(block.N + 1)
        delta_u = u_new - u_cur
        
        # evaluate the change of others
        u1 = self.eta * (block.N) + np.log(block.H - block.N + eps) - self.xi * (block.N) * (block.N) - np.log(block.N  + eps)
        U_cur = cur_block.N * u_cur + block.N * u1
        u2 = self.eta * (cur_block.N-1) + np.log(cur_block.H - cur_block.N + 1) - self.xi * (cur_block.N - 1) * (cur_block.N - 1) - np.log(cur_block.N - 1)
        U_new = (cur_block.N - 1) * u2 + (block.N + 1) * u_new
        delta_U = U_new - U_cur
        
        # calculate the gain
        gain = delta_u + agent.alpha * (delta_U - delta_u)
        
        # calculate the probability to move
        # P = 1 / (1 + np.exp(-1 * gain / self.T))

        
        move = gain>0#random() < P

        if move:
            agent.block = block_idx
            block.N += 1
            if block.N == block.H:
                self.available_blocks.remove(block_idx)
            
            cur_block.N -= 1
            if cur_block.N < block.H and cur_block.idx not in self.available_blocks:
                self.available_blocks.append(cur_block.idx)
            
            # print('An agent moves from block ' + str(cur_block.idx) + ' to block ' + str(block.idx))
        
          
    def display_blocks(self):
        
        densities = []
        for idx in self.d_block:
            densities.append(self.d_block[idx].N / self.d_block[idx].H)
            print('Block', idx, ' density:', self.d_block[idx].N, self.d_block[idx].N / self.d_block[idx].H)
        
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.hist(densities, bins = 20)
        
        
if __name__ == "__main__":
    
    g = SchellingGame(10000, 200, 100)
    g.display_blocks()
    for  _ in range(1000000):
        g.move()
    g.display_blocks()
    
    
    
    
    
    
    
    
    
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:24:00 2022

@author: erinnsun
"""

from random import randint, random
import numpy as np
import matplotlib.pyplot as plt
import imageio


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
    
    def __init__(self, N, n, H, alpha = 20, T = 1, eta = 0.08, xi = 0):
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
        
        
        self.d_block = None
        self.d_agent = None
        self.d_history = None


        self.available_blocks = None
        
        self.initialize()
           
    
    def initialize(self):
        
        self.d_block = {}
        self.d_agent = {}
        self.d_history = {}
        
        available_blocks = []
        # initialize the blocks
        for j in range(1, self.n + 1):
            self.d_block[j] = Block(j, self.H, 0)
            available_blocks.append(j)
            
        # distribute the agents
        for i in range(1, self.N + 1):
            
            block = available_blocks[randint(0, len(available_blocks) - 1)]
            self.d_agent[i] = Agent(i, self.alpha, block)
            self.d_block[block].N += 1
            if self.d_block[block].N == self.d_block[block].H:
                available_blocks.remove(block)
        
        self.available_blocks = available_blocks
       
        return 
        
        
    def utility(self, agent, cur_block, dest_block):
        '''
        1. write everything in terms of the density of the block
        2. define eta * H as a new parameter, N = H * density push to 0
        3. xi * N * N = xi * H * H * density * density, define xi * H * H as a new parameter push to 0
        4. scale delta_u and delta_U
        5. alpha: [0, 1]
        6. visualize the locations of agents at each timestamp
        7. look at a list of parameters that you can test on
        8. look at the paper and study when phase separation will occur, it depends on the three parameters 
        9. If phase segregation does not occur, can I push my alpha higher to make the phase segregation occur
        '''
        
        """Put this eps in utility function"""
        eps = np.finfo(float).eps
        
        # evaluate the gain of the agent (call self.utility)
        u_cur = self.eta * cur_block.N + np.log(cur_block.H - cur_block.N + eps) - self.xi * cur_block.N * cur_block.N - np.log(cur_block.N + eps)
        u_new = self.eta * (dest_block.N + 1) + max(0, np.log(dest_block.H - dest_block.N - 1)) - self.xi * (dest_block.N + 1) * (dest_block.N + 1) - np.log(dest_block.N + 1)
        delta_u = u_new - u_cur
        
        # evaluate the change of others
        u1 = self.eta * (dest_block.N) + np.log(dest_block.H - dest_block.N + eps) - self.xi * (dest_block.N) * (dest_block.N) - np.log(dest_block.N  + eps)
        U_cur = cur_block.N * u_cur + dest_block.N * u1
        u2 = self.eta * (cur_block.N-1) + np.log(cur_block.H - cur_block.N + 1) - self.xi * (cur_block.N - 1) * (cur_block.N - 1) - np.log(cur_block.N - 1)
        U_new = (cur_block.N - 1) * u2 + (dest_block.N + 1) * u_new
        delta_U = U_new - U_cur
        delta_U /= (cur_block.N + dest_block.N) # normalizing term, should not be added
        
        # calculate the gain
        gain = delta_u + agent.alpha * (delta_U - delta_u)
        
        # calculate the probability to move
        P = 1 / (1 + np.exp(-1 * gain / self.T))
        
        return (gain, P)
    
        
    def move(self, timestamp):
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
        
        gain, P = self.utility(agent, cur_block, block)
        
        
        move = gain > 0 #random() < P

        if move:
            agent.block = block_idx
            block.N += 1
            if block.N == block.H:
                self.available_blocks.remove(block_idx)
            
            cur_block.N -= 1
            if cur_block.N < block.H and cur_block.idx not in self.available_blocks:
                self.available_blocks.append(cur_block.idx)
            
            self.store_density(timestamp)
            # print('An agent moves from block ' + str(cur_block.idx) + ' to block ' + str(block.idx))
        
    
    def store_density(self, timestamp):
        
        densities = {}
        for idx in self.d_block:
            densities[idx] = self.d_block[idx].N / self.d_block[idx].H
        
        self.d_history[timestamp] = densities
            
    
    def alpha_experiment(self, l_alpha = [0.5, 1, 2, 4, 8, 64, 128, 256, 512, 5024]):
        
        
        fig, ax = plt.subplots(5, 2, figsize = (15, 10))
        
        for i in range(len(l_alpha)):
            
            self.alpha = l_alpha[i]
            self.initialize()
            
            T = 50000
            for t in range(T):
                self.move(t)
            
            last_timestamp = list(self.d_history.keys())[-1]
            ax[i//2, i%2].hist(list(self.d_history[last_timestamp].values()), bins = 20)

            
            
            
   
        
    def generate_density_gif(self):
        
        filenames = []
        
        for t in self.d_history:
            
            filename = '/Users/erinnsun/Desktop/schellinggame/%d.png'%(t) 
            filenames.append(filename)
            plt.figure()
            plt.bar(list(self.d_history[t].keys()), list(self.d_history[t].values()))
            plt.savefig(filename)
            plt.close()
        
        with imageio.get_writer('mygif.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            
        
if __name__ == "__main__":
    
    g = SchellingGame(10000, 200, 100)
    '''
    1. Change the three parameters: alpha, eta * H, xi * H * H
    '''
   
    g.alpha_experiment()
    
    
    
    
    
    
    
    
    
    
    
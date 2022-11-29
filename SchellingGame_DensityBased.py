#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:28:06 2022

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
    
    def __init__(self, idx, H, rho):
        
        self.idx = idx 
        self.H = H 
        self.rho = rho


class SchellingGame():
    
    def __init__(self, N, n, H, alpha = 20, T = 1, eH = 4, xHH = 0):
        '''
        n: number of blocks, n > 1
        N: number of agents
        H: number of sites in each block
        alpha: altruistic factor of an agent
        T: temperature
        eH = eta * H
        xHH: xi * H * H
        '''
        
        self.n = n
        self.N = N
        self.H = H
        self.alpha = alpha
        self.T = T
        self.eH = eH
        self.xHH = xHH
        
        
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
            
            self.d_block[block].rho = (self.d_block[block].rho * self.d_block[block].H + 1) / self.d_block[block].H
            if self.d_block[block].rho == 1:
                available_blocks.remove(block)
        
        self.available_blocks = available_blocks
       
        return 
        
        
    def utility(self, agent, cur_block, dest_block):
        '''
        6. visualize the locations of agents at each timestamp
        7. look at a list of parameters that you can test on
        8. look at the paper and study when phase separation will occur, it depends on the three parameters 
        9. If phase segregation does not occur, can I push my alpha higher to make the phase segregation occur
        '''
        
        """Put this eps in utility function"""
        eps = np.finfo(float).eps
        
        # evaluate the gain of the agent (call self.utility)
        u_cur = self.eH * cur_block.rho + np.log(cur_block.H * (1 - cur_block.rho))
        u_cur -= self.xHH * cur_block.rho * cur_block.rho + np.log(cur_block.H * cur_block.rho)
        
        new_rho = (dest_block.rho * dest_block.H + 1) / dest_block.H
        u_new = self.eH * new_rho + np.log(dest_block.H * (1 - new_rho))
        u_new -= self.xHH * new_rho * new_rho + np.log(dest_block.H * new_rho)
        
        delta_u = u_new - u_cur
        
        
        
        # evaluate the change of others
        u1 = self.eH * dest_block.rho + np.log(dest_block.H * (1 - dest_block.rho))
        u1 -= self.xHH * dest_block.rho * dest_block.rho + np.log(dest_block.H * dest_block.rho)
        U_cur = cur_block.rho * cur_block.H * u_cur + dest_block.rho * dest_block.H * u1
        
        new_rho = (cur_block.rho * cur_block.H - 1) / cur_block.H
        u2 = self.eH * new_rho + np.log(cur_block.H * (1 - new_rho))
        u2 -= self.xHH * new_rho * new_rho - np.log(cur_block.H * new_rho)
        
        U_new = (cur_block.rho * cur_block.H - 1) * u2 + (dest_block.rho * dest_block.H + 1) * u_new
        
        delta_U = U_new - U_cur
        # delta_U /= (cur_block.N + dest_block.N) # normalizing term, should not be added
        delta_U /= self.N # scaling
        
        # calculate the gain
        gain = (1 - agent.alpha) * delta_u + agent.alpha * delta_U 
        
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
            block.rho = (block.rho * block.H + 1) / block.H
            if block.rho == 1:
                self.available_blocks.remove(block_idx)
            
            cur_block.rho = (cur_block.rho * cur_block.H - 1) / cur_block.H
            if cur_block.rho < 1 and cur_block.idx not in self.available_blocks:
                self.available_blocks.append(cur_block.idx)
            
            self.store_density(timestamp)
            # print('An agent moves from block ' + str(cur_block.idx) + ' to block ' + str(block.idx))
        
    
    def store_density(self, timestamp):
        
        densities = {}
        for idx in self.d_block:
            densities[idx] = self.d_block[idx].rho
        
        self.d_history[timestamp] = densities
            
    
    def alpha_experiment(self, l_alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        
        
        fig, ax = plt.subplots(5, 2, figsize = (15, 10))
        
        for i in range(len(l_alpha)):
            
            self.alpha = l_alpha[i]
            self.initialize()
            
            T = 100000
            for t in range(T):
                self.move(t)
            
            last_timestamp = list(self.d_history.keys())[-1]
            ax[i//2, i%2].hist(list(self.d_history[last_timestamp].values()), bins = 20)
            ax[i//2, i%2].set_title('alpha = %f'%(self.alpha))

            
            
            
   
        
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
    
    '''
    1. Change the three parameters: alpha, eta * H, xi * H * H
    '''
    g = SchellingGame(N = 10000, n = 200, H = 100, alpha = 0.5, eH = 4.5, xHH = 0)
   
    g.alpha_experiment()
    
    
    
    
    
    
    
    
    
    
    
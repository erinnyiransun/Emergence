#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:41:16 2022

@author: erinnsun
"""

import matplotlib.pyplot as plt
import numpy as np



def utility(rho, eH = 5, xHH = 0, H = 100):
    u_cur = eH * rho + np.log(H * (1 - rho))
    u_cur -= xHH * rho * rho + np.log(H * rho)
    return u_cur


lx = list(range(1, 100))
lx = [0.01 * x for x in lx]

ly = []
for x in lx:
   ly.append(utility(x))

plt.scatter(lx, ly)
plt.show()


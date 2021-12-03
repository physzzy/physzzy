# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 15:23:35 2021

@author: 92012
"""


from mpl_toolkits import mplot3d
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time

begin_time = time.time()
sigma_0 = np.identity(2, dtype=complex);
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex);
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex);
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex);

sigma_1 = sigma_x
sigma_2 = sigma_y
sigma_3 = sigma_z


Gamma_0 = np.kron(sigma_3, sigma_0)
Gamma_1 =-np.kron(sigma_2, sigma_1)
Gamma_2 =-np.kron(sigma_2, sigma_2)
Gamma_3 =-np.kron(sigma_2, sigma_3)
Gamma_4 = np.kron(sigma_1, sigma_0)


k_step=101
realspace_step =100000

kx=np.linspace(-np.pi,np.pi,k_step)
x=np.linspace(-50,50,realspace_step)
y=np.zeros(realspace_step)
Lattice=np.linspace(-10,10,101)
def u(k,r):
    u=0
    for R in Lattice:
        u += np.exp(-(r-R))*np.exp(-1j*k*(r-R))
    
    return u

for l in range(realspace_step):
    norm=0
    print(l)
    for i in range(k_step):
        #norm +=np.exp(1j*kx[i]*x[l])
        for R in Lattice:
          norm +=np.exp(-abs(x[l]-R))*np.exp(1j*x[l]*kx[i])
    
    y[l]=np.linalg.norm(norm)*(2*np.pi/k_step)

fig=plt.figure()
plt.scatter(x,y,color='red',s=0.2)
    
    
        
        
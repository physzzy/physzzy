# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:45:08 2021

@author: 92012
"""

from mpl_toolkits import mplot3d
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time
from math import *
import cmath

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
##parameter


gamma = 0.5
lambda0 = 1
delta = 0.0
##————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————##
def hamiltonian(k):
    h = np.zeros((2, 2))*(1+0j)
    h[0,0] = delta
    h[1,1] = -delta
    h[0,1] = gamma+lambda0*cmath.exp(-1j*k)
    h[1,0] = gamma+lambda0*cmath.exp(1j*k)
    return h


def main():
    Num_k = 100
    k_array = np.linspace(-np.pi, np.pi, Num_k)
    vector_array = []
    for k in k_array:
        vector  = get_occupied_bands_vectors(k, hamiltonian)   
        if k != pi:
            vector_array.append(vector)
        else:
            vector_array.append(vector_array[0])

    # 计算Wilson loop
    W_k = 1
    for i0 in range(Num_k-1):
        F = np.dot(vector_array[i0+1].transpose().conj(), vector_array[i0])
        W_k = np.dot(F, W_k)
    nu = np.log(W_k)/2/pi/1j
    # if np.real(nu) < 0:
    #     nu += 1
    print('p=', nu, '\n')
    

def get_occupied_bands_vectors(x, matrix):  
    matrix0 = matrix(x)
    eigenvalue, eigenvector = np.linalg.eig(matrix0) 
    vector = eigenvector[:, np.argsort(np.real(eigenvalue))[0]] 
    return vector


if __name__ == '__main__':
    main()
    
    

##--------------------------------------------------------------------------------------------------------------------------------------##
##计算开放边界条件下的能谱和电荷密度

cell_size = 1000
bands = 2
def open_boundary_SSH_model(_size = bands * cell_size):
    _open_H = np.zeros((_size,_size), dtype = complex)
    
    for i in range(cell_size-1):
        #intercell
        _open_H[i*bands, i*bands+1] = gamma
        _open_H[i*bands+1, i*bands] = gamma
    for i in range(cell_size-1):
        #intrecell
        _open_H[i*bands+1,i*bands+2] = lambda0
        _open_H[i*bands+2,i*bands+1] = lambda0
    for i in range(cell_size -1):
        _open_H[i*bands, i*bands] = delta
        _open_H[i*bands+1, i*bands+1] = -delta
        
    return _open_H

eigval, eigvec = np.linalg.eigh(open_boundary_SSH_model())
Charge_density = np.zeros(cell_size *bands)
edgemodes = np.zeros(cell_size*bands)

for i in range(len(eigval)):
    if eigval[i] >0:
        Charge_density += np.real(eigvec[:,i]*eigvec[:,i].conjugate())/np.linalg.norm(eigvec[:,i])**2
    
    #if np.abs(eigval[i]) <=0.01
    #    edgemodes.append(np.real(eigvec[:,i]*eigvec[:,i].conjugate())/np.linalg.norm(eigvec[:,i])**2)

Charge_density_unitcell = np.zeros(cell_size)

    
for i in range(cell_size):
    Charge_density_unitcell[i] += Charge_density[bands*i]
    Charge_density_unitcell[i] += Charge_density[bands*i+1]


unitcell_num = np.linspace(1, cell_size, cell_size)
unitcell_num_bands = np.linspace(1, cell_size*bands, cell_size*bands)
   
i = cell_size-1 
edge1= np.real(eigvec[:,i]*eigvec[:,i].conjugate())/np.linalg.norm(eigvec[:,i])**2
i = cell_size
edge2= np.real(eigvec[:,i]*eigvec[:,i].conjugate())/np.linalg.norm(eigvec[:,i])**2
fig = plt.figure()
plt.scatter(unitcell_num,Charge_density_unitcell,color='red',s=0.2)
fig = plt.figure()
plt.scatter(unitcell_num_bands,edge1,color='orange',s=0.2)
fig = plt.figure()
plt.scatter(unitcell_num_bands,edge2,color='indigo',s=0.2)        
    
    





























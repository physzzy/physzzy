# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:58:02 2021

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

Gamma_01 = np.kron(sigma_0, sigma_1)
Gamma_02 = np.kron(sigma_0, sigma_2)
Gamma_10 = np.kron(sigma_1, sigma_0)
Gamma_11 = np.kron(sigma_1, sigma_1)
Gamma_30 = np.kron(sigma_3, sigma_0)
Gamma_31 = np.kron(sigma_3, sigma_1)
Gamma_32 = np.kron(sigma_3, sigma_2)
Gamma_21 = np.kron(sigma_2, sigma_1)
Gamma_22 = np.kron(sigma_2, sigma_2)
Gamma_23 = np.kron(sigma_2, sigma_3)

gamma = 0.2
bands = 4

##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##defination of  the Hamiltonian decribed in Eq.(4.27) in PBC
def H(_kx,_ky):
    _H = np.zeros((bands,bands),dtype = complex)
    _H +=np.cos(_kx)*np.cos(_ky)*Gamma_01
    _H +=np.sin(_kx)*np.sin(_ky)*Gamma_31
    _H +=np.sin(_kx)*np.cos(_ky)*Gamma_02
    _H +=-np.cos(_kx)*np.sin(_ky)*Gamma_32
    _H +=gamma* Gamma_10
    _H +=gamma* Gamma_11
    return _H


kx = np.linspace(-np.pi, np.pi, 50)
ky = np.linspace(-np.pi, np.pi, 50)

Z1 = np.zeros((len(kx), len(ky)))
Z2 = np.zeros((len(kx), len(ky)))
Z3 = np.zeros((len(kx), len(ky)))
Z4 = np.zeros((len(kx), len(ky)))

for i in range(len(kx)):
    for j in range(len(ky)):
        _eigval, _eigvec = np.linalg.eigh(H(kx[i],ky[j]))
        eigval  = np.sort(np.real(_eigval))
        
        Z1[i,j] = eigval[0]
        Z2[i,j] = eigval[1]
        Z3[i,j] = eigval[2]
        Z4[i,j] = eigval[3]

X , Y = np.meshgrid(kx, ky)


fig = plt.figure()
ax3 = plt.axes(projection='3d')

ax3.plot_surface(X,Y,Z1,cmap='rainbow')
ax3.plot_surface(X,Y,Z2,cmap='rainbow')
ax3.plot_surface(X,Y,Z3,cmap='rainbow')
ax3.plot_surface(X,Y,Z4,cmap='rainbow')
ax3.view_init(0, 0)



##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##defination of  the Hamiltonian decribed in Eq.(4.27) in x-open boundary condition
size_x = 100
def open_H(ky, _Nx = size_x, _b = bands):
    _open_H = np.zeros((_b*_Nx,_b*_Nx), dtype = complex)
##----------------------------------------------------------##   
    _open_h = np.zeros((_Nx,_Nx), dtype = complex)
    for i in range(_Nx-1):
        _open_h[i, i+1] = np.cos(ky)*0.5
        _open_h[i+1, i] = np.cos(ky)*0.5        
    _open_H +=np.kron(_open_h, Gamma_01)
##----------------------------------------------------------## 
    _open_h = np.zeros((_Nx,_Nx), dtype = complex)
    for i in range(_Nx-1):
        _open_h[i, i+1] =-np.sin(ky)*0.5*(-1j)
        _open_h[i+1, i] = np.sin(ky)*0.5*(-1j)       
    _open_H +=np.kron(_open_h, Gamma_31)
##----------------------------------------------------------##     
    _open_h = np.zeros((_Nx,_Nx), dtype = complex)
    for i in range(_Nx-1):
        _open_h[i, i+1] =-np.cos(ky)*0.5*(-1j)
        _open_h[i+1, i] = np.cos(ky)*0.5*(-1j)        
    _open_H +=np.kron(_open_h, Gamma_02)
##----------------------------------------------------------##     
    _open_h = np.zeros((_Nx,_Nx), dtype = complex)
    for i in range(_Nx-1):
        _open_h[i, i+1] = -np.sin(ky)*0.5
        _open_h[i+1, i] = -np.sin(ky)*0.5        
    _open_H +=np.kron(_open_h, Gamma_32)
##----------------------------------------------------------##     
    _open_h = np.zeros((_Nx,_Nx), dtype = complex)
    for i in range(_Nx):
        _open_h[i, i] = gamma
        
        
    _open_H +=np.kron(_open_h, Gamma_10)
    _open_H +=np.kron(_open_h, Gamma_11)
    
    return _open_H

ky_size = 50
ky = np.linspace(0,2* np.pi, ky_size)
Eigval = np.zeros(ky_size*size_x*bands, dtype=float)
for i in range(ky_size):
    eigval, eigvec = np.linalg.eigh(open_H(ky[i]))
    Eigval[size_x*bands*i:size_x*bands*(i+1)] = eigval


fig = plt.figure()
plt.scatter(np.kron(ky,np.ones(size_x*bands)),Eigval,color='red',s=0.2)
    
    
    
    

##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##
##计算Wilson loop Fig.15(b)

wilson_step_kx = 500

def Wilson_loop_x(ky,base_kx=0):
    step_wilson_loop = wilson_step_kx
    kx_wilson_loop = np.linspace(base_kx, base_kx+2*np.pi, step_wilson_loop)
    vector_array = np.zeros((step_wilson_loop*bands,bands),dtype=complex)
    
    for i in range(step_wilson_loop):
        if kx_wilson_loop[i] != base_kx+2*np.pi:
            _wilsonloop_eigval, _wilsonloop_eigvec = np.linalg.eigh(H(kx_wilson_loop[i], ky))
            vec_index = np.argsort(np.real(_wilsonloop_eigval))
            for k in range(bands):
                vector_array[i*bands+k,:]=_wilsonloop_eigvec[:,vec_index[k]].transpose()*np.linalg.norm(_wilsonloop_eigvec[:,vec_index[k]])**-1
        else:
            #print('Wilson loop首尾接上了，不存在gauge的问题，连接点在：',i)
            _wilsonloop_eigval, _wilsonloop_eigvec = np.linalg.eigh(H(base_kx, ky))
            vec_index = np.argsort(np.real(_wilsonloop_eigval))
            for k in range(bands):
                vector_array[i*bands+k,:]=_wilsonloop_eigvec[:,vec_index[k]].transpose()*np.linalg.norm(_wilsonloop_eigvec[:,vec_index[k]])**-1
    

    Wilson_loop_occ_band= np.eye(2,2)
    F_occ = np.zeros((2,2),dtype=complex)
                
    for l in range(step_wilson_loop-1):       
            for m in range(int(bands/2)):
                for n in range(int(bands/2)):
                    F_occ[m,n] = np.dot(vector_array[(l+1)*bands+m,:].conjugate(),vector_array[l*bands+n,:].transpose())
            Wilson_loop_occ_band = np.dot(F_occ,Wilson_loop_occ_band)
            Wilson_eigval, Wilson_eigvec =np.linalg.eig(Wilson_loop_occ_band)
    
    return (np.angle(Wilson_eigval)/2/np.pi,Wilson_eigvec)
    

step_ky=200
W_x_band0 = np.zeros(step_ky,dtype=complex)
W_x_band1 = np.zeros(step_ky,dtype=complex)                    
        
ky = np.linspace(0,2*np.pi,step_ky)

for i in range(step_ky):
    W_x_band0[i] = Wilson_loop_x(ky[i])[0][0] %1 
    W_x_band1[i] = Wilson_loop_x(ky[i])[0][1] %1 

fig = plt.figure()
plt.scatter(ky,W_x_band0,color='red',s=0.2)
plt.scatter(ky,W_x_band1,color='red',s=0.2)




##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------##





# def mass_invar_space1():
#     _mass_rep = chop(np.real(mass_rep(rep,-1)))

#     init_printing(perm_cyclic=True,pretty_print=False)
#     p = Permutation(tran_Mat_to_Permu(_mass_rep)).full_cyclic_form
#     p_conv = index_to_conv(p)
#     print('mass的不变子空间', p_conv)
#     print('———————————————————————————————————————————————————————————')

#     for i in range(len(p)):
#         if len(p[i])>4:
#             print('mass 的最小子空间维数大于4,')
#         for k in range(len(p[i])):
#             p[i][(k+1) %len(p[i])]
#             print('第%d个不变子空间的基的变换,第%d个mass-->第%d个mass的系数是%d+%dj'%(i+1, p_conv[i][k], p_conv[i][(k+1) %len(p[i])], _mass_rep[p[i][(k+1) %len(p[i])],p[i][k]].real, _mass_rep[p[i][(k+1) %len(p[i])],p[i][k]].imag))
#         for l in range(len(p[i])-1):
#             for j in range(len(p[i])-l-1):     
#                 if (np.dot(Mass_term[p[i][j]],Mass_term[p[i][j+l+1]])-np.dot(Mass_term[p[i][j+l+1]],Mass_term[p[i][j]])==np.zeros(Mass_term[p[i][j]].shape)).all():
#                     print('第%d个不变子空间里面的第%d个基和第%d个基是对易的'%(i+1,j+1,j+l+2))
#                 elif (np.dot(Mass_term[p[i][j]],Mass_term[p[i][j+l+1]])+np.dot(Mass_term[p[i][j+l+1]],Mass_term[p[i][j]])==np.zeros(Mass_term[p[i][j]].shape)).all():
#                     print('第%d个不变子空间里面的第%d个基和第%d个基是反对易的'%(i+1,j+1,j+l+2))
#                 else:
#                     print('第%d个不变子空间里面的第%d个基和第%d个基即不对易也不反对易'%(i+1,j+1,j+l+2))
#         print('———————————————————————————————————————————————————————————')


















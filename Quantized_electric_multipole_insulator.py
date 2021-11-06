# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 19:30:49 2021

@author: 92012
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-



from mpl_toolkits import mplot3d
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time

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

## parameters
gamma = 1
lambda_ = 1
delta = 1

##Four bands system
bands = 4

begintime = time.time()
##The equation(6) in Science 357,61-66(2017)
def Hamiltonian_6(kx, ky):
    H = (gamma + lambda_ * np.cos(kx)) * Gamma_4
    H += lambda_ * np.sin(kx) * Gamma_3
    H += (gamma + lambda_ * np.cos(ky)) * Gamma_2
    H += lambda_ * np.sin(ky) * Gamma_1
    H += delta * Gamma_0
    return H

## The dispersion of equation(6)
kx = np.linspace(-np.pi, np.pi, 50)
ky = np.linspace(-np.pi, np.pi, 50)

Z1 = np.zeros((len(kx), len(ky)))
Z2 = np.zeros((len(kx), len(ky)))
Z3 = np.zeros((len(kx), len(ky)))
Z4 = np.zeros((len(kx), len(ky)))

for i in range(len(kx)):
    for j in range(len(ky)):
        _eigval, _eigvec = np.linalg.eigh(Hamiltonian_6(kx[i],ky[j]))
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

end_plot_time = time.time()
print('计算能谱时间：', end_plot_time-begintime)

##Energy spectrum of Eq.(6) with open boundary condition in x,y-direction
size_x = 2
size_y = 2

def open_boundary_xy_H_6(_size_x = size_x, _size_y = size_y):
    _open_H_full = np.zeros((bands *_size_x * _size_y, bands * _size_x * _size_y), dtype=complex)
    
    ##Gamma_4
    _open_H = np.zeros((_size_x * _size_y,   _size_x * _size_y), dtype=complex)
    for _i in range(_size_x-1):
        for _j in range(_size_y):
            _open_H[ _i + _j*_size_x, _i + 1 + _j*_size_x] = lambda_ * 0.5
            _open_H[ _i + 1 + _j*_size_x, _i + _j*_size_x] = lambda_ * 0.5
    
    _open_H += gamma * np.eye(_size_x*_size_y)        
    _open_H_full += np.kron(_open_H, Gamma_4)
    
    ##Gamma_3
    _open_H = np.zeros((_size_x * _size_y,   _size_x * _size_y), dtype=complex)
    for _i in range(_size_x-1):
        for _j in range(_size_y):
            _open_H[ _i + _j*_size_x, _i + 1 + _j*_size_x] = lambda_ * 0.5 * 1j
            _open_H[ _i + 1 + _j*_size_x, _i + _j*_size_x] = lambda_ * 0.5 * (-1j)
         
    _open_H_full += np.kron(_open_H, Gamma_3)
    
    ##Gamma_2
    _open_H = np.zeros((_size_x * _size_y,   _size_x * _size_y), dtype=complex)
    for _i in range(_size_x):
        for _j in range(_size_y-1):
            _open_H[ _i + _j*_size_x, _i + (1 + _j)*_size_x] = lambda_ * 0.5
            _open_H[ _i + (1 + _j)*_size_x, _i + _j*_size_x] = lambda_ * 0.5
    
    _open_H += gamma * np.eye(_size_x*_size_y)        
    _open_H_full += np.kron(_open_H, Gamma_2)
    
    ##Gamma_1
    _open_H = np.zeros((_size_x * _size_y,   _size_x * _size_y), dtype=complex)
    for _i in range(_size_x):
        for _j in range(_size_y-1):
            _open_H[ _i + _j*_size_x, _i + (1 + _j)*_size_x] = lambda_ * 0.5 * 1j
            _open_H[ _i + (1 + _j)*_size_x, _i + _j*_size_x] = lambda_ * 0.5 * (-1j)
         
    _open_H_full += np.kron(_open_H, Gamma_1)
    
    ##Gamma_0
    _open_H_full += delta * np.kron(np.eye(_size_x* _size_y), Gamma_0)
    
    
    return _open_H_full
        
parameter_change = np.linspace(-2,2,1)
eigenvalues = np.zeros(size_x*size_y*bands*len(parameter_change), dtype=complex)
gamma_lamdba = np.zeros(size_x*size_y*bands*len(parameter_change), dtype=complex)


lambda_ = 1
delta = 10**-3
for i in range(len(parameter_change)):
    print(i)
    gamma = lambda_ * parameter_change[i]
    eigval, eigvec = np.linalg.eigh(open_boundary_xy_H_6())
    eigenvalues[i*len(eigval):(i+1)*len(eigval)] = eigval.T
    gamma_lamdba[i*len(eigval):(i+1)*len(eigval)] = parameter_change[i]
    

fig = plt.figure()
plt.scatter(gamma_lamdba,eigenvalues,color='red',s=0.2)


    
end_open_boundary_spectrum_time = time.time()
print('求得开放边界能谱花费时间：', end_open_boundary_spectrum_time-end_plot_time)
    

# a=time.time()
# np.linalg.eig(np.random.rand(1000,1000))
# b=time.time()
# print(b-a)


#Fig2.B 计算非平庸态中的电荷密度

lambda_ = 1
delta = 10**-3
gamma = 10**-3

eigval, eigvec = np.linalg.eigh(open_boundary_xy_H_6())
eigvalues = np.real(eigval)
Eletronic_charge_density_orbits = np.zeros(bands*size_x*size_y,dtype=complex)
Eletronic_charge_density_unit_cell = np.zeros(size_x*size_y,dtype=complex)

for i in range(len(eigvalues)):
     if eigenvalues[i]>=0:
        Eletronic_charge_density_orbits += eigvec[:,i].conjugate()*eigvec[:,i]*(np.linalg.norm(eigvec[:,i]))**-2
       
        
for i in range(size_x*size_y):
    Eletronic_charge_density_unit_cell[i] += Eletronic_charge_density_orbits[4*i]
    Eletronic_charge_density_unit_cell[i] += Eletronic_charge_density_orbits[4*i+1]
    Eletronic_charge_density_unit_cell[i] += Eletronic_charge_density_orbits[4*i+2]
    Eletronic_charge_density_unit_cell[i] += Eletronic_charge_density_orbits[4*i+3]


X , Y = np.meshgrid(list(range(size_x)), list(range(size_y)))
Z = np.real(Eletronic_charge_density_unit_cell.reshape(size_x,size_y))

fig = plt.figure()
ax3 = plt.axes(projection='3d')
ax3.plot_surface(X,Y,Z,cmap='coolwarm')


end_eletronic_charge_density_time = time.time()
print('计算占据态的电荷密度花费时间：',end_eletronic_charge_density_time-end_open_boundary_spectrum_time)


##计算Wilson loop Fig.3(B)



lambda_ = 1
delta = 0
gamma = 0.3
wilson_step_kx = 1000

def Wilson_loop_x(ky,base_kx=-np.pi):
    step_wilson_loop = wilson_step_kx
    kx_wilson_loop = np.linspace(base_kx, base_kx+2*np.pi, step_wilson_loop)
    vector_array = np.zeros((step_wilson_loop*bands,bands),dtype=complex)
    
    for i in range(step_wilson_loop):
        if kx_wilson_loop[i] != base_kx+2*np.pi:
            _wilsonloop_eigval, _wilsonloop_eigvec = np.linalg.eigh(Hamiltonian_6(kx_wilson_loop[i], ky))
            vec_index = np.argsort(np.real(_wilsonloop_eigval))
            for k in range(bands):
                vector_array[i*bands+k,:]=_wilsonloop_eigvec[:,vec_index[k]].transpose()*np.linalg.norm(_wilsonloop_eigvec[:,vec_index[k]])**-1
        else:
            #print('Wilson loop首尾接上了，不存在gauge的问题，连接点在：',i)
            _wilsonloop_eigval, _wilsonloop_eigvec = np.linalg.eigh(Hamiltonian_6(base_kx, ky))
            for k in range(bands):
                vector_array[i*bands+k,:]=_wilsonloop_eigvec[:,vec_index[k]].transpose()*np.linalg.norm(_wilsonloop_eigvec[:,vec_index[k]])**-1
    
    Wilson_loop_single_band = np.ones(bands)
    F = np.zeros(bands,dtype=complex)
    Wilson_loop_occ_band= np.eye(2,2)
    F_occ = np.zeros((2,2),dtype=complex)
                
    for l in range(step_wilson_loop-1):
            #for k in range(bands):
            #    F[k] = np.dot(vector_array[(l+1)*bands+k,:].conjugate(),vector_array[l*bands+k,:].transpose())
            #print(F[1])
            for m in range(int(bands/2)):
                for n in range(int(bands/2)):
                    F_occ[m,n] = np.dot(vector_array[(l+1)*bands+m,:].conjugate(),vector_array[l*bands+n,:].transpose())
            Wilson_loop_single_band = Wilson_loop_single_band*F
           # print( np.linalg.norm( Wilson_loop_single_band))
            Wilson_loop_occ_band = np.dot(F_occ,Wilson_loop_occ_band)
            Wilson_eigval, Wilson_eigvec =np.linalg.eig(Wilson_loop_occ_band)
    
    return (np.log(Wilson_eigval)*(-1j)/2/np.pi,Wilson_eigvec)
    

# step_ky=600
# W_x_band0 = np.zeros(step_ky,dtype=complex)
# W_x_band1 = np.zeros(step_ky,dtype=complex)                    
        
# ky = np.linspace(-np.pi,np.pi,step_ky)

# for i in range(step_ky):
#     print(i)
#     W_x_band0[i] = Wilson_loop_x(ky[i])[0][0]*(-1j)/2/np.pi
#     W_x_band1[i] = Wilson_loop_x(ky[i])[0][1]*(-1j)/2/np.pi

# end_Wilson_loop_time = time.time()
# print('计算Wilson loop所花费时间：',end_eletronic_charge_density_time-end_open_boundary_spectrum_time)
# fig = plt.figure()
# plt.scatter(ky,W_x_band0,color='red',s=0.2)
# plt.scatter(ky,W_x_band1,color='red',s=0.2)


#计算nested wilson loop

nest_step_ky = 1000
def nest_wilson_loop(kx):
    _nest_step_ky=nest_step_ky
    ky = np.linspace(-np.pi,np.pi,_nest_step_ky)
    vector_array = np.zeros((_nest_step_ky*2, 4),dtype=complex)
 
    for i in range(_nest_step_ky):
        if ky[i] != np.pi:
            _nest_eigval,_nest_eigvec = Wilson_loop_x(ky[i],kx)
            _H_eigval, _H_eigvec =np.linalg.eigh(Hamiltonian_6(kx, ky[i]))
            vec_index = np.argsort(np.real(_nest_eigval))
            vector_array[2*i,:] =(_nest_eigvec[:,vec_index[0]][0]*_H_eigvec[:,0]+_nest_eigvec[:,vec_index[0]][1]*_H_eigvec[:,1])/np.linalg.norm((_nest_eigvec[:,vec_index[0]][0]*_H_eigvec[:,0]+_nest_eigvec[:,vec_index[0]][1]*_H_eigvec[:,1]))**-1
            vector_array[2*i+1,:] =(_nest_eigvec[:,vec_index[1]][0]*_H_eigvec[:,0]+_nest_eigvec[:,vec_index[1]][1]*_H_eigvec[:,1])/np.linalg.norm((_nest_eigvec[:,vec_index[1]][0]*_H_eigvec[:,0]+_nest_eigvec[:,vec_index[1]][1]*_H_eigvec[:,1]))**-1
        else:
            _nest_eigval,_nest_eigvec = Wilson_loop_x(-np.pi,kx)
            _H_eigval, _H_eigvec =np.linalg.eigh(Hamiltonian_6(kx, -np.pi))
            vec_index = np.argsort(np.real(_nest_eigval))
            vector_array[2*i,:] =(_nest_eigvec[:,vec_index[0]][0]*_H_eigvec[0]+_nest_eigvec[:,vec_index[0]][1]*_H_eigvec[:,1])/np.linalg.norm((_nest_eigvec[:,vec_index[0]][0]*_H_eigvec[:,0]+_nest_eigvec[:,vec_index[0]][1]*_H_eigvec[:,1]))**-1
            vector_array[2*i+1,:] =(_nest_eigvec[:,vec_index[1]][0]*_H_eigvec[0]+_nest_eigvec[:,vec_index[1]][1]*_H_eigvec[:,1])/np.linalg.norm((_nest_eigvec[:,vec_index[1]][0]*_H_eigvec[:,0]+_nest_eigvec[:,vec_index[1]][1]*_H_eigvec[:,1]))**-1
    
    nest_wilson_min = 1
    nest_wilson_plus = 1
    for l in range(_nest_step_ky-1):
        #print(np.linalg.norm( np.dot(vector_array[2*(l+1),:].conjugate(),vector_array[2*l,:])))
        #print(np.linalg.norm( np.dot(vector_array[2*(l+1)+1,:].conjugate(),vector_array[2*l+1,:])))
        nest_wilson_min = nest_wilson_min * np.dot(vector_array[2*(l+1),:].conjugate(),vector_array[2*l,:])
        nest_wilson_plus = nest_wilson_plus * np.dot(vector_array[2*(l+1)+1,:].conjugate(),vector_array[2*l+1,:])
    
    return (np.log(nest_wilson_min)*(-1j)/2/np.pi,np.log( nest_wilson_plus)*(-1j)/2/np.pi)





# P_min=0
# P_plus=0
fin_a=0
fin_b=0
l=1
for i in np.linspace(-np.pi,np.pi,10):
    #print(i)
    a,b=nest_wilson_loop(i)
    a= np.real(a) % 1
    b= np.real(b) % 1
    print('对于每个kx算出来的',a,b)
    fin_a=fin_a+a
    fin_b=fin_b+b
    print('平均的kx',fin_a/l,fin_b/l)
    l+=1
    
    


# P_min,P_plus = nest_wilson_loop()
# print(P_min,P_plus)



# for i in np.linspace(-np.pi,np.pi,5):
#     a,b=Wilson_loop_x(0,i)
#     print(np.linalg.norm(b[0]))
#     print(np.linalg.norm(b[1]))











    

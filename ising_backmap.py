import math
import random
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.interpolate import griddata
import pickle

#systemsize
N = 40

#temperature
t = 2.2

#number of flips till EQ
F = 100000

#number of flips between recording
RF = 5000

#number of records for one equilibrated system
rec = 100

white_block = [np.array([[1.0,1.0],[1.0,1.0]]),
               np.array([[-1.0,1.0],[1.0,1.0]]),
               np.array([[1.0,-1.0],[1.0,1.0]]),
               np.array([[1.0,1.0],[-1.0,1.0]]),
               np.array([[1.0,1.0],[1.0,-1.0]]),
               np.array([[1.0,1.0],[1.0,1.0]]),
               np.array([[-1.0,1.0],[1.0,1.0]]),
               np.array([[1.0,-1.0],[1.0,1.0]]),
               np.array([[1.0,1.0],[-1.0,1.0]]),
               np.array([[1.0,1.0],[1.0,-1.0]]),
               np.array([[-1.0,-1.0],[1.0,1.0]]),
               np.array([[1.0,1.0],[-1.0,-1.0]]),
               np.array([[1.0,-1.0],[-1.0,1.0]]),
               np.array([[1.0,-1.0],[1.0,-1.0]]),
               np.array([[-1.0,1.0],[-1.0,1.0]]),
               np.array([[-1.0,1.0],[1.0,-1.0]]),]
               
black_block = [np.array([[-1.0,-1.0],[-1.0,-1.0]]),
               np.array([[1.0,-1.0],[-1.0,-1.0]]),
               np.array([[-1.0,1.0],[-1.0,-1.0]]),
               np.array([[-1.0,-1.0],[-1.0,1.0]]),
               np.array([[-1.0,-1.0],[1.0,-1.0]]),
               np.array([[-1.0,-1.0],[-1.0,-1.0]]),
               np.array([[1.0,-1.0],[-1.0,-1.0]]),
               np.array([[-1.0,1.0],[-1.0,-1.0]]),
               np.array([[-1.0,-1.0],[-1.0,1.0]]),
               np.array([[-1.0,-1.0],[1.0,-1.0]]),
               np.array([[-1.0,-1.0],[1.0,1.0]]),
               np.array([[1.0,1.0],[-1.0,-1.0]]),
               np.array([[1.0,-1.0],[-1.0,1.0]]),
               np.array([[1.0,-1.0],[1.0,-1.0]]),
               np.array([[-1.0,1.0],[-1.0,1.0]]),
               np.array([[-1.0,1.0],[1.0,-1.0]]),]

def flip_block(s, s_renorm):
    x = np.random.randint(int(N/2))
    y = np.random.randint(int(N/2))

    rand = np.random.randint(16)
    if s_renorm[x,y] == 1.0:
        block = white_block[rand]
    else:
        block = black_block[rand]
    
    s[x*2:x*2+2,y*2:y*2+2] = block
    
    return s
    
def start_conf(s_renorm):
    s = np.ones((N,N))
    for i in range(0,int(N/2)):
        for j in range(0,int(N/2)):
            rand = np.random.randint(16)
            rand = 0
            if s_renorm[i,j] == 1.0:
                block = white_block[rand]
            else:
                block = black_block[rand]
            s[i*2:i*2+2,j*2:j*2+2] = block   
    return s



def convert_for_plot(sample):
    N_x = sample.shape[0]
    N_y = sample.shape[1]

    XX,YY,ZZ, W = [],[],[],[]
    for x in range(0,N_x):
        for y in range(0,N_y):
            for z in range(0,1):    
                XX.append(x+0.5)
                YY.append(y+0.5)
                ZZ.append(z+0.5)
                W.append(sample[x][y])
    return XX, YY, ZZ, W
def plot_sample(sample, name):
    XX, YY, ZZ, W = convert_for_plot(sample)    
    
    cm = plt.cm.get_cmap('gray')
    font = {     'size'   : 11}
    plt.rc('font', **font)
    
    plt.figure(figsize=(4,4))

    plt.scatter(XX, YY, c=W, vmin=-1.0, vmax=1.0, s=25, cmap=cm, edgecolors='none', marker = "s")
    #plt.set_xlim([0,40])
    #plt.set_ylim([0,40])
    plt.xticks([])
    plt.yticks([])
    #plt.set_title('Real Sample', fontweight='bold')    
    #plt.colorbar(im,ticks=[-1, 0, 1])

    plt.savefig("pic/"+name+".pdf", bbox_inches='tight')
    plt.close()
    
    
    
def energy(a,x,y):
	left = x-1
	right = x+1
	up = y+1
	down = y-1
	if x == 0:
		left = N-1
	if x == N-1:
		right = 0
	if y == 0:
		down = N-1
	if y == N-1:
		up = 0
	return -(a[x][y]*a[right][y]+a[x][y]*a[left][y]+a[x][y]*a[x][up]+a[x][y]*a[x][down])
#frange fuer range data 
 
def energy_system(lattice):
    #print(lattice.shape)
    en = 0.0
    for i in range(0,N):
        for j in range(0,N):
            en = en + energy(lattice, i,j)
    en = en / 2
    return en


def block_spin(sample):
    renorm = np.zeros((int(N/2),int(N/2)))
    for i in range(0,N,2):
        for j in range(0,N,2):
            s = np.sign(sample[i,j]+ sample[i+1,j]+sample[i,j+1]+sample[i+1,j+1])
            if s == 0:
                s = np.random.randint(2)*2 -1
            renorm[int(i/2), int(j/2)] = s
    return renorm

def frange(start):
    temps = [start]
    for i in range(1,35):
        temps.append(start+i*0.1)
    return temps
print("Using temperatues:")
#temps = [2.2]
temps = [2.2,2.269,2.3,2.4,2.1,1.5,2.0,2.5,3.0]
#temps = [2.3]
#temps = frange(0.1)
print(temps)



############# Equilibration #######################
for t in temps:
    for b in range(0,10):
        	print(t)
        	file = open('data_N8_T'+str(t)+'.pickle', 'rb')
        	data = pickle.load(file)
        	file.close()
        	(N_data, dim, N_x, N_y) =  data.shape
        	rand =  np.random.randint(N_data)
        	renorm_sample = np.reshape(data[rand], (N_x, N_y))
        	plot_sample(renorm_sample,"start") 
        	print(energy_system(renorm_sample)/1600)
        	renorm_sample = block_spin(renorm_sample)
         	#print(renorm_sample)
        	#initialize random spin states
        	s = start_conf(renorm_sample)
        	print(s)
        	print(energy_system(np.ones((40,40)))/1600) 
        	s_energy = energy_system(s)
        	for i in range(0,F):
                 if i % 50000 == 0:
                     print(i)
                     print(s_energy/1600)
                     plot_sample(s,"backmap")
                 s_old = np.copy(s)
                 s = flip_block(s, renorm_sample)
                 new_energy = energy_system(s)
                 energy_diff = new_energy - s_energy
                 if energy_diff > 0:
                     if random.uniform(0,1) > np.exp(-energy_diff / t):
                         s = np.copy(s_old)
                     else:
                         s_energy = new_energy
                         #print("ja")
                 else:
                     s_energy = new_energy
                     #print("ja")
        	###############    Recording    ###### ##########		
        	print("start recording")
        	for r in range(0, rec):
        		print(r)
        		data = np.reshape(s, (1,1,N,N))
        		with open('blockmap'+str(b)+'_N'+str(N)+'_T'+str(t)+'.pickle', 'a') as f:
        			pickle.dump(data, f, protocol=2)	
        
                	for i in range(0,RF):
                         s_old = s[:,:]
                         s_old = np.copy(s)
                         s = flip_block(s, renorm_sample)
                         new_energy = energy_system(s)
                         energy_diff = new_energy - s_energy
                         if energy_diff > 0:
                             if random.uniform(0,1) > np.exp(-energy_diff / t):
                                 s = np.copy(s_old)
                             else:
                                 s_energy = new_energy
                         #print("ja")
                         else:
                             s_energy = new_energy

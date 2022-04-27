import numpy as np
import torch
import matplotlib.pyplot as plt
f=open("./Data_3GHz_Hy_probe10_1cm.txt","r+")
array=[]
for line in f:
    line = line.replace('i', 'j').split()
    if line: 
            line=[complex(i) for i in line]
            array.append(line)
A=torch.tensor(array,dtype=torch.cfloat)
Hx=A[:,1]
Hx=Hx.view(len(Hx),1)
L=36
nx=60
ny=60
data=torch.rand(nx,ny,L, dtype=torch.cfloat)
n=0
for k in range(0,L):
    for x in range(0,nx):
        for y in range(0,ny):
            data[x,y,k]=Hx[n]
            n=n+1
CF=torch.rand(nx,ny,nx,ny,L, dtype=torch.cfloat)
for t in range(0,L):
    for x1 in range(0,nx):
        for y1 in range(0,ny):
            for x2 in range(0,nx):
                for y2 in range(0,ny):
                    CF[x1,y1,x2,y2,t]=data[x1,y1,t]*torch.conj(data[x2,y2,t])
R1=torch.rand(nx,ny,nx,ny, dtype=torch.cfloat)
for x1 in range(0,nx):
    for y1 in range(0,ny):
        for x2 in range(0,nx):
            for y2 in range(0,ny):
                R1[x1,y1,x2,y2]=torch.mean(torch.squeeze(CF[x1,y1,x2,y2,:]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation 
import os.path

BASE = (0,0)
LEN1 = 1
LEN2 = 1
M1 = 2
M2 = 2
g = 9.8
tmax = 20 # Maximum time for the animation to render
fps = 120
tot_frames = int(fps*tmax) # Total number of frames
ndps = 1024

init_state = [(np.pi, np.pi - 0.00001*i, 0, 0) for i in np.arange(1,ndps+1)]

f = 0
filepath = f'dp1000\dp[{tmax}][{fps}][{ndps}][{LEN1}-{LEN2}][{M1}-{M2}].mp4'
while os.path.isfile(filepath):
    f = f + 1
    filepath = f'dp1000\dp[{tmax}][{fps}][{ndps}][{LEN1}-{LEN2}][{M1}-{M2}]({f}).mp4'

plt.style.use('dark_background')   
fig, ax = plt.subplots()

def system(init_state, t):
    tta1, tta2, omga1, omga2 = init_state
    
    a = (LEN2/LEN1)*np.cos(tta1-tta2)*(M2/(M1+M2))
    b = -(LEN2/LEN1)*(M2/(M1+M2))*np.sin(tta1-tta2)*(omga2**2) - (g/LEN1)*np.sin(tta1)
    c = (LEN1/LEN2)*np.cos(tta1-tta2)
    d = -(LEN1/LEN2)*np.sin(tta1-tta2)*(omga1**2) - (g/LEN2)*np.sin(tta2)
    
    # tta1_dot = omga1
    # tta2_dot = omga2
    omga1_dot = (b - a*d)/(1 - a*c)
    omga2_dot = (d - c*b)/(1 - a*c)
    
    return (omga1, omga2, omga1_dot, omga2_dot)

def get_bobs(tta):
    p1 = ( LEN1*np.sin(tta[0]), -LEN1*np.cos(tta[0]) )
    p2 = ( LEN1*np.sin(tta[0])+LEN2*np.sin(tta[1]), -LEN1*np.cos(tta[0])-LEN2*np.cos(tta[1]) )
    return (p1, p2)

CGRW = list(range(256))
CDIM = list(range(255, -1, -1))
CSTL = [0]*256
CSTH = [255]*256

colorspect = [f'#{int(i):02X}{int(j):02X}{int(k):02X}' for i, j, k in zip(CSTH+CDIM+CSTL+CGRW, CGRW+CSTH+CDIM+CSTL, CSTL+CSTL+CGRW+CSTH)]
pends = [ax.plot([0], [0], marker = 'o', color=colorspect[-i-1])[0] for i in range(ndps)]

time_instances = np.linspace(0, tmax, tot_frames)

print('Solving... ')
ttas = [odeint(system, init_state[i], time_instances) for i in range(ndps)]
ps = [[get_bobs(tta) for tta in ttas[i]] for i in range(ndps)]

def animate(val):
    global t
    
    t = int(val)
    print(t) # Printing the frame number (Commenting would result in a slightly faster render, but you won't notice if the render is stuck)
    
    for i in range(ndps):   
        z = list(zip(BASE, ps[i][t][0], ps[i][t][1]))
        pends[i].set_xdata(z[0])
        pends[i].set_ydata(z[1])
    
    return pends


ax.set_aspect(1)
xlim = (LEN1+LEN2+(LEN1*0.1))
ylim = xlim
ax.set(xlim=(-xlim, xlim),
       ylim=(-ylim, ylim))

print("Writing to :", filepath)

anim = FuncAnimation(fig, animate, frames=tot_frames, interval=int((1000*tmax)/tot_frames), blit=True)
anim.save(filepath, writer = 'ffmpeg', fps = fps)

print(f'Rendered to: {filepath}\nTime: {tmax}\nfps: {fps}') 

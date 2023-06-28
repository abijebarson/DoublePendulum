import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import odeint

BASE = (0,0)
LEN1 = 1
LEN2 = 0.7
M1 = 2
M2 = 2
g = 9.8
tmax = 50
tres = 1000
onion = 10

init_state = (np.pi, np.pi, 0, 0)

plt.style.use('dark_background')   
fig, ax = plt.subplots()

# - - - - -

def system(init_state, t):
    tta1, tta2, omga1, omga2 = init_state
    
    a = (LEN2/LEN1)*np.cos(tta1-tta2)*(M2/(M1+M2))
    b = -(LEN2/LEN1)*(M2/(M1+M2))*np.sin(tta1-tta2)*(omga2**2) - (g/LEN1)*np.sin(tta1)
    c = (LEN1/LEN2)*np.cos(tta1-tta2)
    d = -(LEN1/LEN2)*np.sin(tta1-tta2)*(omga1**2) - (g/LEN2)*np.sin(tta2)
    
    omga1_dot = (b - a*d)/(1 - a*c)
    omga2_dot = (d - c*b)/(1 - a*c)
    
    return (omga1, omga2, omga1_dot, omga2_dot)

def get_bobs(tta):
    p1 = ( LEN1*np.sin(tta[0]), -LEN1*np.cos(tta[0]) )
    p2 = ( LEN1*np.sin(tta[0])+LEN2*np.sin(tta[1]), -LEN1*np.cos(tta[0])-LEN2*np.cos(tta[1]) )
    return (p1, p2)

plt.subplots_adjust(bottom=0.1)

axtslide = plt.axes([0.25, 0.05, 0.65, 0.03])
tslide = Slider(axtslide, 'Time',valmin=0, valmax=tmax, valinit=0)

colorspect = [f'#{int(i):02X}2955' for i in np.linspace(0, 100, onion)]
pendh = [ax.plot([0], [0], marker = 'o', color=colorspect[i]) for i in range(onion)]
pend = ax.plot([0], [0], marker = 'o', color='#EE0C99')

time_instances = np.linspace(0, tmax, tres+1)
ttas = odeint(system, init_state, time_instances)
ps = [get_bobs(tta) for tta in ttas]

def update(val):
    global z1, z, t
    
    t = int(tslide.val * (tres/tmax))      
    
    for i in range(onion):
        if (t-i-1) < 0: 
            pendh[i][0].set_xdata([0])
            pendh[i][0].set_ydata([0])
            continue
        zh = list(zip(BASE, ps[t-i-1][0], ps[t-i-1][1]))
        print(i, zh)
        pendh[i][0].set_xdata(zh[0])
        pendh[i][0].set_ydata(zh[1])
        
    z = list(zip(BASE, ps[t][0], ps[t][1]))
    pend[0].set_xdata(z[0])
    pend[0].set_ydata(z[1])

update(None)
tslide.on_changed(update)
# - - - - - 

ax.set_aspect(1)
xlim = (LEN1+LEN2+(LEN1*0.1))
ylim = xlim
ax.set(xlim=(-xlim, xlim),
       ylim=(-ylim, ylim))
plt.show()
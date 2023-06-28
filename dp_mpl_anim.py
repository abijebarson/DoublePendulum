import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation 
import os.path

# Double Pendulum (Lagrangian Equation of Motion solved numerically):
# Things to notice: 
#   Changing the intial state even slightly would result in major variation in the output
#   Since this is a numerical solve, even the change in time difference between two solves, give variation in output
#   Give more onions, remove the marker='o' and change colors as per your wish to notice beautiful patterns

# Initial variables (1 for the top pendulum, 2 for bottom pendulum)
BASE = (0,0)
LEN1 = 1
LEN2 = 0.7
M1 = 2
M2 = 2
g = 9.8
tmax = 60 # Maximum time for the animation to render
fps = 20
tot_frames = int(fps*tmax) # Total number of frames
onion = 10 # History of pendulum shown as onions (general animation reference)

# INITIAL STATE: (Angle1, Angle2, AngularVelocity1, AngularVelocity2)
# Angle1 => angle of the top bob/pendulum with respect to the vertical axis
# Angle2 => angle of the bottom bob/pendulum with respect to the vertical axis
# AngularVelocity1 => angular velocity of top bob/pendulum
# AngularVelocity2 => angular velocity of bottom bob/pendulum
init_state = (np.pi, np.pi + 0.00000001, 0, 0) 

f = 0
filename = f'dp[{tmax}][{fps}][{LEN1}-{LEN2}][{M1}-{M2}][{init_state[0]:.2f}-{init_state[1]:.2f}].mp4'
while os.path.isfile(filename):
    f = f + 1
    filename = f'dp[{tmax}][{fps}][{LEN1}-{LEN2}][{M1}-{M2}][{init_state[0]:.2f}-{init_state[1]:.2f}]({f}).mp4'
print("Writing to :", filename)

plt.style.use('dark_background')   
fig, ax = plt.subplots()

# - - - - -

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

colorspect = [f'#{int(i):02X}2955' for i in np.linspace(0, 100, onion)]

# I'm keeping the onion pendulums and the real pendulum separate for more independent customizability
pendh = [ax.plot([0], [0], marker = 'o', color=colorspect[i]) for i in range(onion)]
pend, = ax.plot([0], [0], marker = 'o', color='#EE0C99', linewidth=2)

time_instances = np.linspace(0, tmax, tot_frames)
ttas = odeint(system, init_state, time_instances)
ps = [get_bobs(tta) for tta in ttas]

def animate(val):
    global zh, z, t
    
    t = int(val)
    print(t) # Printing the frame number (Commenting would result in a slightly faster render, but you won't notice if the render is stuck)
    
    for i in range(onion):
        if (t-i-1) < 0: 
            pendh[i][0].set_xdata([0])
            pendh[i][0].set_ydata([0])
            continue
        zh = list(zip(BASE, ps[t-i-1][0], ps[t-i-1][1]))
        pendh[i][0].set_xdata(zh[0])
        pendh[i][0].set_ydata(zh[1])
        
    z = list(zip(BASE, ps[t][0], ps[t][1]))
    pend.set_xdata(z[0])
    pend.set_ydata(z[1])
    
    return pend, 

# - - - - - 

ax.set_aspect(1)
xlim = (LEN1+LEN2+(LEN1*0.1))
ylim = xlim
ax.set(xlim=(-xlim, xlim),
       ylim=(-ylim, ylim))

anim = FuncAnimation(fig, animate, frames=tot_frames, interval=int((1000*tmax)/tot_frames), blit=True)
anim.save(filename, writer = 'ffmpeg', fps = fps)

print(f'Rendered to: {filename}\nTime: {tmax}\nfps: {fps}')
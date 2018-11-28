
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import librosa
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# In[3]:


filename='flow/20170622-14-37-19-582_001jun_IPEC_006_1.csv'
data=pd.read_csv(filename)

# convert timestemps in second
data.timestamp=data.apply(lambda x: x.timestamp*1e-6,axis=1)

# add speed
data['speed']=np.sqrt(data.VX**2+data.VY**2)


# In[4]:


last_time=data.iloc[-1].timestamp
windows_shift=0.01
n_fft=512
n_frames=math.ceil(last_time/windows_shift)
print('Number of frames = ',n_frames)
times=[windows_shift*i for i in range(n_frames)]
indices=[i for i in range(n_frames-1)]
data['frame_index']=pd.cut(data['timestamp'],times,labels=indices,include_lowest=True)


# In[5]:


frames=data.groupby('frame_index',observed=True)
non_empty_frames=[k for k in frames.indices]

all_frames=[i for i in range(n_frames)]
# In[17]:


fig ,ax = plt.subplots(figsize=(8,10))

def init_plot():
    xmax=310
    ymax=240

    x_ticks=[i for i in range(0,xmax,10)]
    y_ticks=[i for i in range(0,ymax,10)]

    ax.set_xticks(ticks=x_ticks)
    ax.set_yticks(ticks=y_ticks)

    #plt.colorbar()
    ax.grid()
    #ax.set_xlim(0,xmax)
    #ax.set_ylim(0,ymax)

    ax.set_xlim(80,180)
    ax.set_ylim(0,ymax)

def print_single_frame(frame_index):
    plt.cla()
    #print(frame_index)
    xmax=310
    ymax=240

    x_ticks=[i for i in range(0,xmax,10)]
    y_ticks=[i for i in range(0,ymax,10)]

    ax.set_xticks(ticks=x_ticks)
    ax.set_yticks(ticks=y_ticks)

    #plt.colorbar()
    ax.grid()
    ax.set_xlim(80,180)
    ax.set_ylim(0,ymax)
    ax.set_title('frame number = {:d}   time={:.3f} s'.format(frame_index,frame_index*0.01))

    x=data.X.values
    y=data.Y.values

    ax.scatter(x,y,color='r',s=1.0)


    if frame_index in non_empty_frames:
        x=frames.get_group(frame_index)['X']
        y=frames.get_group(frame_index)['Y']
        vx=frames.get_group(frame_index)['VX']
        vy=frames.get_group(frame_index)['VY']
        
        X=x.values
        Y=y.values
        U=vx.values * windows_shift *10
        V=vy.values * windows_shift *10

        #U = U / np.sqrt(U**2 + V**2) * 10;
        #V = V / np.sqrt(U**2 + V**2) * 10;
        
        ax.scatter(x,y)
        ax.quiver(X,Y,U,V,angles='xy',scale_units='xy', scale=1)
        #ax.quiver(X,Y,U,V,scale=600)





Writer = animation.writers['ffmpeg']
writer = Writer(fps=100)



ani = FuncAnimation(fig, print_single_frame, frames=all_frames,init_func=init_plot)

ani.save('test.mp4', writer=writer)
#plt.show()

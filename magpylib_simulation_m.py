import magpylib as mag3
import numpy as np
import matplotlib.pyplot as plt
from magpylib.magnet import Cylinder, Box, Sphere
from magpylib import Collection


def pole_simulation(Main_mag_s,Gap_size,Gap_location,scan_point_num,sscan_len,HallSensor_h):
    
     #10~190
    scan_len=sscan_len*2 
    Gap_location=Gap_location/100 
    x0=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,scan_len),position=(0,0,scan_len/2))
    
    x1=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,scan_len*Gap_location),position=(50,0,(scan_len*Gap_location)/2))
    
    ff=scan_len-scan_len*Gap_location
    x2=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,ff),position=(50,0,scan_len*Gap_location+(ff)/2+Gap_size))
    
    x3=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,scan_len),position=(100,0,scan_len/2))
    
    
    c=Collection(x0,x1,x2,x3)
        
    c.rotate_from_angax(90, 'x',(50,0,scan_len/2))
    c.move((0,(scan_len/2),(-scan_len/2)))
    
    
    xs=np.linspace(0, 100,100)
    ys=np.linspace((scan_len*0.25),(scan_len*0.75),scan_point_num)
    
    POS=np.array([(x,y,HallSensor_h) for y in ys for x in xs ])
    
    Bs=c.getB(POS).reshape(scan_point_num,100,3)
    
    '''
    X,Y=np.meshgrid(xs,ys)
    
    
    
    fig=plt.figure(figsize=(9,5))
    
    ax1=fig.add_subplot(221,projection='3d')
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)
    
    
    c.display(axis=ax1, show_path=True, show_direction=True)
    
    X,Y=np.meshgrid(xs,ys)
    U,V=Bs[:,:,0],Bs[:,:,1]                    #0:x 1:Z 2:Y
    ax2.streamplot(X,Y,U,V, color=np.log(U**2+V**2))
    
    U,V=Bs[:,:,1],Bs[:,:,2]                    #0:x 1:Z 2:Y
    ax3.streamplot(X,Y,U,V, color=np.log(U**2+V**2))
    
    U,V=Bs[:,:,0],Bs[:,:,2]                    #0:x 1:Z 2:Y
    ax4.streamplot(X,Y,U,V, color=np.log(U**2+V**2))
    
    plt.show()
    '''
    return Bs


'''
Main_mag_s=100
HallSensor_h=2

scan_len=500
scan_point_num=2100

Gap_size=-100
Gap_location=50   #inverse 75~25

'''
#Bs,X,Y=pole_simulation(Main_mag_s,Gap_size,Gap_location,scan_point_num,scan_len,HallSensor_h)


'''
################################################################################

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"rowspan": 3,"type": 'surface'},{}],
           [None,{}],
           [None, {}]],

    subplot_titles=("3D","X data plot", "Y data plot","Z data plot"))

fig.add_trace(
    go.Surface( x=X, y=Y,z=Bs[:,:,0], colorscale='Viridis', showscale=False),
    row=1, col=1)

fig.add_trace(
    go.Surface( x=X, y=Y,z=Bs[:,:,1], colorscale='Viridis', showscale=False),
    row=1, col=1)

fig.add_trace(
    go.Surface( x=X, y=Y,z=Bs[:,:,2], colorscale='Viridis', showscale=False),
    row=1, col=1)


# Add traces, one for each slider step
for step1 in range(10,90): #기준 0~250
    step=10
    fig.append_trace(go.Scatter3d(x=[step-10,step-10], y=[0,2100], z=[1,1],visible=False,name="Time = " + str(step),
                                  line = dict(color='blue', width=2)), row=1, col=1)
    fig.append_trace(go.Scatter3d(x=[step+10,step+10], y=[0,2100], z=[1,1],visible=False,name="Time = " + str(step),
                                  line = dict(color='red', width=2)), row=1, col=1)
    fig.append_trace(go.Scatter3d(x=[step,step], y=[0,2100], z=[1,1],visible=False,name="Time = " + str(step),
                                  line = dict(color='black', width=4)), row=1, col=1)
    
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step-10,0], visible=False,name="Time = " + str(step),
                                line = dict(color='blue', width=2)), row=1, col=2)
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step+10,0], visible=False,name="Time = " + str(step),
                                line = dict(color='red', width=2)), row=1, col=2)
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step,0], visible=False,name="Time = " + str(step),
                                line = dict(color='black', width=4)), row=1, col=2)
    
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step-10,1], visible=False,name="Time = " + str(step),
                                line = dict(color='blue', width=2)), row=2, col=2)
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step+10,1], visible=False,name="Time = " + str(step),
                                line = dict(color='red', width=2)), row=2, col=2)
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step,1], visible=False,name="Time = " + str(step),
                                line = dict(color='black', width=4)), row=2, col=2)
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step-10,2], visible=False,name="Time = " + str(step),
                                line = dict(color='blue', width=2)), row=3, col=2)
    
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step+10,2], visible=False,name="Time = " + str(step),
                                line = dict(color='red', width=2)), row=3, col=2)
    fig.append_trace(go.Scatter(x=Y[:,1], y=Bs[:,step,2], visible=False,name="Time = " + str(step),
                                line = dict(color='black', width=4)), row=3, col=2)


fig.data[1].visible = True
fig.data[2].visible = True
fig.data[3].visible = True
print(len(fig.data))
# Create and add slider

fig.show()
'''
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Read data #
print("\n- Reading data -")
fin = h5.File("output.h5", "r")
fin_sg = h5.File("output_sg.h5", "r")
n_data = fin.attrs["n_data"]
zi_sg = fin_sg.attrs["zi_sg"]
zf_sg = fin_sg.attrs["zf_sg"]
data = np.array([fin["data"][str(i)] for i in range(n_data)])
data_sg = np.array([fin_sg["data"][str(i)] for i in range(n_data)])
time = fin["time"][:]

print("\n- Finished reading data -")
print(" Number of time steps: {}".format(n_data))

# Figure #
fig, ax = plt.subplots(figsize=(15, 4))
vmin = data.min()
vmax = data.max()
fps = 30

print("\n- Plotting data -")
print(" Min value: {:.2E}".format(vmin))
print(" Max value: {:.2E}".format(vmax))

def init():
    global text, img, sim_start
    
    data_plt = np.concatenate((
        np.repeat(np.repeat(data[0, :, :zi_sg-1], 2, axis=0), 2, axis=1),
        data_sg[0, :, :],
        #np.zeros(data_sg[0, :, :].shape),
        np.repeat(np.repeat(data[0, :, zf_sg:], 2, axis=0), 2, axis=1)
    ), axis=1)
    
    img = ax.imshow(data_plt, vmin=vmin, vmax=vmax)
    ax.axvline(x=(zi_sg-2)*2 + 1.5, color="white")
    ax.axvline(x=(zi_sg-2)*2 + 2*(zf_sg-zi_sg)+1 + 1.5, color="white")
    if not sim_start:
        fig.colorbar(img, ax=ax)
        text = ax.text(-.5, -1, "n = {:5} --- t = {:.2E}".format(0, time[0]))
    else:
        text.set_text("n = {:5} --- t = {:.2E}".format(0, time[0]))
    
    
    sim_start = True

def inter(i):
    global text, img
    
    data_plt = np.concatenate((
        np.repeat(np.repeat(data[i, :, :zi_sg-1], 2, axis=0), 2, axis=1),
        data_sg[i, :, :],
        #np.zeros(data_sg[0, :, :].shape),
        np.repeat(np.repeat(data[i, :, zf_sg:], 2, axis=0), 2, axis=1)
    ), axis=1)
    
    img.set_data(data_plt)
    text.set_text("n = {:5} --- t = {:.2E}".format(i, time[i]))


sim_start= False
anim = FuncAnimation(fig, inter, np.arange(1, n_data), interval=1000/fps, init_func=init)
plt.show()

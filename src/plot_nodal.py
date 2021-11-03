import matplotlib.pyplot as plt
import numpy as np
from Fourier.functions import FT_inhomogeneous, ppw_to_freq, ppw_to_sigma
from Math import gaussian, gaussian_derivative

name = "_sinLTS"

# Read data #
print("\n- Reading data -")

def readData(filename, howmany=None):
    return np.array(
        list(map(lambda i: list(map(lambda j: float(j), i.split())), open(filename, "r").readlines()[:-1][:howmany]))
    ).swapaxes(0, 1)

finalTime = 2000


time_0, data_0 = readData("output_nodal"+name+"_nosubgrid_0.dat", finalTime)
time_1, data_1 = readData("output_nodal"+name+"_nosubgrid_1.dat", finalTime)

time_sg1_0, data_sg1_0 = readData("output_nodal"+name+"_0.dat", finalTime)
time_sg1_1, data_sg1_1 = readData("output_nodal"+name+"_1.dat", finalTime)

time_teor = np.linspace(0, 1e-7, 10000)
data_teor = gaussian(time_teor, 0.01e-6, 1e-9)

fout = open("input"+name+".dat", "w")
for i in range(len(time_teor)):
    fout.write("{:.5E} {:.5E}\n".format(time_teor[i], data_teor[i]))
fout.close()

ppw = np.logspace(0, 6, 10000)
freq = ppw_to_freq(ppw)

plt.figure()
plt.plot(time_0, data_sg1_0, label="Reflexion bruta")
plt.plot(time_0, data_0, label="Reflexion sin subgrid")
plt.plot(time_0, data_sg1_0 - data_0, label="Reflexion neta")
plt.xlabel("Time (s)")
plt.legend()
plt.figure()
plt.plot(time_1, data_1, label="Transmision sin subgrid")
plt.plot(time_teor, data_teor, label="Input teorico")
plt.xlabel("Time (s)")
plt.show()


FT_1 = np.abs(FT_inhomogeneous(time_1, data_1, freq))
FT_reflection = np.abs(FT_inhomogeneous(time_0, data_sg1_0-data_0, freq))
FT_reflection_alt = np.abs(FT_inhomogeneous(time_0, data_sg1_0, freq))
diff = 20*np.log10(FT_reflection/FT_1)
diff_alt= 20*np.log10(FT_reflection_alt/FT_1)

plt.figure()
plt.semilogx(ppw, diff, label="Reflexion con resta")
plt.semilogx(ppw, diff_alt, label="Reflexion sin resta")
plt.xlabel("ppw")
plt.ylabel("Reflection (dB)")
plt.grid()
plt.legend()

plt.figure()
plt.semilogx(ppw, FT_1, label="Input")
plt.xlabel("ppw")
plt.grid()
plt.legend()

plt.show()

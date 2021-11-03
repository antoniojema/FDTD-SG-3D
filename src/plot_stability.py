import matplotlib.pyplot as plt
import numpy as np

# Read data #
print("\n- Reading data -")

def readData(filename, howmany=None):
    return np.array(
        list(map(lambda i: list(map(lambda j: float(j), i.split())), open(filename, "r").readlines()[:-1][:howmany]))
    ).swapaxes(0, 1)

time, data = readData("output_stability.dat")

plt.figure("Stability")
plt.plot(time, data)
plt.xlabel("Time (seconds)")
plt.show()

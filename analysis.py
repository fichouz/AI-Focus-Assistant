import matplotlib.pyplot as plt
import numpy as np

# sestavi_podatke()

def prikazi_signal(signal, Fvz, naslov="", ylabel="", startInd=None, endInd=None):
    if len(signal) == 0:
        print("Ni podatkov.")
        return

    if startInd is not None and endInd is not None:
        signal = signal[startInd:endInd]

    t = np.arange(len(signal)) / Fvz
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal[:, 0], label='x')
    plt.plot(t, signal[:, 1], label='y')
    plt.plot(t, signal[:, 2], label='z')
    plt.title(f"{naslov} (Fvz = {Fvz:.2f} Hz)")
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# main
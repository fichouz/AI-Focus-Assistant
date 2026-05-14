import matplotlib.pyplot as plt
import numpy as np
from parser import decode_file 

def sestavi_podatke(seznam_paketov):
    all_data = []
    timestamps = []

    for p in seznam_paketov:
        all_data.append(p.data)
        timestamps.append(p.ts)

    if len(all_data) == 0:
        return 0, np.array([])

    signal = np.vstack(all_data)
    timestamps = np.array(timestamps)
    dt = np.diff(timestamps) / 1000.0  

    Tavg = np.mean(dt)
    Nvz = np.mean([len(p.data) for p in seznam_paketov])
    Fvz = Nvz / Tavg if Tavg > 0 else 0

    return Fvz, signal

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
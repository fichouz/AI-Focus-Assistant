import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from analysis import sestavi_podatke

PORT = "COM7"
BAUD = 115200
FILENAME = "sample.bin"

buffer = b""
paketi = []
running = True

def live_plot():
    print("[OK] Starting plot window...")
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    while running:
        if len(paketi) < 5:
            plt.pause(0.1)
            continue

        for ax in axs:
            ax.clear()

        for idx, sensor_id in enumerate([1, 2, 3]):
            try:
                Fvz, sig, name, unit = sestavi_podatke(paketi, sensor_id)

                if sig.size == 0 or Fvz == 0:
                    continue

                N = 300
                s = sig[-N:]

                t = np.arange(len(s)) / Fvz

                axs[idx].plot(t, s[:, 0], label="x")
                axs[idx].plot(t, s[:, 1], label="y")
                axs[idx].plot(t, s[:, 2], label="z")

                axs[idx].set_title(f"{name} (Fvz={Fvz:.1f} Hz)")
                axs[idx].set_ylabel(unit)
                axs[idx].legend()
                axs[idx].grid()

            except Exception as e:
                print("[PLOT ERROR]", e)

        axs[-1].set_xlabel("time (s)")

        plt.tight_layout()
        plt.pause(0.05)